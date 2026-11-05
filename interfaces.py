"""插件契约层（Plugin Contract）。

这是内核与插件之间**唯一**的耦合点：
    内核只认识 AnalysisPlugin / AnalysisContext / Layer 三样东西，
    插件只依赖本文件，永远不需要 import core。

依赖：numpy、cv2。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Tuple

import cv2
import numpy as np

__all__ = [
    "PluginError",
    "PluginContractError",
    "AnalysisContext",
    "Layer",
    "AnalysisPlugin",
    "render_layers",
]


# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #
class PluginError(Exception):
    """插件体系异常基类。"""


class PluginContractError(PluginError):
    """插件违反了契约（taskType 缺失/重复、analyze 返回值非法等）。"""


# --------------------------------------------------------------------------- #
# 上下文：内核只负责打包，插件只负责读取
# --------------------------------------------------------------------------- #
@dataclass
class AnalysisContext:
    """一次分析请求的全部输入。

    image      : BGR 三通道原图 (H, W, 3) uint8
    gt_mask    : 真值掩膜 (H, W) bool
    pred_mask  : 预测掩膜 (H, W) bool
    params     : 前端传入的参数（插件自行读取自己关心的键）
    meta       : 插件间共享的中间结果缓存（例如边界插件复用高亮插件的轮廓）
    """

    image: np.ndarray
    gt_mask: np.ndarray
    pred_mask: np.ndarray
    params: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image = self._as_bgr(self.image)
        self.gt_mask = self._as_mask(self.gt_mask)
        self.pred_mask = self._as_mask(self.pred_mask)
        self.params = dict(self.params or {})
        self.meta = dict(self.meta or {})

        h, w = self.gt_mask.shape[:2]
        if self.pred_mask.shape[:2] != (h, w):
            raise PluginContractError(
                f"GT 与 Pred 尺寸不一致: {self.gt_mask.shape[:2]} vs {self.pred_mask.shape[:2]}"
            )
        if self.image.shape[:2] != (h, w):
            raise PluginContractError(
                f"原图与掩膜尺寸不一致: {self.image.shape[:2]} vs {(h, w)}"
            )

    # -- 归一化工具 ---------------------------------------------------------- #
    @staticmethod
    def _as_bgr(image: np.ndarray) -> np.ndarray:
        if image is None:
            raise PluginContractError("原图为空（image=None）")
        image = np.asarray(image)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim != 3 or image.shape[2] < 3:
            raise PluginContractError(f"原图必须是 HxW 或 HxWx3，收到 {image.shape}")
        return np.ascontiguousarray(image[:, :, :3])

    @staticmethod
    def _as_mask(mask: np.ndarray) -> np.ndarray:
        if mask is None:
            raise PluginContractError("掩膜为空（mask=None）")
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.ndim != 2:
            raise PluginContractError(f"掩膜必须是二维，收到 {mask.shape}")
        return mask > 0

    # -- 便捷属性 ------------------------------------------------------------ #
    @property
    def shape(self) -> Tuple[int, int]:
        return self.gt_mask.shape[:2]

    @property
    def has_gt(self) -> bool:
        return bool(self.gt_mask.any())

    def param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)


# --------------------------------------------------------------------------- #
# 图层：插件唯一的输出形态，内核与前端只认它
# --------------------------------------------------------------------------- #
@dataclass
class Layer:
    """一个可渲染的结果图层。

    mask  : (H, W) 布尔掩膜，None 表示无像素输出（纯统计类插件）
    color : BGR 颜色
    alpha : 叠加透明度 [0, 1]
    meta  : 附加数据（指标值、轮廓点、bbox…）供前端表格/文字展示
    """

    name: str
    mask: np.ndarray | None = None
    color: Tuple[int, int, int] = (0, 0, 255)
    alpha: float = 0.45
    visible: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mask is not None:
            self.mask = AnalysisContext._as_mask(self.mask)
        self.alpha = float(np.clip(self.alpha, 0.0, 1.0))
        self.meta = dict(self.meta or {})

    def render(self, base: np.ndarray) -> np.ndarray:
        """把本图层叠加到 base（BGR）上，返回新图，不修改入参。"""
        base = AnalysisContext._as_bgr(base)
        if not self.visible or self.mask is None:
            return base.copy()
        out = base.copy()
        region = self.mask
        if not region.any():
            return out
        color = np.asarray(self.color, dtype=np.float32)
        a = self.alpha
        out[region] = (out[region].astype(np.float32) * (1.0 - a) + color * a).astype(np.uint8)
        return out

    @classmethod
    def from_mask(cls, name: str, mask: np.ndarray, color=(0, 0, 255), **kw) -> "Layer":
        return cls(name=name, mask=mask, color=color, **kw)


def render_layers(base: np.ndarray, layers: Iterable[Layer]) -> np.ndarray:
    """按顺序叠加多个图层（渲染工具函数，与内核无关）。"""
    out = AnalysisContext._as_bgr(base).copy()
    for layer in layers:
        out = layer.render(out)
    return out


# --------------------------------------------------------------------------- #
# 插件基类：所有扩展点的唯一父类
# --------------------------------------------------------------------------- #
class AnalysisPlugin(ABC):
    """分析插件抽象基类。

    子类只需做三件事：
        1. 声明唯一的 `taskType`（调度器据此路由）
        2. 实现 `analyze(ctx) -> Layer`
        3. （可选）覆写 `default_params` / `validate`

    插件是无状态单例，一次请求的中间数据请放进 ctx.meta，不要挂到 self 上。
    """

    #: 任务类型，调度器的路由键，全局唯一，例如 "highlight" / "iou" / "boundary"
    taskType: str = ""
    #: 展示名
    name: str = ""
    #: 一句话说明（前端插件列表用）
    description: str = ""
    #: 前端参数默认值
    default_params: Dict[str, Any] = {}

    # -- 契约 ---------------------------------------------------------------- #
    @abstractmethod
    def analyze(self, ctx: AnalysisContext) -> Layer:
        """核心方法：读取 ctx，返回一个 Layer。"""
        raise NotImplementedError

    # -- 可选钩子 ------------------------------------------------------------ #
    def validate(self, ctx: AnalysisContext) -> None:
        """前置校验，抛 PluginError 即可中止本次调度。"""

    def on_register(self) -> None:
        """被注册表收录时回调（可用于加载模型/权重）。"""

    # -- 元信息 -------------------------------------------------------------- #
    @classmethod
    def info(cls) -> Dict[str, Any]:
        return {
            "taskType": cls.taskType,
            "name": cls.name or cls.__name__,
            "description": cls.description,
            "defaultParams": dict(cls.default_params),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} taskType={self.taskType!r}>"
