"""漏检 / 误检高亮插件。

任务类型：``highlight``
输入：``ctx.image``（BGR 原图）、``ctx.gt_mask`` / ``ctx.pred_mask``（bool, HxW）
输出：两个可直接叠加的图层 + 一张已合成好的 BGR 图

    FN = GT 有、Pred 无  ->  红色半透明（漏检）
    FP = Pred 有、GT 无  ->  蓝色半透明（误检）

为什么返回两个 Layer 而不是一个：
    ``Layer`` 只承载「一张布尔掩膜 + 一种颜色」，两种颜色无法塞进同一个 Layer。
    所以本插件返回 ``[FN 图层, FP 图层]``，前端调 ``DispatchResult.render(image)``
    即可得到「原图打底 + 红蓝叠加」的成品图；同时把合成结果写进
    ``ctx.meta["highlightImage"]``，前端也可以直接取这张 BGR 图当背景显示。
"""
from __future__ import annotations

from typing import Any, List, Tuple

import cv2
import numpy as np

from interfaces import (
    AnalysisContext,
    AnalysisPlugin,
    Layer,
    PluginError,
    render_layers,
)

__all__ = ["HighlightPlugin"]


class HighlightPlugin(AnalysisPlugin):
    """把 FN 涂红、FP 涂蓝，叠在原图上。"""

    taskType = "highlight"
    name = "漏检 / 误检高亮"
    description = "FN 涂红、FP 涂蓝，半透明叠在原图上，一眼看出漏检与误检"
    default_params = {
        "fnColor": (0, 0, 255),  # BGR 红
        "fpColor": (255, 0, 0),  # BGR 蓝
        "alpha": 0.45,
        # 小于该面积的连通域直接丢弃（去椒盐噪点），0 表示不过滤
        "minArea": 0,
        # 是否额外合成一张完整 BGR 图放进 ctx.meta
        "composeImage": True,
    }

    # ------------------------------------------------------------------ #
    def validate(self, ctx: AnalysisContext) -> None:
        if ctx.gt_mask.shape != ctx.pred_mask.shape:
            raise PluginError(
                f"GT 与 Pred 尺寸不匹配: {ctx.gt_mask.shape} vs {ctx.pred_mask.shape}"
            )
        if ctx.gt_mask.size == 0:
            raise PluginError("掩膜为 0 像素，无法生成高亮")
        if ctx.image.shape[:2] != ctx.gt_mask.shape[:2]:
            raise PluginError(
                f"原图与掩膜尺寸不匹配: {ctx.image.shape[:2]} vs {ctx.gt_mask.shape[:2]}"
            )

    # ------------------------------------------------------------------ #
    def analyze(self, ctx: AnalysisContext) -> List[Layer]:
        gt = np.asarray(ctx.gt_mask, dtype=bool)
        pred = np.asarray(ctx.pred_mask, dtype=bool)
        if gt.shape != pred.shape:
            raise PluginError(f"GT 与 Pred 尺寸不匹配: {gt.shape} vs {pred.shape}")

        fn_color = self._as_color(ctx.param("fnColor", (0, 0, 255)), (0, 0, 255))
        fp_color = self._as_color(ctx.param("fpColor", (255, 0, 0)), (255, 0, 0))
        alpha = self._as_float(ctx.param("alpha", 0.45), 0.45)
        min_area = self._as_int(ctx.param("minArea", 0), 0)

        # 三种像素关系：交集 = 命中，GT-Pred = 漏检，Pred-GT = 误检
        tp = gt & pred
        fn = gt & ~pred
        fp = pred & ~gt
        fn = self._drop_small(fn, min_area)
        fp = self._drop_small(fp, min_area)

        gt_px = int(np.count_nonzero(gt))
        pred_px = int(np.count_nonzero(pred))
        fn_px = int(np.count_nonzero(fn))
        fp_px = int(np.count_nonzero(fp))
        tp_px = int(np.count_nonzero(tp))

        fn_layer = Layer(
            name="FN 漏检",
            mask=fn,
            color=fn_color,
            alpha=alpha,
            meta={
                "kind": "FN",
                "label": "漏检（GT 有 Pred 无）",
                "color": "#%02X%02X%02X" % (fn_color[2], fn_color[1], fn_color[0]),
                "pixels": fn_px,
                "ratio": self._ratio(fn_px, gt_px),
            },
        )
        fp_layer = Layer(
            name="FP 误检",
            mask=fp,
            color=fp_color,
            alpha=alpha,
            meta={
                "kind": "FP",
                "label": "误检（Pred 有 GT 无）",
                "color": "#%02X%02X%02X" % (fp_color[2], fp_color[1], fp_color[0]),
                "pixels": fp_px,
                "ratio": self._ratio(fp_px, pred_px),
            },
        )
        layers: List[Layer] = [fn_layer, fp_layer]

        # 供后续插件/前端复用的中间结果
        ctx.meta["highlight"] = {
            "tpPixels": tp_px,
            "fnPixels": fn_px,
            "fpPixels": fp_px,
            "fnRatio": self._ratio(fn_px, gt_px),
            "fpRatio": self._ratio(fp_px, pred_px),
            "alpha": alpha,
        }
        if bool(ctx.param("composeImage", True)):
            try:
                # 原图打底，按 FN -> FP 顺序叠加（两者互斥，顺序不影响结果）
                ctx.meta["highlightImage"] = render_layers(ctx.image, layers)
            except Exception:  # 合成失败不影响图层本身可用
                ctx.meta["highlightImage"] = None
        return layers

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _drop_small(mask: np.ndarray, min_area: int) -> np.ndarray:
        """丢掉面积小于 min_area 的连通域，压掉零星噪点。"""
        if min_area <= 0 or not mask.any():
            return mask
        try:
            num, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8
            )
        except cv2.error:
            return mask
        if num <= 1:
            return mask
        keep = np.zeros(num, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        return keep[labels]

    @staticmethod
    def _as_color(value: Any, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
        try:
            if isinstance(value, str):  # 支持 "#RRGGBB"（前端传十六进制）
                text = value.lstrip("#")
                if len(text) == 6:
                    r, g, b = (int(text[i : i + 2], 16) for i in (0, 2, 4))
                    return (b, g, r)
            seq = list(value)
            if len(seq) != 3:
                return default
            return tuple(int(np.clip(int(c), 0, 255)) for c in seq)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(np.clip(float(value), 0.0, 1.0))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _ratio(num: int, den: int) -> float:
        return round(num / den, 4) if den > 0 else 0.0
