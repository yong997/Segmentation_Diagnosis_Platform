"""微内核：注册表 + 调度器。

设计原则（架构红线）：
    1. 内核不 import cv2、不读一个像素、不做任何算法。
    2. 内核只做三件事：登记插件（Registry）、按 taskType 找插件（Factory）、
       调用 plugin.analyze(ctx) 并兜住异常（Scheduler）。
    3. 新增能力 = 新增一个插件文件，内核与前端都不用改。

依赖：仅标准库 + interfaces。
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

from interfaces import (
    AnalysisContext,
    AnalysisPlugin,
    Layer,
    PluginContractError,
    PluginError,
    render_layers,
)

__all__ = [
    "PluginNotFoundError",
    "PluginExecutionError",
    "DispatchResult",
    "PluginRegistry",
    "CoreScheduler",
]


# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #
class PluginNotFoundError(PluginError):
    """请求的 taskType 没有任何插件可以处理。"""


class PluginExecutionError(PluginError):
    """插件内部执行失败（已包裹原始异常）。"""


# --------------------------------------------------------------------------- #
# 调度结果
# --------------------------------------------------------------------------- #
@dataclass
class DispatchResult:
    """调度器交给前端的标准产物。"""

    taskType: str
    plugin: str
    layers: List[Layer] = field(default_factory=list)
    elapsed_ms: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def render(self, base) -> Any:
        """把结果图层叠回原图（转调 interfaces 的渲染工具）。"""
        return render_layers(base, self.layers)


# --------------------------------------------------------------------------- #
# 注册表 = 简单工厂
# --------------------------------------------------------------------------- #
class PluginRegistry:
    """taskType -> 插件实例 的登记簿，同时承担简单工厂职责。

    用法：
        registry = PluginRegistry().discover("plugins")     # 自动扫描
        registry.register(MyPlugin)                         # 手动登记
        plugin = registry.create("iou")                     # 工厂取件
    """

    def __init__(self) -> None:
        self._plugins: Dict[str, AnalysisPlugin] = {}

    # -- 登记 ---------------------------------------------------------------- #
    def register(
        self,
        plugin: Union[Type[AnalysisPlugin], AnalysisPlugin],
        *,
        override: bool = False,
    ) -> AnalysisPlugin:
        """登记插件，可传类或实例，返回实例（工厂语义，便于链式取用）。"""
        instance = plugin() if inspect.isclass(plugin) else plugin

        if not isinstance(instance, AnalysisPlugin):
            raise PluginContractError(
                f"{plugin!r} 不是 AnalysisPlugin 的子类或实例"
            )
        if inspect.isabstract(type(instance)):
            raise PluginContractError(f"{type(instance).__name__} 仍是抽象类，无法登记")

        task_type = (instance.taskType or "").strip()
        if not task_type:
            raise PluginContractError(f"{type(instance).__name__} 未声明 taskType")
        if task_type in self._plugins and not override:
            raise PluginContractError(
                f"taskType={task_type!r} 已被 {type(self._plugins[task_type]).__name__} 占用"
            )

        self._plugins[task_type] = instance
        instance.on_register()
        return instance

    def unregister(self, task_type: str) -> None:
        self._plugins.pop(task_type, None)

    # -- 查询 / 工厂 --------------------------------------------------------- #
    def create(self, task_type: str) -> AnalysisPlugin:
        """工厂方法：按 taskType 取插件实例，取不到直接抛错。"""
        try:
            return self._plugins[task_type]
        except KeyError:
            raise PluginNotFoundError(
                f"未找到 taskType={task_type!r} 的插件，已注册: {sorted(self._plugins)}"
            ) from None

    def has(self, task_type: str) -> bool:
        return task_type in self._plugins

    def task_types(self) -> List[str]:
        return sorted(self._plugins)

    def describe(self) -> List[Dict[str, Any]]:
        """给前端渲染插件列表/参数面板用。"""
        return [p.info() for p in self._plugins.values()]

    # -- 自动发现 ------------------------------------------------------------ #
    def discover(self, package: str = "plugins", *, override: bool = False) -> "PluginRegistry":
        """扫描包内所有模块，自动登记其中定义的 AnalysisPlugin 子类。

        约定：plugins/__init__.py 可以为空；插件文件放在 plugins/ 下即可。
        """
        try:
            pkg = importlib.import_module(package)
        except ImportError as exc:
            raise PluginError(f"无法导入插件包 {package!r}: {exc}") from exc

        for mod_info in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
            module = importlib.import_module(mod_info.name)
            for _, obj in vars(module).items():
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, AnalysisPlugin)
                    and obj is not AnalysisPlugin
                    and not inspect.isabstract(obj)
                    and obj.__module__ == module.__name__  # 只认本模块定义的类，跳过 import 进来的
                ):
                    if self.has(obj.taskType) and not override:
                        continue
                    self.register(obj, override=override)
        return self

    # -- 容器协议 ------------------------------------------------------------ #
    def __contains__(self, task_type: str) -> bool:
        return task_type in self._plugins

    def __len__(self) -> int:
        return len(self._plugins)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PluginRegistry {self.task_types()}>"


# --------------------------------------------------------------------------- #
# 调度器
# --------------------------------------------------------------------------- #
class CoreScheduler:
    """微内核入口：接收前端请求 → 路由 → 执行 → 返回结果。

    它只持有 ctx 的引用并原样转发，从不访问 ctx.image / ctx.gt_mask 的像素。
    """

    def __init__(self, registry: Optional[PluginRegistry] = None) -> None:
        self.registry = registry if registry is not None else PluginRegistry()

    # -- 面向前端的元信息 ---------------------------------------------------- #
    def list_plugins(self) -> List[Dict[str, Any]]:
        return self.registry.describe()

    # -- 主流程 -------------------------------------------------------------- #
    def dispatch(
        self,
        task_type: str,
        ctx: AnalysisContext,
        params: Optional[Dict[str, Any]] = None,
    ) -> DispatchResult:
        """执行一次分析任务。

        task_type : 前端选择的任务类型，等于插件的 taskType
        ctx       : 已打包好的上下文（原图 + GT + Pred）
        params    : 本次请求的参数，覆盖插件默认值
        """
        if not isinstance(ctx, AnalysisContext):
            raise PluginContractError("dispatch 需要 AnalysisContext 实例")

        plugin = self.registry.create(task_type)  # 工厂取件

        # 参数合并：插件默认值 ← 上下文参数 ← 本次显式参数
        merged: Dict[str, Any] = {**plugin.default_params, **ctx.params, **(params or {})}
        ctx.params = merged

        plugin.validate(ctx)

        started = time.perf_counter()
        try:
            output = plugin.analyze(ctx)
        except PluginError:
            raise
        except Exception as exc:  # 兜底，避免单个插件拖垮内核
            raise PluginExecutionError(
                f"插件 {type(plugin).__name__}(taskType={task_type!r}) 执行失败: {exc}"
            ) from exc
        elapsed = (time.perf_counter() - started) * 1000.0

        layers = self._normalize(output, task_type)
        return DispatchResult(
            taskType=task_type,
            plugin=type(plugin).__name__,
            layers=layers,
            elapsed_ms=round(elapsed, 2),
            meta={"pluginMeta": plugin.info()},
        )

    # -- 契约校验：analyze 必须返回 Layer（允许 Layer 列表以支持多图层） ----- #
    @staticmethod
    def _normalize(output: Any, task_type: str) -> List[Layer]:
        if isinstance(output, Layer):
            return [output]
        if isinstance(output, Sequence) and all(isinstance(x, Layer) for x in output):
            return list(output)
        raise PluginContractError(
            f"taskType={task_type!r} 的 analyze 必须返回 Layer 或 Layer 列表，"
            f"实际收到 {type(output).__name__}"
        )


# --------------------------------------------------------------------------- #
# 冒烟自测：python core.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import numpy as np

    class _DemoPlugin(AnalysisPlugin):
        taskType = "demo"
        name = "演示插件"
        description = "把 Pred 与 GT 的交集涂成红色"
        default_params = {"color": (0, 0, 255), "alpha": 0.5}

        def analyze(self, ctx: AnalysisContext) -> Layer:
            inter = ctx.pred_mask & ctx.gt_mask
            return Layer(
                name="Intersection",
                mask=inter,
                color=tuple(ctx.param("color", (0, 0, 255))),
                alpha=ctx.param("alpha", 0.5),
                meta={"pixels": int(inter.sum())},
            )

    h, w = 64, 64
    ctx = AnalysisContext(
        image=np.zeros((h, w, 3), np.uint8),
        gt_mask=np.zeros((h, w), np.uint8),
        pred_mask=np.zeros((h, w), np.uint8),
    )
    ctx.gt_mask[10:40, 10:40] = 1
    ctx.pred_mask[20:50, 20:50] = 1

    registry = PluginRegistry()
    registry.register(_DemoPlugin)  # 手动登记（也可 registry.discover("plugins")）
    scheduler = CoreScheduler(registry)
    result = scheduler.dispatch("demo", ctx, params={"alpha": 0.6})

    print("已注册任务:", scheduler.registry.task_types())
    print("图层:", result.layers[0].name, "| 交集像素:", result.layers[0].meta["pixels"])
    print("生效 alpha:", ctx.params["alpha"], "| 耗时: %.2f ms" % result.elapsed_ms)
    print("渲染输出尺寸:", result.render(ctx.image).shape)

    try:
        scheduler.dispatch("not_exist", ctx)
    except PluginNotFoundError as exc:
        print("路由兜底 OK:", exc)
