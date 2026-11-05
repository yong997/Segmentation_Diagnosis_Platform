"""插件包。

约定（与 core.PluginRegistry.discover 对齐）：
    * 本文件保持为空/仅注释，不要在这里 import 插件类；
    * 每个插件一个文件，文件内定义且只定义自己的 AnalysisPlugin 子类；
    * 插件只 import interfaces，永远不要 import core。

当前插件：
    iou_metric.py -> IoUMetricPlugin   (taskType="iou")
    highlight.py  -> HighlightPlugin   (taskType="highlight")
    boundary.py   -> BoundaryPlugin    (taskType="boundary")
"""
