"""边界轮廓对比插件。

任务类型：``boundary``
输入：``ctx.image``（BGR 原图）、``ctx.gt_mask`` / ``ctx.pred_mask``（bool, HxW）
输出：两个线状图层 + 一张已合成好的 BGR 图

    GT 轮廓   ->  绿色线
    Pred 轮廓 ->  红色线

轮廓提取用 ``cv2.findContours`` + ``cv2.drawContours``（可控线宽、能顺带统计轮廓
数量与周长）；``method="canny"`` 时改用 Canny + 膨胀，适合边界毛糙的掩膜。
线宽、颜色、透明度都可由前端参数覆盖。
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from interfaces import (
    AnalysisContext,
    AnalysisPlugin,
    Layer,
    PluginError,
    render_layers,
)

__all__ = ["BoundaryPlugin"]


class BoundaryPlugin(AnalysisPlugin):
    """GT 轮廓绿线、Pred 轮廓红线，叠在原图上。"""

    taskType = "boundary"
    name = "边界轮廓对比"
    description = "GT 轮廓画绿线、Pred 轮廓画红线，叠在原图上对比边界贴合度"
    default_params = {
        "gtColor": (0, 255, 0),  # BGR 绿
        "predColor": (0, 0, 255),  # BGR 红
        "thickness": 2,
        "alpha": 0.9,
        "method": "contour",  # contour | canny
        "composeImage": True,
    }

    # ------------------------------------------------------------------ #
    def validate(self, ctx: AnalysisContext) -> None:
        if ctx.gt_mask.shape != ctx.pred_mask.shape:
            raise PluginError(
                f"GT 与 Pred 尺寸不匹配: {ctx.gt_mask.shape} vs {ctx.pred_mask.shape}"
            )
        if ctx.gt_mask.size == 0:
            raise PluginError("掩膜为 0 像素，无法提取轮廓")
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

        gt_color = self._as_color(ctx.param("gtColor", (0, 255, 0)), (0, 255, 0))
        pred_color = self._as_color(ctx.param("predColor", (0, 0, 255)), (0, 0, 255))
        thickness = self._as_int(ctx.param("thickness", 2), 2, low=1, high=15)
        alpha = self._as_float(ctx.param("alpha", 0.9), 0.9)
        method = str(ctx.param("method", "contour") or "contour").lower()

        gt_line, gt_info = self._edge_mask(gt, thickness, method)
        pred_line, pred_info = self._edge_mask(pred, thickness, method)

        gt_layer = Layer(
            name="GT 轮廓",
            mask=gt_line,
            color=gt_color,
            alpha=alpha,
            meta={
                "kind": "GT",
                "label": "真值边界",
                "color": "#%02X%02X%02X" % (gt_color[2], gt_color[1], gt_color[0]),
                "pixels": int(np.count_nonzero(gt_line)),
                **gt_info,
            },
        )
        pred_layer = Layer(
            name="Pred 轮廓",
            mask=pred_line,
            color=pred_color,
            alpha=alpha,
            meta={
                "kind": "Pred",
                "label": "预测边界",
                "color": "#%02X%02X%02X" % (pred_color[2], pred_color[1], pred_color[0]),
                "pixels": int(np.count_nonzero(pred_line)),
                **pred_info,
            },
        )
        layers: List[Layer] = [gt_layer, pred_layer]

        ctx.meta["boundary"] = {
            "gtContours": gt_info["contourCount"],
            "predContours": pred_info["contourCount"],
            "gtLength": gt_info["contourLength"],
            "predLength": pred_info["contourLength"],
            "thickness": thickness,
            "method": method,
        }
        if bool(ctx.param("composeImage", True)):
            try:
                ctx.meta["boundaryImage"] = render_layers(ctx.image, layers)
            except Exception:  # 合成失败不影响图层本身可用
                ctx.meta["boundaryImage"] = None
        return layers

    # ------------------------------------------------------------------ #
    # 轮廓提取
    # ------------------------------------------------------------------ #
    @staticmethod
    def _edge_mask(
        mask: np.ndarray, thickness: int, method: str
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """返回 (线状布尔掩膜, 统计信息)。空掩膜返回全 False，不抛异常。"""
        empty = {
            "contourCount": 0,
            "contourLength": 0.0,
            "emptyMask": True,
        }
        if not mask.any():
            return np.zeros_like(mask, dtype=bool), empty

        # findContours 需要 uint8 单通道，且要传副本避免被就地修改
        mask_u8 = (mask.astype(np.uint8) * 255)

        try:
            if method == "canny":
                edges = cv2.Canny(mask_u8, 50, 150)
                if thickness > 1:
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (thickness, thickness)
                    )
                    edges = cv2.dilate(edges, kernel, iterations=1)
                line = edges > 0
                pixels = int(np.count_nonzero(line))
                return line, {
                    "contourCount": 0,
                    "contourLength": 0.0,
                    "edgePixels": pixels,
                    "emptyMask": False,
                }

            contours, _ = cv2.findContours(
                mask_u8.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )
        except cv2.error as exc:  # OpenCV 层异常兜住，退化成“无轮廓”
            raise PluginError(f"轮廓提取失败: {exc}") from exc

        canvas = np.zeros(mask_u8.shape, dtype=np.uint8)
        if contours:
            try:
                cv2.drawContours(canvas, contours, -1, 255, int(thickness))
            except cv2.error as exc:
                raise PluginError(f"轮廓绘制失败: {exc}") from exc

        length = 0.0
        for contour in contours:
            if len(contour) >= 2:
                length += float(cv2.arcLength(contour, True))

        return canvas > 0, {
            "contourCount": int(len(contours)),
            "contourLength": round(length, 2),
            "emptyMask": False,
        }

    # ------------------------------------------------------------------ #
    # 参数清洗：前端可能传来字符串/十六进制/越界值
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_color(value: Any, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
        try:
            if isinstance(value, str):  # 支持 "#RRGGBB"
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
    def _as_int(value: Any, default: int, low: int, high: int) -> int:
        try:
            return int(np.clip(int(value), low, high))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(np.clip(float(value), 0.0, 1.0))
        except (TypeError, ValueError):
            return default
