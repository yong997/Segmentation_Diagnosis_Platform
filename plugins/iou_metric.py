"""IoU / mIoU 指标插件。

任务类型：``iou``
输入：``ctx.gt_mask`` / ``ctx.pred_mask``（bool, HxW）
输出：一个 ``mask=None`` 的统计图层，所有指标放进 ``Layer.meta``，前端直接读表格。

指标口径
--------
二值场景：
    ``iou``  = |GT ∩ Pred| / |GT ∪ Pred|  （前景交并比）
    ``miou`` = 与 ``iou`` 相同（默认不掺背景，见 ``includeBackground``）
多类场景：
    上游若把整型标签图放进 ``ctx.meta["gt_labels"]`` / ``["pred_labels"]``
    （(H, W) 标签图或 (H, W, C) 的 one-hot），本插件按类逐个算 IoU，
    ``miou`` = 出现过的类别（GT ∪ Pred 有像素）的 IoU 平均分，空类不计入。

边界约定：
    * 两边都空（union == 0）时 IoU 记为 1.0，并在 meta 里标 ``emptyUnion=True``；
    * 只在 GT 出现 → IoU = 0；只在 Pred 出现 → IoU = 0；
    * 尺寸不匹配 / 空掩膜直接抛 ``PluginError``，由内核兜住，不会打崩进程。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from interfaces import AnalysisContext, AnalysisPlugin, Layer, PluginError

__all__ = ["IoUMetricPlugin"]


class IoUMetricPlugin(AnalysisPlugin):
    """GT 与 Pred 的交并比，多类时给出平均分 mIoU。"""

    taskType = "iou"
    name = "IoU / mIoU 指标"
    description = "计算 GT 与 Pred 的交并比；多类时输出平均 IoU（mIoU）"
    default_params = {
        # 二值场景是否把背景也算作一类参与 mIoU 平均（默认不算，miou == iou）
        "includeBackground": False,
        # meta 中数值保留的小数位
        "precision": 4,
    }

    # ------------------------------------------------------------------ #
    # 前置校验：把问题挡在 analyze 之前，错误信息更干净
    # ------------------------------------------------------------------ #
    def validate(self, ctx: AnalysisContext) -> None:
        gt = np.asarray(ctx.gt_mask)
        pred = np.asarray(ctx.pred_mask)
        if gt.ndim != 2 or pred.ndim != 2:
            raise PluginError(f"IoU 只接受二维掩膜，收到 {gt.shape} / {pred.shape}")
        if gt.shape != pred.shape:
            raise PluginError(f"GT 与 Pred 尺寸不匹配: {gt.shape} vs {pred.shape}")
        if gt.size == 0:
            raise PluginError("掩膜为 0 像素，无法计算 IoU")

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def analyze(self, ctx: AnalysisContext) -> Layer:
        gt = np.asarray(ctx.gt_mask, dtype=bool)
        pred = np.asarray(ctx.pred_mask, dtype=bool)
        if gt.shape != pred.shape:  # 兜底：validate 之后掩膜仍可能被替换
            raise PluginError(f"GT 与 Pred 尺寸不匹配: {gt.shape} vs {pred.shape}")

        digits = self._as_int(ctx.param("precision", 4), 4, low=0, high=10)
        include_bg = bool(ctx.param("includeBackground", False))

        labels = self._read_label_maps(ctx, gt.shape)
        if labels is None:
            # ---- 二值口径 ---------------------------------------------- #
            per_class: List[Dict[str, Any]] = [
                self._class_stats(0, "foreground", "前景", gt, pred)
            ]
            if include_bg:
                per_class.append(
                    self._class_stats(1, "background", "背景", ~gt, ~pred)
                )
            metric = "IoU" if len(per_class) == 1 else "mIoU"
        else:
            # ---- 多类口径 ---------------------------------------------- #
            gt_lab, pred_lab, num_classes = labels
            per_class = [
                self._class_stats(c, f"class{c}", f"类别 {c}", gt_lab == c, pred_lab == c)
                for c in range(num_classes)
            ]
            metric = "mIoU"

        fg = per_class[0]
        present = [c for c in per_class if c["present"]]
        miou = float(np.mean([c["iou"] for c in present])) if present else 0.0
        score = miou if metric == "mIoU" else fg["iou"]

        meta: Dict[str, Any] = {
            "metric": metric,
            "score": self._round(score, digits),
            "iou": self._round(fg["iou"], digits),
            "miou": self._round(miou, digits),
            "dice": self._round(fg["dice"], digits),
            "precisionScore": self._round(fg["precision"], digits),
            "recallScore": self._round(fg["recall"], digits),
            # 像素级计数，前端做柱状/饼图不用再算
            "tpPixels": fg["intersection"],
            "fpPixels": fg["fp"],
            "fnPixels": fg["fn"],
            "unionPixels": fg["union"],
            "gtPixels": fg["gtPixels"],
            "predPixels": fg["predPixels"],
            "classCount": len(per_class),
            "perClass": [
                {**c, "iou": self._round(c["iou"], digits)} for c in per_class
            ],
            "emptyGt": not bool(gt.any()),
            "emptyPred": not bool(pred.any()),
            "emptyUnion": fg["union"] == 0,
            # 给前端一句话结论，省得前端再拼字符串
            "summary": f"{metric} = {self._round(score, digits):.4f}",
        }
        if meta["emptyUnion"]:
            meta["note"] = "GT 与 Pred 均为空，按约定 IoU 记为 1.0"
        elif not meta["emptyGt"] and meta["emptyPred"]:
            meta["note"] = "Pred 全空，完全漏检"
        elif meta["emptyGt"] and not meta["emptyPred"]:
            meta["note"] = "GT 全空，Pred 全为误检"

        # 纯统计插件：mask=None，内核渲染时原样返回原图
        return Layer(
            name=f"{metric} 指标",
            mask=None,
            color=(168, 79, 30),  # BGR，对应设计稿主色 #1E4FA8
            alpha=0.0,
            visible=False,
            meta=meta,
        )

    # ------------------------------------------------------------------ #
    # 单类统计
    # ------------------------------------------------------------------ #
    @staticmethod
    def _class_stats(
        class_id: int, key: str, label: str, gt: np.ndarray, pred: np.ndarray
    ) -> Dict[str, Any]:
        gt = np.asarray(gt, dtype=bool)
        pred = np.asarray(pred, dtype=bool)
        inter = int(np.count_nonzero(gt & pred))
        gt_px = int(np.count_nonzero(gt))
        pred_px = int(np.count_nonzero(pred))
        union = int(np.count_nonzero(gt | pred))
        # union == 0：两边都没有这一类，按惯例记满分，由 present 标记区分
        iou = 1.0 if union == 0 else inter / union
        denom_dice = gt_px + pred_px
        dice = 1.0 if denom_dice == 0 else 2.0 * inter / denom_dice
        precision = 1.0 if pred_px == 0 else inter / pred_px
        recall = 1.0 if gt_px == 0 else inter / gt_px
        return {
            "classId": class_id,
            "key": key,
            "label": label,
            "iou": float(iou),
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall),
            "intersection": inter,
            "union": union,
            "tp": inter,
            "fp": pred_px - inter,
            "fn": gt_px - inter,
            "gtPixels": gt_px,
            "predPixels": pred_px,
            "present": union > 0,
        }

    # ------------------------------------------------------------------ #
    # 多类标签图（可选）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_label_maps(
        ctx: AnalysisContext, shape: Tuple[int, int]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """从 ctx.meta 读取可选的多类标签图；没有就返回 None 走二值分支。"""
        gt_lab = ctx.meta.get("gt_labels")
        pred_lab = ctx.meta.get("pred_labels")
        if gt_lab is None:
            gt_lab = ctx.meta.get("gtLabels")
        if pred_lab is None:
            pred_lab = ctx.meta.get("predLabels")
        if gt_lab is None or pred_lab is None:
            return None

        gt_lab = np.asarray(gt_lab)
        pred_lab = np.asarray(pred_lab)
        if gt_lab.shape[:2] != shape or pred_lab.shape[:2] != shape:
            raise PluginError(
                f"多类标签图尺寸与原图不一致: gt={gt_lab.shape} pred={pred_lab.shape} 期望 {shape}"
            )
        # (H, W, C) one-hot -> (H, W) 标签图
        if gt_lab.ndim == 3:
            gt_lab = np.argmax(gt_lab, axis=2)
        if pred_lab.ndim == 3:
            pred_lab = np.argmax(pred_lab, axis=2)
        if gt_lab.ndim != 2 or pred_lab.ndim != 2:
            raise PluginError(
                f"多类标签图必须是二维或 one-hot，收到 {gt_lab.shape} / {pred_lab.shape}"
            )

        gt_lab = gt_lab.astype(np.int32, copy=False)
        pred_lab = pred_lab.astype(np.int32, copy=False)

        declared = ctx.meta.get("num_classes") or ctx.meta.get("numClasses")
        try:
            num_classes = int(declared) if declared is not None else 0
        except (TypeError, ValueError):
            num_classes = 0
        if num_classes <= 0:
            num_classes = int(max(gt_lab.max(initial=0), pred_lab.max(initial=0))) + 1
        if num_classes <= 0:
            return None
        return gt_lab, pred_lab, num_classes

    # ------------------------------------------------------------------ #
    # 小工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _round(value: float, digits: int) -> float:
        try:
            return round(float(value), digits)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _as_int(value: Any, default: int, low: int = 0, high: int = 10) -> int:
        try:
            return int(np.clip(int(value), low, high))
        except (TypeError, ValueError):
            return default
