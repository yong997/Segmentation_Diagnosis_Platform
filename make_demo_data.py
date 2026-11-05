"""生成"目标分割对比诊断系统"的演示三件套。

输出文件（默认保存到本脚本所在目录）：
    1_image.png   512x512 三通道彩色原图（BGR）
    2_gt.png      512x512 单通道黑白掩膜，白色=目标，黑色=背景（0 / 255）
    3_pred.png    512x512 单通道黑白掩膜，同上

Pred 掩膜被故意做错三处，用于验证诊断系统能否正确检出：
    1) 方形 -> 椭圆      ：边界偏差（boundary error）
    2) 圆形 -> 完全缺失  ：漏检 FN（false negative）
    3) 右下角 -> 噪点块  ：误检 FP（false positive）

运行：python make_demo_data.py
依赖：opencv-python、numpy
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# 全局参数
# --------------------------------------------------------------------------- #
SIZE = 512                       # 图像边长（正方形画布）
BG_COLOR = (60, 60, 60)          # 深灰背景（BGR）
BLUE = (255, 0, 0)               # 纯蓝（BGR）
RED = (0, 0, 255)                # 纯红（BGR）

# 方形：左边，边长 160，中心 (140, 256)
SQ_X1, SQ_Y1, SQ_X2, SQ_Y2 = 60, 176, 220, 336
SQ_CENTER = ((SQ_X1 + SQ_X2) // 2, (SQ_Y1 + SQ_Y2) // 2)   # (140, 256)

# 圆形：右边，中心 (370, 256)，半径 80
CIRCLE_CENTER = (370, 256)
CIRCLE_RADIUS = 80

# 误检块：右下角一块实心白块 + 周围随机散点（模拟"噪点块"）
FP_X1, FP_Y1, FP_X2, FP_Y2 = 392, 396, 468, 466

# 输出目录 = 脚本所在目录（保证在任意 cwd 下运行都落在同一处）
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# 1. 原图
# --------------------------------------------------------------------------- #
def make_image() -> np.ndarray:
    """深灰背景 + 左侧蓝色正方形 + 右侧红色圆形。"""
    img = np.full((SIZE, SIZE, 3), BG_COLOR, dtype=np.uint8)

    # 正方形（thickness=-1 表示实心填充）
    cv2.rectangle(img, (SQ_X1, SQ_Y1), (SQ_X2, SQ_Y2), BLUE, thickness=-1)

    # 圆形
    cv2.circle(img, CIRCLE_CENTER, CIRCLE_RADIUS, RED, thickness=-1)

    return img


# --------------------------------------------------------------------------- #
# 2. GT 掩膜（标准答案）
# --------------------------------------------------------------------------- #
def make_gt() -> np.ndarray:
    """单通道黑白图：方形和圆形的位置填 255，其余为 0。"""
    gt = np.zeros((SIZE, SIZE), dtype=np.uint8)

    cv2.rectangle(gt, (SQ_X1, SQ_Y1), (SQ_X2, SQ_Y2), 255, thickness=-1)
    cv2.circle(gt, CIRCLE_CENTER, CIRCLE_RADIUS, 255, thickness=-1)

    return gt


# --------------------------------------------------------------------------- #
# 3. Pred 掩膜（故意做错）
# --------------------------------------------------------------------------- #
def make_pred() -> np.ndarray:
    """三处人为错误：椭圆化 / 漏检圆形 / 右下角误检块。"""
    pred = np.zeros((SIZE, SIZE), dtype=np.uint8)

    # (1) 方形被预测成椭圆：轴长与原方形不匹配，且略微旋转 -> 边界偏差
    #     原方形是 160x160，这里给 176x140 并旋转 12 度
    cv2.ellipse(
        pred,
        center=SQ_CENTER,
        axes=(88, 70),          # 半轴长 (a, b)
        angle=12,               # 旋转角（度）
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    )

    # (2) 圆形完全没有被预测出来 -> 什么都不画，制造 FN

    # (3) 右下角多出一个白色噪点块 -> 制造 FP
    cv2.rectangle(pred, (FP_X1, FP_Y1), (FP_X2, FP_Y2), 255, thickness=-1)

    # 再撒一些随机小白点，让"噪点"看起来更像真实模型的碎屑预测（固定种子保证可复现）
    rng = np.random.default_rng(seed=42)
    for _ in range(60):
        cx = int(rng.integers(FP_X1 - 24, FP_X2 + 24))
        cy = int(rng.integers(FP_Y1 - 24, FP_Y2 + 24))
        r = int(rng.integers(1, 4))
        cv2.circle(pred, (cx, cy), r, 255, thickness=-1)

    return pred


# --------------------------------------------------------------------------- #
# 自检：尺寸一致性 + 三项错误是否都到位
# --------------------------------------------------------------------------- #
def report(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> None:
    """打印尺寸校验和基础指标，方便肉眼确认数据没问题。"""
    assert image.shape == (SIZE, SIZE, 3), f"原图尺寸异常: {image.shape}"
    assert gt.shape == (SIZE, SIZE), f"GT 尺寸异常: {gt.shape}"
    assert pred.shape == (SIZE, SIZE), f"Pred 尺寸异常: {pred.shape}"
    assert gt.dtype == pred.dtype == np.uint8

    g = gt > 0
    p = pred > 0
    tp = int(np.logical_and(g, p).sum())
    fp = int(np.logical_and(~g, p).sum())
    fn = int(np.logical_and(g, ~p).sum())
    union = int(np.logical_or(g, p).sum())

    iou = tp / union if union else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    print("=" * 56)
    print(f"原图     : {image.shape}  dtype={image.dtype}")
    print(f"GT 掩膜  : {gt.shape}   前景像素={int(g.sum())}")
    print(f"Pred掩膜 : {pred.shape}   前景像素={int(p.sum())}")
    print("-" * 56)
    print(f"TP={tp}  FP={fp}  FN={fn}")
    print(f"IoU={iou:.4f}   Dice={dice:.4f}")
    print("-" * 56)
    # 分别看左半（方形区）和右半（圆形区）的表现，确认错误类型对得上
    left_fp = int(np.logical_and(~g, p)[:, :256].sum())
    right_fn = int(np.logical_and(g, ~p)[:, 256:].sum())
    print(f"左侧方形区误检像素(边界偏差) = {left_fp}")
    print(f"右侧圆形区漏检像素(完全漏检) = {right_fn}")
    print(f"右下角噪点块误检像素         = {int(np.logical_and(~g, p)[380:, 380:].sum())}")
    print("=" * 56)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    image = make_image()
    gt = make_gt()
    pred = make_pred()

    # 保存路径：1_image.png / 2_gt.png / 3_pred.png
    paths = {
        "1_image.png": image,
        "2_gt.png": gt,
        "3_pred.png": pred,
    }
    for name, data in paths.items():
        full = os.path.join(OUT_DIR, name)
        ok = cv2.imwrite(full, data)
        if not ok:
            raise IOError(f"写入失败: {full}")
        print(f"已保存: {full}  ({data.shape[1]}x{data.shape[0]}, {data.dtype})")

    report(image, gt, pred)


if __name__ == "__main__":
    main()
