"""目标分割预测诊断平台 —— 后端服务（FastAPI）。

职责边界（前后端分离）：
    后端：收文件 → 解码 → 对齐 → 调度插件 → **返回原图 + 每个图层的掩膜**
    前端：收文件 → POST 一次 → 把结果缓存进 session_state → 用 OpenCV 本地拼合，
          切换复选框 / 滑块时**不再请求后端**，秒级重绘。

启动：
    pip install -r requirements.txt
    uvicorn api:app --host 127.0.0.1 --port 8000 --reload
    或直接：python api.py

接口：
    GET  /api/health    健康检查 + 已注册插件列表
    POST /api/analyze   上传 原图 / GT / Pred，一次性返回指标 + 原图 + 全部图层掩膜
    交互式文档：http://127.0.0.1:8000/docs

一次请求把「所有可能需要切换的图层」都算完（FN / FP / GT 轮廓 / Pred 轮廓），
前端才能纯本地做显隐切换，不必为每个复选框再打一次网络请求。
"""
from __future__ import annotations

import base64
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Tuple

# 保证无论从哪个目录启动，都能 import 到同级的 core / interfaces / plugins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from core import CoreScheduler, PluginRegistry
from interfaces import AnalysisContext, Layer, PluginError

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
API_TITLE = "目标分割预测诊断平台 · 后端 API"
API_VERSION = "1.1.0"

# 轮廓掩膜统一按 1px 返回，前端按滑块值做膨胀（thickness 变化不再触发请求）
CONTOUR_BASE_THICKNESS = 1


# --------------------------------------------------------------------------- #
# 响应模型（自动生成 OpenAPI 文档，前端可直接照着写）
# --------------------------------------------------------------------------- #
class ImagePayload(BaseModel):
    """一张编码好的图片。"""

    base64: str = Field(..., description="PNG 的 Base64 字符串（不含 data: 前缀）")
    width: int
    height: int
    format: str = "png"


class LayerInfo(BaseModel):
    """一个可切换的差异图层：掩膜本身 + 叠加所需的颜色/透明度。"""

    name: str
    kind: str = "-"
    label: str = ""
    pixels: int = 0
    color: str = Field("-", description="十六进制色，如 #FF0000")
    colorBgr: List[int] = Field(default_factory=list, description="[B, G, R]，前端直接拿去叠加")
    alpha: float = 0.0
    mask: ImagePayload = Field(..., description="该图层的二值掩膜（0/255 单通道 PNG）")


class SourceInfo(BaseModel):
    """输入对齐情况，前端可据此提示「掩膜尺寸不一致已缩放」。"""

    width: int
    height: int
    gtResized: bool = False
    predResized: bool = False
    gtPixels: int = 0
    predPixels: int = 0


class AnalyzeResponse(BaseModel):
    ok: bool = True
    elapsedMs: float = Field(0.0, description="后端总耗时（毫秒）")
    metrics: Dict[str, Any] = Field(..., description="iou 插件的完整 meta：miou/iou/dice/perClass…")
    sourceImage: ImagePayload = Field(..., description="归一化后的原图（BGR→PNG），前端拼合时打底")
    layers: List[LayerInfo] = Field(default_factory=list, description="FN / FP / GT 轮廓 / Pred 轮廓")
    source: SourceInfo
    warnings: List[str] = Field(default_factory=list)



# --------------------------------------------------------------------------- #
# 图像解码 / 对齐（从原 app.py 迁入，纯函数、可单测）
# --------------------------------------------------------------------------- #
def to_uint8(arr: np.ndarray) -> np.ndarray:
    """把 16/32 位图归一化到 uint8，避免后续 cv2 运算报错。"""
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255
    arr = arr.astype(np.float32)
    peak = float(np.nanmax(arr)) if arr.size else 0.0
    if peak <= 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    if peak <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def decode_bgr(data: bytes) -> np.ndarray:
    """字节流 -> BGR 三通道 uint8 图。"""
    if not data:
        raise ValueError("文件内容为空")
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("无法解码图像，请确认文件是有效的图片格式")
    img = to_uint8(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"不支持的图像形状: {img.shape}")
    return np.ascontiguousarray(img)


def decode_mask_gray(data: bytes) -> np.ndarray:
    """字节流 -> 二维灰度掩膜（uint8），兼容单通道 / RGB / RGBA 掩膜。"""
    if not data:
        raise ValueError("文件内容为空")
    buf = np.frombuffer(data, dtype=np.uint8)
    mask = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError("无法解码掩膜，请确认文件是有效的图片格式")
    mask = to_uint8(mask)

    if mask.ndim == 2:
        return mask
    if mask.ndim != 3:
        raise ValueError(f"不支持的掩膜形状: {mask.shape}")
    if mask.shape[2] == 1:
        return mask[:, :, 0]

    if mask.shape[2] >= 4:
        alpha = mask[:, :, 3]
        rgb_peak = int(mask[:, :, :3].max()) if mask[:, :, :3].size else 0
        alpha_peak = int(alpha.max())
        alpha_varies = int(alpha.min()) < alpha_peak
        if (rgb_peak == 0 and alpha_peak > 0) or alpha_varies:
            return alpha
    return mask[:, :, :3].max(axis=2)


def binarize(gray: np.ndarray, threshold: int = 127) -> np.ndarray:
    """灰度掩膜 -> 布尔掩膜。0/1 掩膜自动走 >0 分支。"""
    gray = np.asarray(gray)
    if gray.size == 0:
        raise ValueError("掩膜为空（0 像素）")
    peak = int(gray.max())
    thr = 0 if peak <= 1 else int(np.clip(threshold, 0, 255))
    return gray > thr


def align_mask(mask: np.ndarray, shape: Tuple[int, int]) -> Tuple[np.ndarray, bool]:
    """把掩膜对齐到原图尺寸（最近邻，保持二值），返回 (mask, 是否缩放)。"""
    h, w = int(shape[0]), int(shape[1])
    if mask.shape[:2] == (h, w):
        return mask, False
    resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return resized > 0, True


def encode_png_base64(image: np.ndarray) -> str:
    """BGR / 灰度 ndarray -> PNG 的 Base64 字符串。"""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG 编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_mask_base64(mask: np.ndarray) -> str:
    """布尔掩膜 -> 0/255 单通道 PNG 的 Base64，前端按需解码成二值图。"""
    binary = np.asarray(mask, dtype=bool).astype(np.uint8) * 255
    return encode_png_base64(binary)


# --------------------------------------------------------------------------- #
# 分析主流程（不依赖 FastAPI，方便单测直接调用）
# --------------------------------------------------------------------------- #
def analyze_bytes(
    scheduler: CoreScheduler,
    image_bytes: bytes,
    gt_bytes: bytes,
    pred_bytes: bytes,
) -> Dict[str, Any]:
    """核心分析：解码 → 对齐 → 调度 iou/highlight/boundary → 打包 JSON。

    一次性把所有图层都算出来（FN / FP / GT 轮廓 / Pred 轮廓），
    前端据此做纯本地显隐切换，不需要为复选框再发请求。
    """
    started = time.perf_counter()
    warnings: List[str] = []

    # ---- 1. 解码 ---------------------------------------------------------- #
    try:
        image = decode_bgr(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"原图读取失败：{exc}") from exc
    try:
        gt_raw = binarize(decode_mask_gray(gt_bytes), 127)
        pred_raw = binarize(decode_mask_gray(pred_bytes), 127)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"掩膜读取失败：{exc}") from exc

    # ---- 2. 对齐（掩膜尺寸与原图不一致时最近邻缩放） ---------------------- #
    gt, gt_resized = align_mask(gt_raw, image.shape[:2])
    pred, pred_resized = align_mask(pred_raw, image.shape[:2])
    if gt_resized or pred_resized:
        which = " / ".join(
            n for n, flag in (("GT", gt_resized), ("Pred", pred_resized)) if flag
        )
        warnings.append(f"{which} 掩膜尺寸与原图不一致，已用最近邻缩放对齐。")

    # ---- 3. 调度插件（全量计算，不做任何 show_* 过滤） -------------------- #
    try:
        # 3.1 指标：iou 插件（含背景类，得到 mIoU）
        metric_result = scheduler.dispatch(
            "iou",
            AnalysisContext(image=image, gt_mask=gt, pred_mask=pred),
            params={"includeBackground": True},
        )
        metrics = dict(metric_result.layers[0].meta)

        # 3.2 漏检 / 误检：两个图层一起返回
        hl = scheduler.dispatch(
            "highlight",
            AnalysisContext(image=image, gt_mask=gt, pred_mask=pred),
        )
        layers: List[Layer] = list(hl.layers)

        # 3.3 边界轮廓：固定 1px，前端按滑块值膨胀（线宽变化不触发请求）
        bd = scheduler.dispatch(
            "boundary",
            AnalysisContext(image=image, gt_mask=gt, pred_mask=pred),
            params={"thickness": CONTOUR_BASE_THICKNESS},
        )
        layers.extend(bd.layers)
    except PluginError as exc:
        raise HTTPException(status_code=422, detail=f"分析失败：{exc}") from exc
    except Exception as exc:  # 兜底，避免 500 堆栈直接抛给前端
        raise HTTPException(
            status_code=500, detail=f"分析出现未知错误：{type(exc).__name__}: {exc}"
        ) from exc

    # ---- 4. 打包：原图 + 每层掩膜都转 Base64 ------------------------------ #
    h, w = image.shape[:2]
    source_b64 = encode_png_base64(image)

    return {
        "ok": True,
        "elapsedMs": round((time.perf_counter() - started) * 1000.0, 2),
        "metrics": metrics,
        "sourceImage": {
            "base64": source_b64,
            "width": int(w),
            "height": int(h),
            "format": "png",
        },
        "layers": [
            {
                "name": layer.name,
                "kind": str(layer.meta.get("kind", "-")),
                "label": str(layer.meta.get("label", "")),
                "pixels": int(layer.meta.get("pixels", 0)),
                "color": str(layer.meta.get("color", "-")),
                "colorBgr": [int(c) for c in layer.color],
                "alpha": round(float(layer.alpha), 3),
                "mask": {
                    "base64": encode_mask_base64(layer.mask),
                    "width": int(w),
                    "height": int(h),
                    "format": "png",
                },
            }
            for layer in layers
            if layer.mask is not None
        ],
        "source": {
            "width": int(w),
            "height": int(h),
            "gtResized": bool(gt_resized),
            "predResized": bool(pred_resized),
            "gtPixels": int(gt.sum()),
            "predPixels": int(pred.sum()),
        },
        "warnings": warnings,
    }



# --------------------------------------------------------------------------- #
# 应用与生命周期
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时扫描插件目录，构建调度器（进程内单例）。"""
    registry = PluginRegistry().discover("plugins")
    app.state.scheduler = CoreScheduler(registry)
    app.state.started_at = time.time()
    yield


app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)

# CORS：允许任意来源跨域（本地演示放开；生产环境请改成具体域名白名单）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # allow_origins=["*"] 时不能为 True
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# 路由
# --------------------------------------------------------------------------- #
@app.get("/api/health", summary="健康检查")
def health() -> Dict[str, Any]:
    """前端可据此显示「后端在线」，也方便排查插件是否加载成功。"""
    scheduler: CoreScheduler = app.state.scheduler
    return {
        "status": "ok",
        "version": API_VERSION,
        "plugins": scheduler.registry.task_types(),
        "uptimeSec": round(time.time() - getattr(app.state, "started_at", time.time()), 1),
    }


@app.post("/api/analyze", response_model=AnalyzeResponse, summary="分割诊断分析（全量图层）")
async def analyze(
    image: UploadFile = File(..., description="原图（BGR/RGB/灰度均可）"),
    gt: UploadFile = File(..., description="GT 掩膜（标准答案）"),
    pred: UploadFile = File(..., description="Pred 掩膜（预测结果）"),
) -> Dict[str, Any]:
    """上传三张图，一次性返回指标 + 原图 + 全部图层掩膜。

    不带任何显示开关参数：显隐、透明度、线宽全部由前端在本地完成，
    这样切换复选框不需要再请求后端。

    用 `async def` 只做 IO 读取，CPU 密集的分析丢进线程池执行，
    避免阻塞事件循环（否则并发请求会排队）。
    """
    image_bytes = await image.read()
    gt_bytes = await gt.read()
    pred_bytes = await pred.read()

    if not image_bytes or not gt_bytes or not pred_bytes:
        raise HTTPException(status_code=400, detail="上传的文件为空")

    scheduler: CoreScheduler = app.state.scheduler
    return await run_in_threadpool(
        analyze_bytes,
        scheduler,
        image_bytes,
        gt_bytes,
        pred_bytes,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
