"""面向目标分割任务的预测诊断与可视化分析平台 —— Streamlit 前端。

前后端分离 + 本地渲染：
    后端（api.py）：解码、对齐、调度插件、把**原图和每个图层的掩膜**编码成 Base64。
    前端（本文件）：只在点「开始诊断」时发一次请求，把结果解码成 ndarray 缓存进
                   st.session_state；之后勾选复选框 / 拖滑块全部在本地用 OpenCV
                   重新拼合，**不再请求后端**，所以是瞬时生效的。

启动：
    1) 后端：uvicorn api:app --host 127.0.0.1 --port 8000
    2) 前端：streamlit run app.py

    后端地址可用环境变量覆盖：SEG_API_BASE=http://127.0.0.1:8000

布局（极简汇报版）：
    顶部深蓝标题条 → 三栏（左：三路上传 + 开始诊断 / 中：固定「左原图 · 右诊断结果」
    + 差异工具栏 / 右：指标面板）→ 底部状态栏。
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
API_BASE = os.getenv("SEG_API_BASE", "http://127.0.0.1:8000").rstrip("/")
API_ANALYZE = f"{API_BASE}/api/analyze"
API_HEALTH = f"{API_BASE}/api/health"
REQUEST_TIMEOUT = 60  # 秒，大图 + 边界插件可能稍慢

PAGE_TITLE = "面向目标分割任务的预测诊断与可视化分析平台"
HEADER_TITLE = "分割预测诊断台"
IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]

# 主视图两列的固定标题（不做切换，汇报时零歧义）
TITLE_LEFT = "原图"
TITLE_RIGHT = "预测诊断结果"

# 工具栏默认值
DEFAULT_FN, DEFAULT_FP, DEFAULT_BD = True, True, False
DEFAULT_ALPHA, DEFAULT_THICK = 0.5, 2

# 诊断图底图压暗系数：让红蓝高亮在深色/同色目标上也能跳出来（1.0 = 不压暗）
DIM_FACTOR = 0.55

# 图例配色（与后端插件输出的 BGR 一一对应）
LEGEND_BOUNDARY = [("#3AA76D", "GT 轮廓"), ("#D9534F", "Pred 轮廓")]


class ApiError(Exception):
    """后端调用失败（连不上 / 非 200 / 返回体异常）。"""


# 关键：忽略系统代理。本机若设置了 HTTP_PROXY/HTTPS_PROXY，requests 会把
# 发往 127.0.0.1 的请求也交给代理，结果后端明明没起却收到 502，
# 错误信息会变成「upstream connect failed」这种看不出原因的内容。
_SESSION = requests.Session()
_SESSION.trust_env = False


# --------------------------------------------------------------------------- #
# 后端调用
# --------------------------------------------------------------------------- #
def post_analyze(img_file, gt_file, pred_file) -> Dict[str, Any]:
    """把三张图 POST 给后端，返回原始 JSON。只在上传/点击按钮时调用。"""
    files = {
        "image": (
            img_file.name,
            img_file.getvalue(),
            img_file.type or "application/octet-stream",
        ),
        "gt": (gt_file.name, gt_file.getvalue(), gt_file.type or "application/octet-stream"),
        "pred": (
            pred_file.name,
            pred_file.getvalue(),
            pred_file.type or "application/octet-stream",
        ),
    }

    try:
        resp = _SESSION.post(API_ANALYZE, files=files, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(
            f"无法连接后端 {API_BASE}。请先在项目目录执行：uvicorn api:app --port 8000"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError(f"后端请求超时（>{REQUEST_TIMEOUT}s），请检查后端日志") from exc

    if resp.status_code != 200:
        detail: Any = ""
        try:
            body = resp.json()
            detail = body.get("detail", "") if isinstance(body, dict) else str(body)
            if isinstance(detail, list):  # FastAPI 参数校验错误是列表
                detail = "; ".join(str(d.get("msg", d)) for d in detail)
        except ValueError:
            detail = resp.text[:300]
        raise ApiError(f"后端返回 {resp.status_code}：{detail or '未知错误'}")

    try:
        return resp.json()
    except ValueError as exc:
        raise ApiError("后端返回的不是合法 JSON") from exc


def get_health() -> Optional[Dict[str, Any]]:
    """探活；连不上就返回 None，界面显示「离线」。"""
    try:
        resp = _SESSION.get(API_HEALTH, timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return None


# --------------------------------------------------------------------------- #
# 本地渲染：JSON(Base64) -> ndarray -> 拼合
# --------------------------------------------------------------------------- #
def decode_b64_png(data: str, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    """Base64 字符串 -> BGR / 灰度 ndarray。"""
    if not data:
        return None
    try:
        buf = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
        img = cv2.imdecode(buf, flags)
    except (ValueError, cv2.error):
        return None
    if img is None:
        return None
    if flags == cv2.IMREAD_COLOR and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def build_cache(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """把后端 JSON 一次性解码成可直接参与运算的缓存对象。

    存进 session_state 后，切换复选框就不需要再碰网络了。
    """
    source = decode_b64_png(
        (payload.get("sourceImage") or {}).get("base64", ""), cv2.IMREAD_COLOR
    )
    if source is None:
        return None

    layers: List[Dict[str, Any]] = []
    for item in payload.get("layers", []) or []:
        mask = decode_b64_png(
            (item.get("mask") or {}).get("base64", ""), cv2.IMREAD_GRAYSCALE
        )
        if mask is None:
            continue
        if mask.shape[:2] != source.shape[:2]:  # 兜底：尺寸不符就最近邻对齐
            mask = cv2.resize(
                mask, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        color = list(item.get("colorBgr") or [255, 0, 0])[:3]
        layers.append(
            {
                "name": str(item.get("name", "图层")),
                "kind": str(item.get("kind", "-")),
                "label": str(item.get("label", "")),
                "pixels": int(item.get("pixels", 0)),
                "color": str(item.get("color", "-")),
                "colorBgr": tuple(int(c) for c in color),
                "alpha": float(item.get("alpha", 0.5)),
                "mask": mask > 127,
            }
        )

    return {
        "source": source,
        "layers": layers,
        "metrics": dict(payload.get("metrics", {}) or {}),
        "sourceInfo": dict(payload.get("source", {}) or {}),
        "elapsedMs": float(payload.get("elapsedMs", 0.0)),
        "warnings": list(payload.get("warnings", []) or []),
    }


def thicken(mask: np.ndarray, thickness: int) -> np.ndarray:
    """把 1px 轮廓膨胀到目标线宽（拖滑块只做这一步，不请求后端）。"""
    t = int(thickness)
    if t <= 1:
        return mask
    k = 2 * t - 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def compose(
    cache: Dict[str, Any],
    *,
    show_fn: bool,
    show_fp: bool,
    show_boundary: bool,
    alpha: float,
    thickness: int,
) -> np.ndarray:
    """在本地把可见图层叠到压暗的原图上。

    纯 numpy 运算，512×512 大约 2~5 ms，足够做到「勾一下就变」。
    """
    base = cache["source"]
    out = np.clip(base.astype(np.float32) * DIM_FACTOR, 0, 255).astype(np.uint8)

    for layer in cache["layers"]:
        kind = layer["kind"]
        if kind == "FN" and not show_fn:
            continue
        if kind == "FP" and not show_fp:
            continue
        if kind in ("GT", "Pred") and not show_boundary:
            continue

        mask = layer["mask"]
        if kind in ("GT", "Pred"):
            # 轮廓：线宽由滑块决定，透明度用插件默认值（0.9，线条要实）
            mask = thicken(mask, thickness)
            a = layer["alpha"]
        else:
            # 漏检 / 误检：色块透明度由滑块决定
            a = float(alpha)

        if not mask.any():
            continue
        color = np.asarray(layer["colorBgr"], dtype=np.float32)
        region = out[mask].astype(np.float32)
        out[mask] = (region * (1.0 - a) + color * a).astype(np.uint8)

    return out


def encode_png_bytes(image: np.ndarray) -> Optional[bytes]:
    """ndarray -> PNG 字节，给 st.image / 下载按钮用。"""
    ok, buf = cv2.imencode(".png", image)
    return buf.tobytes() if ok else None


# --------------------------------------------------------------------------- #
# HTML 片段
# --------------------------------------------------------------------------- #
def bars_html(per_class: List[Dict[str, Any]]) -> str:
    """逐类 IoU 条形图；title 属性即悬停提示（原生 tooltip，无需 JS）。"""
    if not per_class:
        per_class = [
            {"label": "前景", "iou": 0.0},
            {"label": "背景", "iou": 0.0},
        ]
    rows = []
    for c in per_class:
        iou = float(c.get("iou", 0.0))
        pct = max(0.0, min(1.0, iou)) * 100.0
        tip = (
            f"IoU {iou:.4f} · TP {int(c.get('intersection', 0)):,} px · "
            f"FN {int(c.get('fn', 0)):,} · FP {int(c.get('fp', 0)):,} · "
            f"GT {int(c.get('gtPixels', 0)):,} px"
        )
        rows.append(
            f'<div class="bar-row"><span class="bar-label">{c.get("label", "类别")}</span>'
            f'<div class="bar-track" title="{tip}"><div class="bar-fill" '
            f'style="width:{pct:.1f}%"></div></div>'
            f'<span class="bar-val">{iou:.3f}</span></div>'
        )
    return '<div class="bars">' + "".join(rows) + "</div>"


def legend_html(layers: List[Dict[str, Any]], show_boundary: bool) -> str:
    """勾选「边界」且缓存里确实有轮廓图层时才出图例。

    漏检红 / 误检蓝 已由复选框左侧色块表达，不重复出图例，保持极简。
    """
    if not show_boundary:
        return ""
    if not any(layer["kind"] in ("GT", "Pred") for layer in layers):
        return ""
    chips = [
        f'<span class="legend-chip"><span class="legend-dot" '
        f'style="background:{color}"></span>{text}</span>'
        for color, text in LEGEND_BOUNDARY
    ]
    return '<div class="legend-row">' + "".join(chips) + "</div>"


# --------------------------------------------------------------------------- #
# 样式：对齐原型草图
# --------------------------------------------------------------------------- #
CSS = """
<style>
  /* ---------- 页面骨架 ---------- */
  [data-testid="stHeader"], [data-testid="stToolbar"],
  [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none; }
  [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
  [data-testid="stAppViewContainer"] { background: #F7F9FC; }
  .block-container { padding: 0 26px 0 26px !important; max-width: 1440px !important; }
  [data-testid="stVerticalBlock"] { gap: .5rem; }

  /* ---------- 顶部标题条 ---------- */
  .wb-header { display: flex; align-items: center; justify-content: space-between;
    height: 46px; padding: 0 26px; margin: 0 -26px 14px -26px;
    background: #1E4FA8; color: #FFFFFF; }
  .wb-header .t { font-size: .95rem; font-weight: 600; letter-spacing: .6px; }
  .wb-header .r { font-size: .76rem; opacity: .88; letter-spacing: .4px; }

  /* ---------- 卡片 ---------- */
  [data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF; border: 1px solid #E5E7EB !important; border-radius: 10px;
    padding: 14px 16px !important; }
  [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] {
    border: none !important; padding: 0 !important; }

  /* ---------- 上传卡：虚线框 + 居中「+」 ---------- */
  [data-testid="stFileUploader"] { margin-bottom: .15rem; }
  [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p {
    font-size: .84rem; font-weight: 600; color: #1A2230; text-align: center; }
  [data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed #BFD3EE !important; border-radius: 8px;
    background: #FBFDFF; min-height: 90px; padding: 8px 6px; }
  [data-testid="stFileUploaderDropzone"]:hover { border-color: #3D7BD9 !important;
    background: #F3F8FF; }
  [data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
  [data-testid="stFileUploaderDropzone"] button {
    background: transparent !important; border: none !important; box-shadow: none !important;
    color: #1E4FA8 !important; font-weight: 600; width: 100%; justify-content: center;
    display: flex; flex-direction: column; gap: 1px; }
  [data-testid="stFileUploaderDropzone"] button:hover { color: #3D7BD9 !important; }
  [data-testid="stFileUploaderDropzone"] button div { display: none; }
  [data-testid="stFileUploaderDropzone"] button::before {
    content: "+"; font-size: 1.25rem; line-height: 1; color: #3D7BD9; font-weight: 400; }
  [data-testid="stFileUploaderDropzone"] button::after {
    content: "点击或拖拽上传"; font-size: .72rem; color: #8B97A8; font-weight: 400; }

  /* ---------- 开始诊断按钮 ---------- */
  [data-testid="stButton"] button { border-radius: 8px; font-size: .86rem;
    font-weight: 600; letter-spacing: .5px; height: 42px; margin-top: .55rem; }

  /* ---------- 工具栏：复选框颜色块 ---------- */
  .st-key-chk_fn [data-testid="stCheckbox"] label::before,
  .st-key-chk_fp [data-testid="stCheckbox"] label::before,
  .st-key-chk_bd [data-testid="stCheckbox"] label::before {
    content: ""; width: .7rem; height: .7rem; border-radius: 2px; display: inline-block;
    margin-right: .34rem; border: 1px solid rgba(0,0,0,.16); vertical-align: middle; }
  .st-key-chk_fn [data-testid="stCheckbox"] label::before { background: #D9534F; }
  .st-key-chk_fp [data-testid="stCheckbox"] label::before { background: #1E4FA8; }
  .st-key-chk_bd [data-testid="stCheckbox"] label::before {
    background: linear-gradient(90deg, #3AA76D 0 50%, #D9534F 50% 100%); }
  [data-testid="stCheckbox"] { margin-bottom: .1rem; }
  [data-testid="stCheckbox"] label p { font-size: .82rem; color: #4A5568;
    white-space: nowrap; overflow: visible; text-overflow: clip; }
  [data-testid="stCheckbox"] { overflow: visible; }
  [data-testid="stSlider"] label p { font-size: .74rem; color: #8B97A8; }
  [data-testid="stSlider"] { padding-top: .2rem; }

  /* ---------- 文本 ---------- */
  .sec-title { font-size: .92rem; font-weight: 700; color: #1A2230;
    padding-bottom: .45rem; border-bottom: 1px solid #E5E7EB; margin-bottom: .7rem; }
  .hint { font-size: .75rem; color: #8B97A8; line-height: 1.65; }
  .panel-title { text-align: center; font-size: .84rem; font-weight: 600;
    color: #1A2230; letter-spacing: .6px; padding: .1rem 0 .55rem 0; }

  /* ---------- 空状态占位 ---------- */
  .panel-empty { height: 290px; display: flex; align-items: center; justify-content: center;
    background: #F7F9FC; border: 1px solid #E5E7EB; border-radius: 8px;
    color: #8B97A8; font-size: .88rem; letter-spacing: .5px; }

  /* ---------- 指标面板 ---------- */
  .m-label { font-size: .76rem; color: #8B97A8; margin-bottom: .1rem; }
  .m-value { font-size: 2.4rem; font-weight: 700; color: #1E4FA8; line-height: 1.1; }
  .m-place { height: 40px; width: 70%; margin: .12rem 0 .1rem 0; background: #E8EFF8;
    border-radius: 6px; display: flex; align-items: center; justify-content: center;
    font-size: .72rem; color: #8B97A8; }
  .m-unit { font-size: .72rem; color: #8B97A8; margin-left: .45rem; font-weight: 400; }
  .bars { margin: .75rem 0 .5rem 0; }
  .bar-row { display: flex; align-items: center; gap: .5rem; margin: .42rem 0; }
  .bar-label { width: 2.9rem; flex: none; font-size: .76rem; color: #4A5568; }
  .bar-track { flex: 1; height: 8px; background: #E8EFF8; border-radius: 4px;
    overflow: hidden; cursor: help; }
  .bar-fill { height: 100%; background: #1E4FA8; border-radius: 4px; }
  .bar-val { width: 2.4rem; flex: none; text-align: right; font-size: .72rem;
    color: #1E4FA8; font-variant-numeric: tabular-nums; }
  .kv { display: flex; justify-content: space-between; font-size: .75rem;
    color: #4A5568; padding: .16rem 0; }
  .kv span:last-child { color: #1A2230; font-weight: 600;
    font-variant-numeric: tabular-nums; }

  /* ---------- 图例 ---------- */
  .legend-row { display: flex; flex-wrap: wrap; gap: .85rem; margin: .15rem 0 .1rem 0; }
  .legend-chip { display: inline-flex; align-items: center; gap: .32rem;
    font-size: .72rem; color: #8B97A8; }
  .legend-dot { width: .58rem; height: .58rem; border-radius: 2px; display: inline-block;
    border: 1px solid rgba(0,0,0,.1); }

  /* ---------- 底部状态栏 ---------- */
  .wb-footer { display: flex; align-items: center; justify-content: space-between;
    margin: 14px -26px 0 -26px; padding: 9px 26px 11px 26px;
    border-top: 1px solid #D6DCE5; background: #FFFFFF;
    font-size: .75rem; color: #8B97A8; }
  .wb-footer .ok { color: #3D7BD9; }
  .wb-footer .off { color: #D9534F; }
</style>
"""


# --------------------------------------------------------------------------- #
# 小工具：控件默认值只在首次出现时给，避免与 session_state 冲突
# --------------------------------------------------------------------------- #
def _checkbox(label: str, key: str, default: bool) -> bool:
    kwargs: Dict[str, Any] = {"key": key}
    if key not in st.session_state:
        kwargs["value"] = default
    return bool(st.checkbox(label, **kwargs))


def _slider(
    label: str, lo: float, hi: float, default: float, step: float, key: str
) -> float:
    kwargs: Dict[str, Any] = {"min_value": lo, "max_value": hi, "step": step, "key": key}
    if key not in st.session_state:
        kwargs["value"] = default
    return float(st.slider(label, **kwargs))


def _file_sig(img_file, gt_file, pred_file) -> Optional[str]:
    """只由三张图决定的指纹：文件换了就丢弃旧缓存，避免图和结果对不上。"""
    if img_file is None or gt_file is None or pred_file is None:
        return None
    return "|".join(f"{f.name}:{f.size}" for f in (img_file, gt_file, pred_file))


# --------------------------------------------------------------------------- #
# UI 片段
# --------------------------------------------------------------------------- #
def render_header(health: Optional[Dict[str, Any]]) -> None:
    tag = f"后端 {API_BASE} · {'在线' if health else '离线'}"
    st.markdown(
        f'<div class="wb-header"><span class="t">{HEADER_TITLE}</span>'
        f'<span class="r">单图模式　|　{tag}</span></div>',
        unsafe_allow_html=True,
    )


def render_footer(
    loaded: int,
    layer_count: int,
    health: Optional[Dict[str, Any]],
    elapsed: float,
    cached: bool,
) -> None:
    if health:
        status = f'<span class="ok">后端在线</span> · 插件 {len(health.get("plugins", []))} 个'
    else:
        status = '<span class="off">后端离线</span> · 请启动 uvicorn api:app --port 8000'
    speed = f" · 分析耗时 {elapsed:.1f} ms" if elapsed > 0 else ""
    mode = " · 本地渲染" if cached else ""
    st.markdown(
        f'<div class="wb-footer"><span>{status} · 已载入 {loaded} 个文件'
        f" · 差异图层 {layer_count} 项可见{speed}{mode}</span>"
        f"<span>指标由后端计算，图层叠加由前端实时完成</span></div>",
        unsafe_allow_html=True,
    )


def render_metric_panel(metrics: Optional[Dict[str, Any]]) -> None:
    """右侧指标面板：mIoU 大字 + 逐类 IoU 条形 + 关键计数。"""
    st.markdown('<div class="sec-title">指标面板</div>', unsafe_allow_html=True)

    miou = float(metrics.get("miou", 0.0)) if metrics else None
    st.markdown('<div class="m-label">mIoU</div>', unsafe_allow_html=True)
    if miou is None:
        st.markdown('<div class="m-place">等待分析</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="m-value">{miou:.2f}<span class="m-unit">各类平均</span></div>',
            unsafe_allow_html=True,
        )

    per_class = list(metrics.get("perClass", [])) if metrics else []
    st.markdown(bars_html(per_class), unsafe_allow_html=True)

    if metrics:
        gt_px = max(int(metrics.get("gtPixels", 0)), 1)
        pred_px = max(int(metrics.get("predPixels", 0)), 1)
        st.markdown(
            f'<div class="kv"><span>前景 IoU</span>'
            f'<span>{float(metrics.get("iou", 0)):.4f}</span></div>'
            f'<div class="kv"><span>Dice</span>'
            f'<span>{float(metrics.get("dice", 0)):.4f}</span></div>'
            f'<div class="kv"><span>漏检 / GT</span>'
            f'<span>{int(metrics.get("fnPixels", 0)) / gt_px:.1%}</span></div>'
            f'<div class="kv"><span>误检 / Pred</span>'
            f'<span>{int(metrics.get("fpPixels", 0)) / pred_px:.1%}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="hint" style="margin-top:.6rem">逐类 IoU，随诊断结果实时刷新<br>'
        "悬停条形可查看像素计数</div>",
        unsafe_allow_html=True,
    )


def render_panel(title: str, image_bytes: Optional[bytes], empty_hint: str = "等待上传") -> None:
    """主视图单侧：固定标题（在上）+ 图像（或占位框）。"""
    st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)
    if not image_bytes:
        st.markdown(f'<div class="panel-empty">{empty_hint}</div>', unsafe_allow_html=True)
        return
    try:
        st.image(image_bytes, width="stretch")
    except Exception:
        st.markdown('<div class="panel-empty">该格式无法预览</div>', unsafe_allow_html=True)


def render_details(cache: Dict[str, Any], composite: Optional[np.ndarray]) -> None:
    """明细与导出（默认折叠，不干扰主视图）。"""
    metrics = cache["metrics"]
    layers = cache["layers"]
    info = cache["sourceInfo"]

    with st.expander("明细与导出", expanded=False):
        rows = [
            {
                "类别": c.get("label", c.get("classId")),
                "IoU": round(float(c.get("iou", 0.0)), 4),
                "Dice": round(float(c.get("dice", 0.0)), 4),
                "Precision": round(float(c.get("precision", 0.0)), 4),
                "Recall": round(float(c.get("recall", 0.0)), 4),
                "GT 像素": int(c.get("gtPixels", 0)),
                "Pred 像素": int(c.get("predPixels", 0)),
            }
            for c in metrics.get("perClass", [])
        ]
        if rows:
            st.dataframe(rows)
        if layers:
            st.dataframe(
                [
                    {
                        "图层": layer["name"],
                        "类型": layer["kind"],
                        "像素数": int(layer["pixels"]),
                        "颜色": layer["color"],
                        "透明度": round(float(layer["alpha"]), 2),
                    }
                    for layer in layers
                ]
            )

        st.caption(
            f"后端耗时 {cache['elapsedMs']:.1f} ms　|　"
            f"图像 {info.get('width', '-')}×{info.get('height', '-')}　|　"
            f"GT 前景 {int(info.get('gtPixels', 0)):,} px　"
            f"Pred 前景 {int(info.get('predPixels', 0)):,} px"
        )

        c1, c2 = st.columns(2)
        png = encode_png_bytes(composite) if composite is not None else None
        if png:
            c1.download_button(
                "下载诊断图 PNG",
                data=png,
                file_name="segmentation_overlay.png",
                mime="image/png",
                key="dl_png",
            )
        c2.download_button(
            "下载指标 JSON",
            data=json.dumps(metrics, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="segmentation_metrics.json",
            mime="application/json",
            key="dl_json",
        )


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    health = get_health()
    render_header(health)

    # ---------------- 读交互状态（控件在下方创建，这里先读，保证当次生效） ---------------- #
    show_fn = bool(st.session_state.get("chk_fn", DEFAULT_FN))
    show_fp = bool(st.session_state.get("chk_fp", DEFAULT_FP))
    show_boundary = bool(st.session_state.get("chk_bd", DEFAULT_BD))
    alpha = float(st.session_state.get("sld_alpha", DEFAULT_ALPHA))
    thickness = int(st.session_state.get("sld_thick", DEFAULT_THICK))

    left, mid, right = st.columns([0.92, 3.05, 1.18], gap="medium")

    # ---------------- 左栏：三路输入 + 开始诊断 ---------------- #
    with left:
        img_file = st.file_uploader("原图 RGB", type=IMAGE_TYPES, key="up_img")
        gt_file = st.file_uploader("GT 掩膜", type=IMAGE_TYPES, key="up_gt")
        pred_file = st.file_uploader("Pred 掩膜", type=IMAGE_TYPES, key="up_pred")
        all_ready = all(f is not None for f in (img_file, gt_file, pred_file))
        clicked = st.button(
            "开始诊断",
            type="primary",
            use_container_width=True,
            disabled=not all_ready,
            key="btn_run",
        )
        st.markdown(
            '<div class="hint" style="margin-top:.5rem">三路输入齐全后点击开始诊断<br>'
            "图层显隐由前端本地切换</div>",
            unsafe_allow_html=True,
        )

    loaded = sum(1 for f in (img_file, gt_file, pred_file) if f is not None)
    file_sig = _file_sig(img_file, gt_file, pred_file)

    # 换图就丢弃旧缓存：宁可回到占位态，也不让右图和新原图对不上
    if st.session_state.get("cache_sig") != file_sig:
        st.session_state["cache"] = None
        st.session_state["cache_sig"] = file_sig
        st.session_state["err"] = None

    # ---------------- 中栏：固定「左原图 · 右诊断结果」 ---------------- #
    with mid:
        with st.container(border=True):
            # 全流程唯一的网络请求：点按钮时才打后端
            if clicked and all_ready:
                with st.spinner("正在请求后端分析…"):
                    try:
                        payload = post_analyze(img_file, gt_file, pred_file)
                        built = build_cache(payload)
                        if built is None:
                            raise ApiError("后端返回的图像无法解码")
                        st.session_state["cache"] = built
                        st.session_state["err"] = None
                    except ApiError as exc:
                        st.session_state["err"] = str(exc)
                        st.session_state["cache"] = None

            cache: Optional[Dict[str, Any]] = st.session_state.get("cache")
            err: Optional[str] = st.session_state.get("err")
            if err:
                st.error(err)

            for msg in (cache or {}).get("warnings", []):
                st.warning(msg)

            # 左侧永远是上传的原图；右侧是本地拼合结果（勾选变化即时重绘）
            left_bytes = img_file.getvalue() if img_file is not None else None
            composite: Optional[np.ndarray] = None
            if cache is not None:
                composite = compose(
                    cache,
                    show_fn=show_fn,
                    show_fp=show_fp,
                    show_boundary=show_boundary,
                    alpha=alpha,
                    thickness=thickness,
                )

            p1, p2 = st.columns(2, gap="small")
            with p1:
                render_panel(TITLE_LEFT, left_bytes, "请先上传原图")
            with p2:
                render_panel(
                    TITLE_RIGHT,
                    encode_png_bytes(composite) if composite is not None else None,
                    "点击「开始诊断」生成结果",
                )

            if cache is not None:
                legend = legend_html(cache["layers"], show_boundary)
                if legend:
                    st.markdown(legend, unsafe_allow_html=True)

            t1, t2, t3, t4, t5 = st.columns([0.8, 0.8, 0.8, 2.1, 2.1])
            with t1:
                show_fn = _checkbox("漏检", "chk_fn", DEFAULT_FN)
            with t2:
                show_fp = _checkbox("误检", "chk_fp", DEFAULT_FP)
            with t3:
                show_boundary = _checkbox("边界", "chk_bd", DEFAULT_BD)
            with t4:
                thickness = int(_slider("边界宽度", 1, 8, DEFAULT_THICK, 1, "sld_thick"))
            with t5:
                alpha = _slider("叠加透明度", 0.1, 1.0, DEFAULT_ALPHA, 0.05, "sld_alpha")

            if cache is not None:
                render_details(cache, composite)

    # ---------------- 右栏：指标面板 ---------------- #
    with right:
        with st.container(border=True):
            cache = st.session_state.get("cache")
            render_metric_panel(cache["metrics"] if cache else None)
            if cache:
                st.markdown(
                    f'<div class="hint" style="margin-top:.35rem">'
                    f"后端耗时 {cache['elapsedMs']:.1f} ms</div>",
                    unsafe_allow_html=True,
                )
            elif not health:
                st.markdown(
                    '<div class="hint" style="margin-top:.35rem;color:#D9534F">'
                    "后端未连接，无法分析</div>",
                    unsafe_allow_html=True,
                )

    # ---------------- 底部状态栏 ---------------- #
    cache = st.session_state.get("cache")
    visible = 0
    if cache is not None:
        visible = sum(
            1
            for layer in cache["layers"]
            if (layer["kind"] == "FN" and show_fn)
            or (layer["kind"] == "FP" and show_fp)
            or (layer["kind"] in ("GT", "Pred") and show_boundary)
        )
    render_footer(
        loaded,
        visible,
        health,
        float(cache["elapsedMs"]) if cache else 0.0,
        cache is not None,
    )


if __name__ == "__main__":
    main()
