"""api_queue.py —— api.py 的「生产者 / 消费者」版本处理逻辑（极简）。

改造点只有一处：把原来「请求进来就地同步算完再返回」改成异步流水线。

    生产者（Producer）：POST /api/analyze 只做 收文件 -> 生成 task_id -> 入队 -> 立刻返回 202。
    队列（Buffer）    ：queue.Queue，有界队列，满了直接拒绝，天然做背压。
    消费者（Consumer）：进程内常驻的后台线程（Worker），从队列取任务跑真实分析，
                        结果写进全局字典 RESULTS，前端凭 task_id 轮询 GET /api/task/{task_id}。

好处：HTTP 线程不再被 CPU 密集的推理阻塞，并发请求只会在队列里排队，
      不会把事件循环 / 线程池占满。

启动：
    uvicorn api_queue:app --host 127.0.0.1 --port 8000

说明：这里复用 api.py 里已有的纯函数 analyze_bytes（解码 / 对齐 / 调度插件 / 打包）。
      如果你直接把本文件的内容合并进 api.py，删掉下面这行 import 即可。
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from api import analyze_bytes  # 复用的纯函数；合并进 api.py 时删掉本行
from core import CoreScheduler, PluginRegistry

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
API_TITLE = "目标分割预测诊断平台 · 后端 API（队列版）"
API_VERSION = "1.2.0"

WORKER_COUNT = 2          # 消费者线程数（按 CPU 核数 / GPU 显存调）
MAX_QUEUE_SIZE = 64       # 有界队列容量，满了返回 503，避免内存被任务撑爆
RESULT_TTL_SEC = 600      # 结果在全局字典里的保留时长，过期由 Worker 顺手清理

# 任务状态机：pending -> running -> done / failed
PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"


# --------------------------------------------------------------------------- #
# 全局共享状态（进程内）
# --------------------------------------------------------------------------- #
# 队列本身是线程安全的，多个 Worker 同时 get() 不会拿到同一个任务
TASK_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=MAX_QUEUE_SIZE)

# 结果字典本身不是线程安全的，读写都必须持锁
RESULTS: Dict[str, Dict[str, Any]] = {}
RESULTS_LOCK = threading.Lock()


def _new_task_id() -> str:
    """短小且不重复的任务号。"""
    return uuid.uuid4().hex


def _set_result(task_id: str, **fields: Any) -> None:
    """线程安全地更新某条任务记录。"""
    with RESULTS_LOCK:
        entry = RESULTS.get(task_id)
        if entry is not None:
            entry.update(fields)


def _purge_expired() -> None:
    """清理过期的已完成任务，防止全局字典无限增长（内存泄漏）。"""
    deadline = time.time() - RESULT_TTL_SEC
    with RESULTS_LOCK:
        stale = [
            tid for tid, item in RESULTS.items()
            if item["status"] in (DONE, FAILED) and item.get("finishedAt", 0.0) < deadline
        ]
        for tid in stale:
            RESULTS.pop(tid, None)


# --------------------------------------------------------------------------- #
# 消费者：后台线程 Worker
# --------------------------------------------------------------------------- #
def _worker_loop(scheduler: CoreScheduler, worker_name: str) -> None:
    """常驻后台线程：不停从队列取任务 -> 处理 -> 写结果。

    get() 是阻塞的，队列空时线程挂起，不占 CPU；
    daemon=True 保证进程退出时不会因为它在阻塞而被挂住。
    """
    while True:
        task = TASK_QUEUE.get()          # 阻塞等待，直到有任务进来
        task_id = task.get("task_id", "")
        try:
            _set_result(task_id, status=RUNNING, startedAt=time.time(), worker=worker_name)

            # ---- 真正的业务处理：CPU 密集，放在这里就不影响接口响应 ---- #
            result = analyze_bytes(
                scheduler,
                task["image"],
                task["gt"],
                task["pred"],
            )

            _set_result(
                task_id,
                status=DONE,
                result=result,
                error=None,
                finishedAt=time.time(),
            )
        except HTTPException as exc:      # analyze_bytes 内部抛的业务错误（400 / 422）
            _set_result(
                task_id,
                status=FAILED,
                error=str(exc.detail),
                finishedAt=time.time(),
            )
        except Exception as exc:          # 兜底：Worker 线程绝不能因为单个任务崩掉
            _set_result(
                task_id,
                status=FAILED,
                error=f"{type(exc).__name__}: {exc}",
                finishedAt=time.time(),
            )
        finally:
            TASK_QUEUE.task_done()        # 与 get() 配对，供 join() 统计未完成任务数
            _purge_expired()


# --------------------------------------------------------------------------- #
# 响应模型
# --------------------------------------------------------------------------- #
class SubmitResponse(BaseModel):
    """生产者返回：只给任务号，不等结果。"""

    ok: bool = True
    taskId: str
    status: str = PENDING
    queued: int = Field(0, description="当前队列中待处理的任务数")


class TaskResponse(BaseModel):
    """消费者结果查询：前端轮询用。"""

    ok: bool = True
    taskId: str
    status: str
    result: Optional[Dict[str, Any]] = Field(None, description="status=done 时才有值")
    error: Optional[str] = Field(None, description="status=failed 时才有值")
    elapsedMs: Optional[float] = None


# --------------------------------------------------------------------------- #
# 应用与生命周期
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动插件调度器 + 拉起后台 Worker 线程；关闭时随进程一起回收。"""
    registry = PluginRegistry().discover("plugins")
    scheduler = CoreScheduler(registry)
    app.state.scheduler = scheduler
    app.state.started_at = time.time()

    for i in range(WORKER_COUNT):
        threading.Thread(
            target=_worker_loop,
            args=(scheduler, f"worker-{i}"),
            name=f"analysis-worker-{i}",
            daemon=True,          # 主进程退出时自动结束，不需要手动 join
        ).start()

    yield  # 服务运行中


app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)


# --------------------------------------------------------------------------- #
# 路由
# --------------------------------------------------------------------------- #
@app.get("/api/health", summary="健康检查（含队列水位）")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": API_VERSION,
        "workers": WORKER_COUNT,
        "queueSize": TASK_QUEUE.qsize(),
        "queueMax": MAX_QUEUE_SIZE,
        "pendingResults": len(RESULTS),
        "uptimeSec": round(time.time() - getattr(app.state, "started_at", time.time()), 1),
    }


@app.post(
    "/api/analyze",
    response_model=SubmitResponse,
    status_code=202,
    summary="提交分析任务（生产者，立即返回 task_id）",
)
async def submit_analyze(
    image: UploadFile = File(..., description="原图"),
    gt: UploadFile = File(..., description="GT 掩膜"),
    pred: UploadFile = File(..., description="Pred 掩膜"),
) -> Dict[str, Any]:
    """只做 IO 读取 + 入队，绝不在这里跑推理，所以响应时间是毫秒级。"""
    image_bytes = await image.read()
    gt_bytes = await gt.read()
    pred_bytes = await pred.read()

    if not image_bytes or not gt_bytes or not pred_bytes:
        raise HTTPException(status_code=400, detail="上传的文件为空")

    task_id = _new_task_id()

    # 先登记状态，再入队：避免 Worker 抢先处理完、前端却查不到记录
    with RESULTS_LOCK:
        RESULTS[task_id] = {
            "taskId": task_id,
            "status": PENDING,
            "submittedAt": time.time(),
            "startedAt": None,
            "finishedAt": None,
            "result": None,
            "error": None,
        }

    try:
        TASK_QUEUE.put_nowait({
            "task_id": task_id,
            "image": image_bytes,
            "gt": gt_bytes,
            "pred": pred_bytes,
            "submittedAt": time.time(),
        })
    except queue.Full:
        with RESULTS_LOCK:
            RESULTS.pop(task_id, None)
        raise HTTPException(status_code=503, detail="任务队列已满，请稍后重试") from None

    return {
        "ok": True,
        "taskId": task_id,
        "status": PENDING,
        "queued": TASK_QUEUE.qsize(),
    }


@app.get(
    "/api/task/{task_id}",
    response_model=TaskResponse,
    summary="查询任务结果（消费者产物）",
)
def get_task(task_id: str) -> Dict[str, Any]:
    """前端拿到 task_id 后轮询这里，status=done 时取 result。"""
    with RESULTS_LOCK:
        entry = RESULTS.get(task_id)

    if entry is None:
        raise HTTPException(status_code=404, detail=f"任务不存在或已过期：{task_id}")

    started = entry.get("startedAt") or entry["submittedAt"]
    finished = entry.get("finishedAt") or time.time()

    return {
        "ok": True,
        "taskId": entry["taskId"],
        "status": entry["status"],
        "result": entry["result"],
        "error": entry["error"],
        "elapsedMs": round((finished - started) * 1000.0, 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_queue:app", host="127.0.0.1", port=8000, reload=False)
