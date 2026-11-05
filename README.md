# 分割预测诊断平台

> **Segmentation Diagnosis Platform** —— 面向目标分割任务的预测诊断与可视化分析平台。
> 上传「原图 + GT + Pred」，一次请求拿回全部指标与差异图层；微内核 + 插件化架构，新增一种诊断能力只需新增一个文件。

[![Repo](https://img.shields.io/badge/GitHub-yong997%2FSegmentation__Diagnosis__Platform-181717?logo=github&logoColor=white)](https://github.com/yong997/Segmentation_Diagnosis_Platform)
![Stars](https://img.shields.io/github/stars/yong997/Segmentation_Diagnosis_Platform?color=yellow&logo=github&logoColor=white)
![Last Commit](https://img.shields.io/github/last-commit/yong997/Segmentation_Diagnosis_Platform?logo=git&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.46%2B-FF4B4B?logo=streamlit&logoColor=white)
![Code Style](https://img.shields.io/badge/code%20style-PEP%208-black)
![License](https://img.shields.io/badge/license-MIT-blue)

**仓库地址**：<https://github.com/yong997/Segmentation_Diagnosis_Platform>

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [架构设计](#架构设计)
- [快速开始](#快速开始)
- [目录结构](#目录结构)
- [插件开发指南](#插件开发指南)
- [API 参考](#api-参考)
- [工程规范](#工程规范)
- [路线图](#路线图)
- [许可证](#许可证)

---

## 项目简介

本项目解决一个具体而高频的问题：**模型训完了，但它到底哪里错、错多少、错在边界还是错在整体，往往只能靠肉眼盯图。**

平台把「分割质量评估」拆成可插拔的诊断能力，用一条统一流水线输出可视化结论：

| 维度 | 输出 |
| --- | --- |
| 定量指标 | IoU、mIoU、Dice、逐类指标（含背景类） |
| 定性差异 | 漏检（FN）、误检（FP）高亮图层 |
| 边界质量 | GT 轮廓与 Pred 轮廓的贴合度对比 |

设计上有三个刻意的取舍：

**1. 微内核（Microkernel）**
`core.py` 是内核，**不 import cv2、不读一个像素、不做任何算法**，只做三件事：登记插件、按 `taskType` 取件、调用并兜住异常。算法全部下沉到插件，内核永远不需要为「新增一种指标」而改动。这条约束写进了模块 docstring，是架构红线而非建议。

**2. 插件化（Plugin-based）**
`plugins/` 下每个文件就是一个独立能力，实现 `AnalysisPlugin` 契约即可被自动发现（`pkgutil` 扫描 + 子类识别），**不需要注册表硬编码、不需要改前端、不需要改路由**。插件之间零耦合，各自只依赖 `interfaces.py`，永远不 `import core`——依赖方向单向，杜绝循环依赖。

**3. FastAPI + 前后端分离**
后端是纯 API 服务（FastAPI + Uvicorn），自动生成 OpenAPI 文档；前端是独立的 Streamlit 单页应用，通过 HTTP 通信。两者可分别部署、分别扩容，前端换框架也不影响后端。

> **一次请求，全量返回。** 后端把所有可能需要切换的图层（FN / FP / GT 轮廓 / Pred 轮廓）一次算完并编码为 Base64，前端缓存在 `session_state` 里，勾选复选框、拖动透明度与线宽时**纯本地重绘，不再产生网络请求**——交互是秒级的，后端也不会被交互事件打满。

---

## 核心特性

- **统一入口**：`POST /api/analyze` 一次上传三张图，返回指标 + 原图 + 全部图层。
- **自动对齐**：掩膜与原图尺寸不一致时自动最近邻缩放，并在响应里回传 `gtResized` / `predResized` 标记，前端据此提示用户。
- **健壮解码**：兼容 8/16/32 位、灰度 / RGB / RGBA、0-1 与 0-255 掩膜，统一归一化为 `uint8` BGR 与布尔掩膜，避免下游 cv2 运算崩溃。
- **契约校验**：插件返回类型不合法、`taskType` 冲突、抽象类未实现等，一律在登记或调度阶段抛出可读的 `PluginContractError`，而不是运行时 `AttributeError`。
- **异常隔离**：单个插件抛错不会污染内核，`CoreScheduler` 统一包装为 `PluginExecutionError` 并保留原始堆栈。
- **零配置扩展**：把文件丢进 `plugins/` 就完成注册。
- **队列化可选实现**：`api_queue.py` 提供 `queue.Queue` + 后台 Worker 的异步版本（提交返回 `202 + task_id`，结果轮询获取），用于高并发场景，见 [API 参考](#api-参考)。

---

## 架构设计

```
┌────────────────────────────────────────────────────────────────────────────┐
│ ● 用户 · 浏览器  http://127.0.0.1:8501                                     │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│ ▸ 前端  app.py · Streamlit :8501                                           │
│   三路输入：原图 / GT 掩膜 / Pred 掩膜 → 点击「开始诊断」                  │
│   一次请求拿回全部图层 → session_state 缓存 → OpenCV 本地拼合              │
│   勾选与滑块只触发本地重绘，不再产生网络请求                               │
└────────────────────────────────────────────────────────────────────────────┘

        │  POST /api/analyze  ·  GET /api/health
        ▼

┌────────────────────────────────────────────────────────────────────────────┐
│ ▸ 后端  api.py · FastAPI + Uvicorn :8000                                   │
│   路由层    POST /api/analyze（multipart）· GET /api/health                │
│   预处理    字节解码 → 位深归一化 → 掩膜最近邻对齐                         │
│   打包      指标 + 原图 + 每层掩膜 → Base64 JSON                           │
└────────────────────────────────────────────────────────────────────────────┘

        │  dispatch(taskType, ctx)
        ▼

┌────────────────────────────────────────────────────────────────────────────┐
│ ▸ 微内核  core.py · 不 import cv2，不读一个像素                            │
│   PluginRegistry    扫描 plugins/ 自动登记 · taskType → 实例（简单工厂）   │
│   CoreScheduler     合并参数 → 契约校验 → 调用插件 → 兜住异常              │
└────────────────────────────────────────────────────────────────────────────┘

        │  契约层  interfaces.py · AnalysisContext / Layer / AnalysisPlugin
        ▼

┌────────────────────────────────────────────────────────────────────────────┐
│ ▸ 插件层  plugins/ · 一个能力一个文件，只依赖 interfaces.py                │
│   iou_metric.py   taskType="iou"        IoU / mIoU / Dice / 逐类指标       │
│   highlight.py    taskType="highlight"  FN 涂红 · FP 涂蓝                  │
│   boundary.py     taskType="boundary"   GT 轮廓绿 · Pred 轮廓红            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 分层职责与依赖方向

| 层 | 文件 | 职责 | 允许依赖 |
| --- | --- | --- | --- |
| 前端 | `app.py` | 上传、缓存、拼合、显隐交互 | `requests`、`streamlit`、`cv2` |
| 后端 | `api.py` | HTTP 接口、解码对齐、结果打包 | `fastapi`、`cv2`、`core`、`interfaces` |
| 内核 | `core.py` | 注册表 + 调度器，**零算法** | 标准库 + `interfaces` |
| 契约 | `interfaces.py` | 上下文 / 图层 / 插件抽象基类 | `numpy`、`cv2` |
| 插件 | `plugins/*.py` | 单一诊断能力 | **只允许** `interfaces` |

依赖是单向的：`app.py → api.py → core.py → interfaces.py ← plugins/*.py`。
内核不知道插件存在，插件不知道内核存在，两者只通过契约相识。

### 一次请求的生命周期

1. 前端把三张图打包成 `multipart/form-data`，`POST /api/analyze`。
2. 后端读取字节流 → 解码为 BGR 原图与布尔掩膜 → 尺寸不一致则最近邻对齐。
3. 后端按 `taskType` 依次调度 `iou`、`highlight`、`boundary` 三个插件，每次构造一个 `AnalysisContext` 传入。
4. 内核合并参数（插件默认值 ← 上下文参数 ← 显式参数）→ 校验 → 调用 → 归一化返回图层。
5. 后端把原图与每层掩膜编码为 Base64 PNG，连同指标一起序列化为 JSON。
6. 前端把响应存入 `session_state`，用 OpenCV 本地拼合出可视化结果；之后的每次勾选/滑动都在本地完成。

---

## 快速开始

### 环境要求

- Python **3.10+**（开发环境为 3.13）
- 无需 GPU，无需深度学习框架，纯 CPU 即可运行
- 依赖仅 5 个：FastAPI、Uvicorn、Streamlit、OpenCV（headless）、NumPy

### 1. 获取代码并安装依赖

```bash
git clone https://github.com/yong997/Segmentation_Diagnosis_Platform.git
cd Segmentation_Diagnosis_Platform

python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> 想直接下载而不装 Git，可在仓库页面点 `Code → Download ZIP` 解压后从 `cd` 开始。
> 若已有 SSH 密钥并配置到 GitHub，也可用 `git clone git@github.com:yong997/Segmentation_Diagnosis_Platform.git`。

### 2. 启动后端（务必先起）

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

看到下面这行才算就绪，再去做下一步：

```
INFO:     Uvicorn running on http://127.0.0.1:8000  (Press CTRL+C to quit)
```

接口文档：<http://127.0.0.1:8000/docs> · 健康检查：<http://127.0.0.1:8000/api/health>

### 3. 启动前端（另开一个终端）

> ⚠️ **顺序很重要：先 `uvicorn`，后 `streamlit`。** 前端启动时会探测 `/api/health`，后端没起来的话页面顶部会显示「后端离线」，此时点「开始诊断」会直接报连接失败。

```bash
# 保持后端终端运行，新开一个终端，同样激活虚拟环境
python -m streamlit run app.py --server.port 8501
```

浏览器打开 <http://127.0.0.1:8501>，上传 `test/` 目录下的三件套即可跑通完整流程：

```
test/1_image.png   原图
test/2_gt.png      GT 掩膜（标准答案）
test/3_pred.png    Pred 掩膜（模型预测）
```

### 4. 换个后端地址（可选）

前端默认连 `http://127.0.0.1:8000`，后端部署在别处时用环境变量覆盖：

```bash
# Windows PowerShell
$env:SEG_API_BASE = "http://192.168.1.10:8000"

# Linux / macOS
export SEG_API_BASE=http://192.168.1.10:8000
```

### 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 页面顶部显示「后端离线」 | 后端未启动或端口不符 | 先执行第 2 步，确认 `/docs` 能打开 |
| `Address already in use` | 8000 端口被占用 | 换端口：`--port 8001`，并同步设置 `SEG_API_BASE` |
| 上传大图失败 | Streamlit 默认上传上限 50 MB | 改 `.streamlit/config.toml` 的 `maxUploadSize` |
| 掩膜被提示「已缩放」 | 掩膜与原图分辨率不一致 | 正常行为，后端已最近邻对齐，不影响指标 |

---

## 目录结构

```
Segmentation_Diagnosis_Platform/
├── api.py                    # 后端入口：FastAPI 路由 + 解码 / 对齐 / 打包
├── api_queue.py              # 可选：队列化后端（queue.Queue + 后台 Worker）
├── app.py                    # 前端入口：Streamlit 单页应用
├── core.py                   # 微内核：PluginRegistry（注册表/工厂）+ CoreScheduler（调度）
├── interfaces.py             # 插件契约：AnalysisContext / Layer / AnalysisPlugin / 渲染工具
├── make_demo_data.py         # 生成演示用合成掩膜，便于无数据时快速试跑
├── requirements.txt          # 依赖清单（已按后端 / 前端 / 图像处理分组注释）
├── plugins/                  # 插件目录：一个能力一个文件，丢进来即注册
│   ├── __init__.py           # 仅放约定说明，不 import 任何插件类
│   ├── iou_metric.py         # taskType="iou"        指标计算
│   ├── highlight.py          # taskType="highlight"  漏检 / 误检高亮
│   └── boundary.py           # taskType="boundary"   边界轮廓对比
├── test/                     # 冒烟测试样例（原图 + GT + Pred 三件套）
│   ├── 1_image.png
│   ├── 2_gt.png
│   └── 3_pred.png
├── .streamlit/
│   └── config.toml           # 主题配色与上传大小限制
└── .vscode/                  # 编辑器工作区配置
```

**几个约定，新贡献者请先读：**

- `plugins/__init__.py` **保持为空或仅注释**，不要在里面 import 插件类——自动发现依赖「模块内定义」这一判据，提前 import 会导致类归属模块错乱。
- 每个插件文件**只定义一个** `AnalysisPlugin` 子类，且文件名与能力一一对应。
- 插件**只 import `interfaces`**，永远不要 import `core`。需要内核能力时，说明该能力应该进契约层。

---

## 插件开发指南

新增一种诊断能力 = 新增一个文件，四步：

**① 在 `plugins/` 下新建文件，例如 `plugins/centroid.py`**

```python
"""质心偏移诊断：比较 GT 与 Pred 的质心距离。"""
from __future__ import annotations

import cv2
import numpy as np

from interfaces import AnalysisContext, AnalysisPlugin, Layer


class CentroidPlugin(AnalysisPlugin):
    taskType = "centroid"
    name = "质心偏移"
    description = "计算 GT 与 Pred 掩膜质心的像素距离，并在原图上标记"
    default_params = {"color": (0, 255, 255), "alpha": 0.8, "radius": 6}

    def analyze(self, ctx: AnalysisContext) -> Layer:
        ys, xs = np.nonzero(ctx.gt_mask)
        yp, xp = np.nonzero(ctx.pred_mask)
        if xs.size == 0 or xp.size == 0:
            return Layer(name="Centroid", mask=np.zeros(ctx.shape, bool))

        gc = (int(xs.mean()), int(ys.mean()))
        pc = (int(xp.mean()), int(yp.mean()))

        canvas = np.zeros(ctx.shape, dtype=np.uint8)
        for center in (gc, pc):
            cv2.circle(canvas, center, int(ctx.param("radius", 6)), 255, -1)

        return Layer(
            name="Centroid",
            mask=canvas > 0,
            color=tuple(ctx.param("color", (0, 255, 255))),
            alpha=float(ctx.param("alpha", 0.8)),
            meta={"kind": "centroid", "gtCenter": gc, "predCenter": pc},
        )
```

**② 重启后端** —— `PluginRegistry.discover("plugins")` 在 lifespan 中扫描目录，新文件自动登记，无需任何注册代码。

**③ 验证注册成功**

```bash
curl http://127.0.0.1:8000/api/health
# {"status":"ok",...,"plugins":["boundary","centroid","highlight","iou"]}
```

**④ 按需在前端接入**（可选）—— 若希望出现新的开关，在 `app.py` 的图层显隐逻辑里加一个 `kind` 分支即可。**后端与内核无需任何改动**，这正是插件化的收益。

> 契约要求：`analyze()` 必须返回 `Layer` 或 `Layer` 列表，否则 `CoreScheduler._normalize` 会抛出 `PluginContractError`。`Layer.mask` 为布尔数组，`color` 为 BGR 三元组，`alpha ∈ [0, 1]`。

---

## API 参考

### 同步版（`api.py`，默认）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/health` | 健康检查：版本、已注册插件列表、运行时长 |
| `POST` | `/api/analyze` | 上传 `image` / `gt` / `pred` 三个文件，返回指标 + 原图 + 全部图层 |
| `GET` | `/docs` | Swagger UI（由 FastAPI 自动生成） |

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -F "image=@test/1_image.png" \
  -F "gt=@test/2_gt.png" \
  -F "pred=@test/3_pred.png"
```

响应结构（节选）：

```json
{
  "ok": true,
  "elapsedMs": 42.7,
  "metrics": { "iou": 0.812, "miou": 0.735, "dice": 0.896, "perClass": [] },
  "sourceImage": { "base64": "iVBORw0...", "width": 512, "height": 512, "format": "png" },
  "layers": [
    {
      "name": "FN 漏检",
      "kind": "FN",
      "label": "漏检（GT 有 Pred 无）",
      "pixels": 1024,
      "color": "#FF0000",
      "colorBgr": [0, 0, 255],
      "alpha": 0.45,
      "mask": { "base64": "iVBORw0...", "width": 512, "height": 512, "format": "png" }
    }
  ],
  "source": { "width": 512, "height": 512, "gtResized": false, "predResized": false },
  "warnings": []
}
```

### 异步版（`api_queue.py`，可选）

面向高并发：接口只负责入队，推理在后台线程池消费，HTTP 线程不再被 CPU 密集任务阻塞。

```bash
python -m uvicorn api_queue:app --host 127.0.0.1 --port 8000
```

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/analyze` | 入队后立即返回 `202` + `taskId`（队列满时返回 `503`） |
| `GET` | `/api/task/{task_id}` | 轮询任务状态：`pending` → `running` → `done` / `failed` |

```bash
curl -X POST http://127.0.0.1:8000/api/analyze -F "image=@a.png" -F "gt=@b.png" -F "pred=@c.png"
# {"ok":true,"taskId":"7f3c...","status":"pending","queued":0}

curl http://127.0.0.1:8000/api/task/7f3c...
# {"ok":true,"taskId":"7f3c...","status":"done","result":{...},"error":null,"elapsedMs":51.2}
```

实现要点：`queue.Queue(maxsize=64)` 有界队列形成背压；`WORKER_COUNT` 个 daemon 线程常驻消费；结果写入全局字典并加锁，`RESULT_TTL_SEC` 后由 Worker 顺手清理，避免内存无限增长。

---

## 工程规范

### 代码风格：严格遵循 PEP 8

- **命名**：模块 / 函数 / 变量 `snake_case`，类 `PascalCase`，常量 `UPPER_CASE`。
- **行宽**：统一控制在 **100 字符以内**（全仓库最长行 98 字符）。
- **类型注解**：公开函数全部标注参数与返回值类型，模块首行启用 `from __future__ import annotations` 以支持前向引用。
- **文档字符串**：每个模块、类、公开函数都有中文 docstring，说明**职责边界**而不只是复述函数名（例如「内核不 import cv2」写在 `core.py` 顶部）。
- **import 分组**：标准库 → 第三方 → 本地模块，组间空行分隔，禁止通配符导入。
- **单一职责**：`api.py` 只做 IO 与打包，`core.py` 零算法，`interfaces.py` 只放契约与纯函数。
- **分节注释**：用 `# ---- 小节标题 ---- #` 的横幅注释划分逻辑段，长函数也能一眼看清结构。

自查命令：

```bash
python -m py_compile api.py app.py core.py interfaces.py plugins/*.py

# 若本地安装了检查工具
ruff check .
black --check --line-length 100 .
```

### 版本控制：Git

项目托管于 GitHub：<https://github.com/yong997/Segmentation_Diagnosis_Platform>，全程使用 **Git** 管理，遵循主干开发 + 短生命周期分支模型：

| 分支 | 用途 |
| --- | --- |
| `main` | 始终可运行、可演示的稳定分支 |
| `feat/<name>` | 新功能（如 `feat/centroid-plugin`） |
| `fix/<name>` | 缺陷修复 |
| `docs/<name>` | 文档与注释完善 |

**提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/)**：

```
<type>(<scope>): <subject>

feat(plugins): 新增质心偏移诊断插件
fix(api): 修正掩膜尺寸不一致时的对齐插值方式
refactor(core): 将参数合并逻辑收敛到 CoreScheduler
docs(readme): 补充队列化后端的启动说明
test(iou): 补充背景类 mIoU 的边界用例
chore(deps): 升级 streamlit 至 1.46
```

约定：一个提交只做一件事；提交前必须通过 `py_compile` 自查；功能分支合并前经一次代码审查（重点看**是否破坏了依赖方向**——插件是否误 import 了 `core`）。

发布使用语义化版本标签：`git tag -a v1.1.0 -m "release: v1.1.0"`。

`.gitignore` 覆盖 `__pycache__/`、`*.pyc`、`.venv/`、`.streamlit/secrets.toml`、`~$*.docx` 等临时产物，保证仓库只含源码与样例数据。

---

## 路线图

- [x] 微内核 + 插件化骨架（注册表 / 调度器 / 契约）
- [x] FastAPI 后端与 Streamlit 前端分离
- [x] IoU / mIoU / Dice、漏检误检高亮、边界轮廓三个内置插件
- [x] 队列化异步后端（`api_queue.py`）
- [ ] 批量评测：一次上传整个目录，导出 CSV 报告
- [ ] 插件热加载：无需重启即可注册新插件
- [ ] 多类分割的混淆矩阵可视化

---

## 许可证

本项目基于 [MIT License](LICENSE) 开源，可自由用于学习、研究与二次开发。
如需引用，请注明出处：*分割预测诊断平台 / Segmentation Diagnosis Platform*。
