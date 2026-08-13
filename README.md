# 会记 v2：实时会议记录与行动项

这是一个面向 Windows 本机运行的会议记录应用。浏览器只负责采集主持人麦克风和显示结果；音频、语言识别、转写、翻译、文件保存以及 Jimo 请求均由本机后端完成。

首版支持：

- 中文、英文、德文实时原文转写；英文和德文翻译为简体中文。
- 会议结束后自动完成 ASR 精修、说话人重排和翻译；完成后由用户点击按钮生成中文 Markdown 会议纪要。
- 用户触发纪要后，纪要原子保存，再使用独立 Jimo 节点、独立 session 自动生成 To-do-list JSON。
- 会议历史、下载、删除、总结重试、To-do 独立重试和进程重启恢复。
- 本地文件存储、任务状态和企业部署所需的 Store / Queue / LLMProvider 扩展边界。

## 快速运行

要求 Python 3.11、Chrome 或 Edge。PowerShell 中执行：

```powershell
.\start.ps1
```

也可以在项目目录中双击 `启动会记.cmd`。两个启动方式都以启动文件所在目录为项目根目录，不依赖固定盘符或用户名。

首次启动脚本会创建 `.venv`、安装基础依赖和音频依赖，并从 `.env.example` 创建 `.env`。启动后打开 <http://127.0.0.1:8765>。

首次部署或需要完全离线运行时，建议先显式准备并校验所有模型。模型下载和转换可能超过启动脚本的 120 秒就绪等待时间：

```powershell
& .venv\Scripts\python.exe scripts\prepare_models.py --download-translation
& .venv\Scripts\python.exe scripts\prepare_models.py --check-only
```

也可以手动安装：

```powershell
uv venv --python 3.11 .venv
uv sync --extra audio --extra dev
Copy-Item .env.example .env
& .venv\Scripts\python.exe -m uvicorn realtime_meeting.server:app --host 127.0.0.1 --port 8765
```

本机启动脚本按 RTX 5060 Laptop 目标安装 CUDA 12.8 PyTorch。若部署到其他 GPU，请把 `pyproject.toml` 中的 CUDA wheel 改成与驱动匹配的版本；没有可用 GPU 时可去掉 `torch==...+cu128` 并用基础依赖运行 CPU 模式。实时 ASR 固定为 `large-v3-turbo`，停止后释放实时模型，再按需加载 `large-v3` 完成后处理。VAD 使用 FunASR `fsmn-vad`，依赖与当前 PyTorch/CUDA wheel 配套的 `torchaudio`；翻译模型在构建阶段下载并转换为项目内 CTranslate2 int8 格式，生产运行阶段只从本地模型目录加载。

说话人重排全程使用本地 Resemblyzer 声纹嵌入、能量 VAD 和余弦相似度聚类，不依赖外部授权、不在运行时下载说话人模型，也不回退到其他 diarization 技术栈。输出匿名 `speaker_1/...`、置信度和时间区间；模型缺失时健康检查会阻止新建会议，并明确报告缺失原因。

## Jimo 配置

复制 `.env.example` 后只在本机 `.env` 中填写：

```dotenv
JIMO_API_URL=原来的会议纪要节点URL
JIMO_AUTHORIZATION=原来的完整Authorization值
JIMO_TODO_API_URL=https://jimoai-bot-api.xiaohuodui.cn/v2/chat/completions/share?shareId=jSBaVou1SZDrd4bX
```

两个节点共用 `JIMO_AUTHORIZATION`。浏览器不会收到 URL 或 Authorization，健康检查、日志和 WebSocket URL 也不会回显密钥。

Jimo 请求继续使用旧版兼容格式：`messages`、`sessionId`、`source: "api"`、`extra: {}`。客户端不向 `extra` 注入未知的 `model` 或 `temperature` 字段。

平台节点请设置：

| 节点 | 模型 | Temperature | 用途 |
|---|---|---:|---|
| `meeting-summary` | `5.6luna` / `gpt-5.6-luna` | `0.2` | 多轮会议状态、最终会议纪要 |
| `meeting-todo` | `5.6luna` / `gpt-5.6-luna` | `0.1` | 根据已保存纪要单轮抽取行动项 |

完整提示词见 [docs/PROMPTS.md](docs/PROMPTS.md)，运行时的唯一代码来源是 [realtime_meeting/prompts.py](realtime_meeting/prompts.py)。

## 本地输出

会议结果写入 `result/meetings/<meeting-id>/`，包括：

`transcript.jsonl`、`transcript.json`、`transcript_events.jsonl`、`speaker_segments.json`、`meeting_transcript.md`、`translated_zh.md`、`original_zh.md`、`original_en.md`、`original_de.md`、`meeting_minutes.md`、`todo_list.json`、`todo_list.md`、`audio/`、`audio_manifest.json`、`manifest.json` 和 `session_state.json`。

默认保留完整会议结果 30 天，服务启动时会删除超过 `MEETING_RETENTION_DAYS` 的已结束会议目录；设为 `0` 可关闭自动过期删除。设置 `MEETING_KEEP_AUDIO=0` 时，录音会暂时保留到 ASR 精修和说话人重排完成，随后自动删除；转写、精修、说话人结果、纪要和 To-do-list 会继续保留。后处理失败时会保留录音，以便重试。

## API

主要接口：

```text
GET    /api/v2/health
GET    /api/v2/metrics
POST   /api/v2/auth/session
GET    /api/v2/meetings
POST   /api/v2/meetings
POST   /api/v2/meetings/{id}/start
GET    /api/v2/meetings/{id}
GET    /api/v2/meetings/{id}/transcript?offset=0&limit=500
DELETE /api/v2/meetings/{id}
POST   /api/v2/meetings/{id}/stream-ticket
WS     /api/v2/meetings/{id}/stream
POST   /api/v2/meetings/{id}/stop
POST   /api/v2/meetings/{id}/summary
POST   /api/v2/meetings/{id}/todo
POST   /api/v2/meetings/{id}/postprocess
GET    /api/v2/meetings/{id}/files/{path}
```

`POST /summary` 只有在 ASR 精修、说话人重排和翻译全部完成后才接受请求。生成纪要成功后，服务会自动使用当前纪要版本生成 To-do-list；`POST /todo` 用于单独重试行动项。

## 并发边界

本机默认 `MEETING_MAX_ACTIVE_MEETINGS=1`、`MEETING_INFERENCE_QUEUE_SIZE=64`、`MEETING_REFINEMENT_QUEUE_SIZE=16`，针对 RTX 5060 Laptop 8 GB / 32 GB 内存的单会议稳定运行目标。当前实现使用单个进程内 GPU 资源锁串行执行重任务，不提供可配置的 worker pool 或显存硬配额。队列有界，音频包、WebSocket ticket、文件下载和会议 ID 均有校验。WebSocket 短暂断线保留 15 秒恢复窗口，恢复窗口后仍无客户端才自动结束会议。

企业部署的第一阶段基线是 10 路同时会议，但不把它当作单张 GPU 的承诺容量。应使用目标 GPU、中文/英文/德文混合音频和 30 分钟会议压测，确认安全流数。扩展拓扑、容量指标和切换点见 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 开发与测试

```powershell
& .venv\Scripts\python.exe -m pytest -q
& .venv\Scripts\python.exe -m compileall -q realtime_meeting
```

真实 Jimo smoke test 默认读取未纳入版本控制的本地 `.env`；脚本只输出调用状态和数量，不打印 Authorization 或模型响应正文：

```powershell
& .venv\Scripts\python.exe scripts\run_jimo_smoke.py --env-file .env
```

没有配置 Jimo 时，实时转写、自动精修、翻译和文件保存仍可运行；点击生成纪要后会显示可重试的配置错误。固定合成会议数据见 [tests/fixtures/sample_meeting.jsonl](tests/fixtures/sample_meeting.jsonl)。
