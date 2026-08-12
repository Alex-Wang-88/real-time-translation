# v2 部署与扩展边界

## Windows 本机首版

```text
Chrome / Edge
    │  AudioWorklet PCM16 16 kHz mono + sequence
    ▼
FastAPI REST + WebSocket
    ├── 有界音频队列
    ├── VAD / 分段 / 语言识别 / 匿名说话人编号
    ├── faster-whisper large-v3-turbo（实时）
    ├── faster-whisper large-v3（停录后按需精修）
    ├── Resemblyzer voice encoder（停录后匿名说话人重排）
    ├── OPUS-MT en→zh-CN、de→zh-CN（本地 CTranslate2 int8）
    ├── 本地文件和 session_state.json
    └── Jimo SSE：summary 多轮 → minutes 原子保存 → todo 单轮
```

默认优先级：实时 ASR > 实时翻译 > 音频落盘 > 停止后精修 > 会议纪要 > To-do-list。

默认限制：

- 单主持人麦克风、单用户、单会议。
- `MEETING_MAX_ACTIVE_MEETINGS=1`。
- `MEETING_INFERENCE_QUEUE_SIZE=64`，不使用无限队列。
- WebSocket 短暂断线保留有限恢复窗口，窗口后无人连接才停止会议，避免断线造成孤立录音。
- 实时链路只使用 `large-v3-turbo`，不隐式回退到 Fun-ASR；`large-v3` 只在停止后按需加载，失败时保留已保存的 turbo 快速稿并把后处理标记为错误。
- Jimo 请求有连接/整体超时和指数退避；失败状态可通过 REST 或页面重试。
- To-do 只读取保存成功的当前纪要版本，不把转写切片加入 To-do 上下文。
- 说话人重排固定使用 Resemblyzer voice encoder、16 kHz 单声道输入、能量 VAD 和余弦相似度聚类。Resemblyzer 权重随 Python 音频依赖部署；缺少包或权重时健康检查失败并禁止新建会议，不使用外部授权和运行时回退。
- VAD 使用 FunASR `fsmn-vad`，依赖 `torchaudio` 与当前 PyTorch/CUDA wheel 版本配套；OPUS-MT 模型在构建或预部署阶段下载、转换并校验 `model.bin`、SentencePiece 和 `meeting_model.json`，运行时不访问 Hugging Face/ModelScope。
- 服务器部署前运行 `.venv\Scripts\python.exe scripts\prepare_models.py --check-only`；该命令同时检查 VAD 本地快照、torchaudio、Resemblyzer 和两个 OPUS-MT 目录。生产环境将 `MEETING_ASR_AUTODOWNLOAD=0`、`MEETING_TRANSLATION_AUTODOWNLOAD=0`，避免服务启动时联网下载。

## 企业切换接口

代码中的运行边界按以下接口演进，不需要改变前端契约：

| 本机实现 | 企业替换 |
|---|---|
| `LocalMeetingStore` | PostgreSQL 元数据仓库 |
| `result/meetings/<id>` | S3 / MinIO 对象存储 |
| `asyncio.Queue` 和 session state | Redis、RabbitMQ 或 NATS JobQueue |
| 单进程 `LiveModelRuntime` | ASR GPU worker pool + scheduler |
| `JimoClient` | 公网 Jimo 或企业内网 OpenAI-compatible LLMProvider |
| 单用户本机认证 | OIDC / LDAP / SSO IdentityProvider |
| FastAPI 静态托管 | 统一网关、WebSocket gateway、独立前端 |

目标拓扑：

```text
统一网关 / SSO
      │
REST API + WebSocket Gateway
      │
会议元数据数据库 + 对象存储
      │
实时 ASR GPU Worker Pool
      │
LLM Summary / To-do Job Queue
      │
Jimo 公网节点或企业内网模型网关
```

## 容量和故障策略

企业第一阶段按 10 路同时实时会议建立压测基线，但实际安全流数必须由目标 GPU 和真实语言混合数据测出。至少测量 1、5、10 路和 30 分钟连续录音：首个 partial、稳定句延迟、翻译延迟、实时处理比、队列深度、丢包、显存和恢复时间。

验收条件：队列不持续增长、没有 GPU OOM、进程不崩溃、重启后每场会议可恢复。超出容量时，新会议返回 HTTP 429，已有会议继续写入转写和音频。

生产部署还应补充：

- 每个 worker 的 CPU/GPU/显存/队列/音频丢包/LLM 延迟指标。
- LLM 并发信号量、最大排队数、超时、指数退避和熔断。
- 企业密钥管理、录音加密、访问审计、对象存储生命周期和删除审计。
- 数据保留、跨区域传输、会议参与者告知和语言识别误差的合规评估。
