# v2 部署说明

## 本机拓扑

```text
Chrome / Edge
    │  WebSocket PCM16 16 kHz mono
    ▼
FastAPI
    ├── VAD / 静音阈值 / 技术性音频切片
    ├── Qwen3-ASR-1.7B：唯一常驻 ASR、语言确认与冲突重识别模型
    ├── 连续语言/方言段落聚合器
    ├── 单 worker 稳定前缀翻译队列
    ├── schema 2.0 文件存储
    └── Jimo SSE：会议纪要与 To-do
```

停录流程为：flush 最后音频段 → 等待实时 ASR → 等待翻译队列 → `recording_state=complete` → `ready_for_summary`。实时模型不因停录释放，也不存在会后模型阶段。

## 模型准备

```powershell
& .venv\Scripts\python.exe scripts/prepare_models.py --download-translation
& .venv\Scripts\python.exe scripts/prepare_models.py --check-only
```

检查报告包含 Qwen 1.7B、VAD 和 en/de 翻译模型；fallback/LID 角色在单模型模式下只是同一 checkpoint 的兼容别名。生产环境推荐：

生产环境固定使用单模型模式；录音开始后模型锁定。fallback 和 language ID 配置字段保留是为了兼容旧客户端，不会加载第二个 ASR checkpoint。

```dotenv
MEETING_SINGLE_ASR_MODEL=1
MEETING_ASR_PRIMARY=Qwen/Qwen3-ASR-1.7B
MEETING_ASR_FALLBACK=Qwen/Qwen3-ASR-1.7B
MEETING_ASR_LANGUAGE_ID=Qwen/Qwen3-ASR-1.7B
MEETING_ASR_AUTODOWNLOAD=0
MEETING_PARTIAL_INTERVAL_MS=1000
MEETING_LANGUAGE_ID_MIN_SECONDS=1.0
MEETING_LANGUAGE_ID_ON_SEGMENT=1
MEETING_LANGUAGE_CONFLICT_CONFIRMATIONS=3
MEETING_LANGUAGE_SWITCH_WINDOW_MS=800
MEETING_LANGUAGE_SWITCH_MAX_WAIT_MS=1800
MEETING_TRANSLATION_AUTODOWNLOAD=0
MEETING_TRANSLATION_WARMUP=1
MEETING_POST_TRANSLATION_ENABLED=0
```

## 存储和恢复

`result/meetings/<id>` 使用 transcript schema `2.0`。同一个 `segment_id` 通过 JSONL 事件和 `revision` 追加更新，`transcript.json` 保存当前 `paragraphs` 投影。旧 schema 不读取、不迁移；切换版本前直接清理旧测试目录。

翻译任务按段落顺序串行执行。结果提交前检查 `source_revision`，过期任务丢弃；连续失败保留原文并可调用：

当前实时队列按 segment_id 合并任务，新的 source revision 会替换尚未执行的旧任务；final 翻译优先于 provisional partial。自动会后本地复译默认关闭，如需更高质量翻译应由独立翻译智能体显式处理。

```text
POST /api/v2/meetings/{id}/translation/retry
```

## 当前实时数据流

~~~mermaid
flowchart LR
    A["Chrome / Edge<br/>PCM16 16 kHz mono"] --> B["WebSocket / FastAPI"]
    B --> C["VAD + 时间帧保留<br/>技术性语音切片"]
    C --> D["Qwen3-ASR-1.7B<br/>partial / final"]
    D --> E["同一 1.7B<br/>分段级语言确认"]
    E --> F["语言证据聚合<br/>zh / en / de / 方言"]
    F --> G["时间边界重切片<br/>段落提交"]
    G --> H["segment_id 合并翻译队列"]
    H --> I["OPUS-MT en/de -> zh<br/>本地 warm-up"]
    G --> J["paragraph_update<br/>JSONL schema 2.0"]
    I --> J
    J --> K["会议纪要 / To-do"]
~~~

系统默认约 1 秒后做一次同模语言探测，连续三次一致证据才确认切换。切换确认后使用带时间戳的音频帧重切旧语言和新语言窗口；短暂 unknown 不切段，中文方言只更新 speech_variant。

GET /api/v2/metrics 会返回首个 partial、语言切换、ASR/翻译队列等待、最终翻译耗时、GPU 等待、过期任务丢弃数量和队列最大深度等指标。

## 企业扩展边界

| 本机实现 | 企业替换 |
|---|---|
| `LocalMeetingStore` | PostgreSQL 元数据仓库 |
| `result/meetings/<id>` | S3 / MinIO |
| `asyncio.Queue` | Redis、NATS 或 Job Queue |
| 单进程 Qwen runtime | GPU worker pool |
| `JimoClient` | 企业 LLM 网关 |
| 本地认证 | OIDC / SSO |

压测应覆盖普通话、英文、德文、Qwen 官方 22 类中文方言中的代表类别（含浙江、粤语香港/广东口音、四川、吴语和闽南语）、混合语音、短暂低音量、噪声和多人同时讲话，并记录首个 partial 延迟、稳定前缀翻译延迟、队列深度、显存和重启恢复时间。
