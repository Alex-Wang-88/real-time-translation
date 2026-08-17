# v2 部署说明

## 本机拓扑

```text
Chrome / Edge
    │  WebSocket PCM16 16 kHz mono
    ▼
FastAPI
    ├── VAD / 静音阈值 / 技术性音频切片
    ├── Qwen3-ASR-0.6B：语言与方言判断、fallback
    ├── Qwen3-ASR-1.7B：可选的高质量实时转写
    ├── Qwen3-ASR-0.6B：可选的低延迟实时转写
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

检查报告只包含 Qwen 1.7B、Qwen 0.6B、VAD 和 en/de 翻译模型。生产环境推荐：

每场会议可以在“识别设置”选择 `primary`（1.7B）或 `small`（0.6B）；录音开始后模型锁定。两者的模型 ID 通过下面的服务配置提供。

```dotenv
MEETING_ASR_PRIMARY=Qwen/Qwen3-ASR-1.7B
MEETING_ASR_FALLBACK=Qwen/Qwen3-ASR-0.6B
MEETING_ASR_LANGUAGE_ID=Qwen/Qwen3-ASR-0.6B
MEETING_ASR_AUTODOWNLOAD=0
MEETING_TRANSLATION_AUTODOWNLOAD=0
```

## 存储和恢复

`result/meetings/<id>` 使用 transcript schema `2.0`。同一个 `segment_id` 通过 JSONL 事件和 `revision` 追加更新，`transcript.json` 保存当前 `paragraphs` 投影。旧 schema 不读取、不迁移；切换版本前直接清理旧测试目录。

翻译任务按段落顺序串行执行。结果提交前检查 `source_revision`，过期任务丢弃；连续失败保留原文并可调用：

```text
POST /api/v2/meetings/{id}/translation/retry
```

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
