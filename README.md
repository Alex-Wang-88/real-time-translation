# 会记 v2：实时段落转写

本项目是一个本机运行的实时会议记录应用。浏览器采集 16 kHz 单声道 PCM，后端使用 Qwen3-ASR 自动判断语言/方言并实时转写；结果按连续语音段落保存和显示，不包含人员身份。

## 模型链路与识别策略

- `Qwen/Qwen3-ASR-1.7B`：实时转写、分段级语言确认和冲突重识别共用的唯一 ASR 模型。
- `fsmn-vad`：保留现有 VAD 和静音阈值机制。
- `OPUS-MT en -> zh`、`OPUS-MT de -> zh`：英文和德文的本地中文翻译。

当前唯一启用的策略为 `single_1_7b_no_lid`：名称保留用于兼容旧配置，但含义是“不加载独立 LID 模型”。每个技术语音段在约 1 秒后使用同一个 1.7B 做一次语言确认；连续三次证据才确认切换，必要时再用同一模型重识别切换后的时间片。系统保留带时间戳的音频帧，并在确认边界后切片，优先保证中英德切换时不漏字、不重复、不把旧语言强行套到新段落。

流处理默认把 partial 间隔调到 1 秒、静音收尾调到 950 ms，并通过 latest-wins 合并过期 partial；这会牺牲少量即时性，换取更稳定的语言判断。`OPUS-MT en -> zh` 和 `OPUS-MT de -> zh` 仍然本地实时翻译并在启动时 warm-up。自动会后本地复译默认关闭；如需更高质量，可在会议结束后显式请求外部翻译智能体，不增加常驻本地模型。

设置页暂时不显示识别策略选择；应用固定使用该策略，并在代码配置中保留备注。`MEETING_RECOGNITION_ARCHITECTURE` 也固定使用该值。历史多模型 benchmark 脚本和报告仍保留在 `scripts/` 与 `result/`，但不会被线上应用加载。

Qwen 结果保留当前支持的语言 `zh/en/de/unknown`，中文方言则按官方 22 类归一化：安徽、东北、福建、甘肃、贵州、河北、河南、湖北、湖南、江西、宁夏、山东、陕西、山西、四川、天津、云南、浙江、粤语（香港口音）、粤语（广东口音）、吴语和闽南语；普通话使用 `mandarin`，粤语无法区分口音时使用内部的 `cantonese_unknown`。杭州话按官方类别归入 `zhejiang`，不再单独生成 `hangzhou`。普通话及中文方言直接简体化输出；英文、德文以及中英混合段落按稳定原文前缀排队翻译。语言首次出现先作为候选，连续确认后才触发段落边界；短暂 `unknown` 不切段。

## 快速运行

要求 Python 3.11、Chrome 或 Edge：

```powershell
uv venv --python 3.11 .venv
uv sync --extra audio --extra dev
Copy-Item .env.example .env
& .venv\Scripts\python.exe -m uvicorn realtime_meeting.server:app --host 127.0.0.1 --port 8765
```

首次部署可先准备模型：

```powershell
& .venv\Scripts\python.exe scripts/prepare_models.py --download-translation
& .venv\Scripts\python.exe scripts/prepare_models.py --check-only
```

生产环境建议设置 `MEETING_ASR_AUTODOWNLOAD=0` 和 `MEETING_TRANSLATION_AUTODOWNLOAD=0`，启动时只读取本地模型缓存。

最终 ASR 段默认启用同模型二次识别：空结果、短于 1.8 秒或低质量段会清空上一段上下文后再次调用同一个 `Qwen/Qwen3-ASR-1.7B`，不会加载第二个 ASR 模型。每场会议的 `pipeline_metrics` 会记录触发原因、替换次数、失败次数以及 VAD/分段诊断；可通过 `MEETING_ASR_SECONDARY_RETRY_*` 调整或关闭。

## 输出

每场会议位于 `result/meetings/<meeting-id>/`，主要文件为：

`transcript.jsonl`、`transcript_events.jsonl`、`transcript.json`、`meeting_transcript.md`、`translated_zh.md`、`original_zh.md`、`original_en.md`、`original_de.md`、`audio_manifest.json`、`manifest.json`、`session_state.json` 以及可选的 `audio/`。

transcript schema 为 `2.0`：`transcript.json` 保存 `paragraphs`，JSONL 以 `revision` 追加同一段落的更新。旧会议不迁移；部署前可清理旧的 `result/meetings` 测试目录。

## API

```text
GET    /api/v2/health
GET    /api/v2/metrics
GET    /api/v2/meetings
POST   /api/v2/meetings
POST   /api/v2/meetings/{id}/start
GET    /api/v2/meetings/{id}
PATCH  /api/v2/meetings/{id}              # 重命名会议
PATCH  /api/v2/meetings/{id}/settings
GET    /api/v2/meetings/{id}/transcript
DELETE /api/v2/meetings/{id}
POST   /api/v2/meetings/{id}/stream-ticket
WS     /api/v2/meetings/{id}/stream
POST   /api/v2/meetings/{id}/stop
POST   /api/v2/meetings/{id}/translation/retry
POST   /api/v2/meetings/{id}/summary
POST   /api/v2/meetings/{id}/todo
GET    /api/v2/meetings/{id}/files/{path}
```

WebSocket 实时事件统一为：

```json
{
  "type": "paragraph_update",
  "paragraph": {
    "segment_id": "p-000001",
    "language": "en",
    "speech_variant": null,
    "text": "We will ship the beta next week.",
    "translation_zh": "我们将在下周发布 beta 版本。",
    "translation_status": "streaming",
    "revision": 4,
    "source_revision": 3,
    "closed": false
  }
}
```

## Jimo

会议纪要和 To-do 仍使用本地配置的 Jimo SSE 节点。纪要 prompt 只使用段落中的时间、语言/方言、原文和译文，不要求人员编号。

## 测试

```powershell
uv run pytest -q
uv run python -m compileall -q realtime_meeting
```

四川方言真实评测集的下载、双文本标注和实时回放流程见
[`docs/SICHUAN_EVAL.md`](docs/SICHUAN_EVAL.md)。项目使用 WSC-Eval-ASR 的 `text_sichuan` 作为表面
识别参考；人工补充 `text_mandarin` 后，才会启用普通话语义保持评测。
