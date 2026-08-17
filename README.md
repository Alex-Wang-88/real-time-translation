# 会记 v2：实时段落转写

本项目是一个本机运行的实时会议记录应用。浏览器采集 16 kHz 单声道 PCM，后端使用 Qwen3-ASR 自动判断语言/方言并实时转写；结果按连续语音段落保存和显示，不包含人员身份。

## 模型链路

- `Qwen/Qwen3-ASR-1.7B`：常驻实时转写主模型。
- `Qwen/Qwen3-ASR-0.6B`：新语音段开始后的语言/方言判断，同时作为主模型加载或推理失败时的 fallback。
- `fsmn-vad`：保留现有 VAD 和静音阈值机制。
- `OPUS-MT en -> zh`、`OPUS-MT de -> zh`：英文和德文的本地中文翻译。

Qwen 结果保留当前支持的语言 `zh/en/de/unknown`，中文方言则按官方 22 类归一化：安徽、东北、福建、甘肃、贵州、河北、河南、湖北、湖南、江西、宁夏、山东、陕西、山西、四川、天津、云南、浙江、粤语（香港口音）、粤语（广东口音）、吴语和闽南语；普通话使用 `mandarin`，粤语无法区分口音时使用内部的 `cantonese_unknown`。杭州话按官方类别归入 `zhejiang`，不再单独生成 `hangzhou`。普通话及中文方言直接简体化输出；英文、德文以及中英混合段落按稳定原文前缀排队翻译。语言首次出现先作为候选，连续确认后才触发段落边界；短暂 `unknown` 不切段。

新建会议时可在识别设置中选择实时模型：`primary` 使用 1.7B，`small` 使用 0.6B。官方 Qwen3-ASR 当前提供这两个模型尺寸；0.6B 更低延迟，仍由 0.6B 负责语言/方言判断。

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
