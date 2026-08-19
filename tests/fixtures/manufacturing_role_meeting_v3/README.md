# 制造业月会实时回放测试夹具

这个目录是项目内自包含的测试输入，运行测试不再依赖 toolbox 中的原始文件。

## 文件

- `audio/full_normal_random_pauses_16k.wav`：完整测试音频，16 kHz、单声道、PCM16，时长 767.56 秒。
- `audio/segments/`：对应 22 个发言段的音频切片。
- `manuscript.jsonl`：22 条带语言、方言、说话人和参考文本的文字稿。
- `manifest.json`：回放脚本使用的清单，所有路径均相对于本目录。

## 测试流程

在项目根目录运行：

```powershell
uv run python scripts/replay_generated_meeting_realtime.py `
  tests/fixtures/manufacturing_role_meeting_v3/manifest.json `
  --output result/benchmarks/manufacturing_role_meeting_v3/realtime_current `
  --chunk-seconds 0.5 `
  --playback-rate 1.0
```

`--playback-rate 1.0` 会按真实时间逐块输入音频，模拟实时语音输入。回放结束后，脚本会模拟用户通过项目本地 HTTP 路由点击一次“生成三段结果”：

1. `POST /api/v2/meetings/{meeting_id}/summary`

主要结果位于：

- `result/benchmarks/manufacturing_role_meeting_v3/realtime_current/realtime_replay_report.json`
- `result/benchmarks/manufacturing_role_meeting_v3/realtime_current/meetings/manufacturing-role-meeting-v3-realtime/transcript.json`
- `result/benchmarks/manufacturing_role_meeting_v3/realtime_current/meetings/manufacturing-role-meeting-v3-realtime/meeting_minutes.md`
- `result/benchmarks/manufacturing_role_meeting_v3/realtime_current/meetings/manufacturing-role-meeting-v3-realtime/todo_list.json`

对比时，以 `manuscript.jsonl` 的 `text`、`language` 和 `speech_variant` 作为识别参考；以 `realtime_replay_report.json` 中的 `paragraphs`、`translation_statuses` 和 `postprocess_api` 作为实际结果。

回放脚本还会自动生成 `automatic_evaluation`，将参考稿和实际段落按顺序对齐，并汇总 ASR、语言、四川方言、翻译、段落覆盖及单次三段结果请求的指标。若只想评测已有报告，可运行：

```powershell
uv run python scripts/evaluate_realtime_replay.py `
  tests/fixtures/manufacturing_role_meeting_v3/manifest.json `
  result/benchmarks/manufacturing_role_meeting_v3/realtime_current/realtime_replay_report.json
```

说明：`result/` 是运行结果目录，已被 `.gitignore` 忽略；本目录中的音频、文字稿、manifest 和流程说明才是固定测试输入。

四川方言模式是可选的实时回放参数：`--speech-variant-mode sichuan`。它只对中文识别增加方言提示和热词，保留外语路由；模式结果仍需通过 `automatic_evaluation` 与默认 `auto` 结果对比后才能接受。
