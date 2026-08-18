# 模型与段落链路基准

基准 fixture 使用 `samples` 数组。每个样本可以包含 `audio`、
`duration_seconds`、`language`、`speech_variant`、`text`、`hypothesis`、
`translation` 和 `translation_hypothesis`。

运行无额外模型依赖的回归报告：

```powershell
uv run python scripts/benchmark.py tests/fixtures/benchmark.json --output result/benchmark.json
```

报告包含 WER/CER 风格错误率、翻译字符重叠率、P50/P95 延迟、RTF、段落数量以及语言/方言覆盖。真实模型压测应另外记录首个 partial 延迟、稳定前缀翻译延迟、峰值显存、队列深度和停止收尾耗时。

线上默认链路只加载一个 Qwen3-ASR-1.7B；语言确认不能通过加载 0.6B 取得。OPUS-MT en/de -> zh 仍作为独立本地翻译阶段，自动会后本地复译默认关闭。

段落基准不评估人员身份或说话人分离。混合语言验收应覆盖普通话、英文、德文，以及 Qwen 官方 22 类中文方言（包括浙江、粤语香港/广东口音、四川、吴语和闽南语）和语言/方言切换；短暂 `unknown` 不应制造额外段落。

本仓库 fixture 只验证指标管线。真实本地模型 smoke test 需要显式提供音频文件，脚本不依赖历史会议 ID：

```powershell
uv run python scripts/smoke_test_local_models.py .\samples\speech.wav --device auto
```

## 生成会议实时回放测试

项目内自包含的制造业月会音频、文字稿、manifest 和测试流程位于：

tests/fixtures/manufacturing_role_meeting_v3/README.md

该流程使用 scripts/replay_generated_meeting_realtime.py 按 1.0 倍速逐块回放完整音频，并在回放完成后请求会议纪要和 To-do-list 两个本地 API。

回放脚本会在同一份 `realtime_replay_report.json` 中写入 `automatic_evaluation`。评测会按顺序将 `manuscript.jsonl`/`manifest.json` 的参考段与实际段落对齐，允许一条参考发言对应多个实时识别段，并输出：

- ASR token 加权错误率、宏平均错误率和字符 n-gram 重叠率；
- 语言识别准确率与语言覆盖率；
- 四川方言检测的发出率与准确率；
- 英语/德语中文翻译的完成率和重叠率；
- 段落覆盖、拆分/多余段落和两次会后 API 请求的完整性。

也可以对已经生成的回放报告单独评测，不需要重新加载模型：

```powershell
uv run python scripts/evaluate_realtime_replay.py `
  tests/fixtures/manufacturing_role_meeting_v3/manifest.json `
  result/benchmarks/manufacturing_role_meeting_v3/realtime_current/realtime_replay_report.json
```

默认会在回放报告旁生成 `automatic_evaluation.json`。这样可以在不同优化版本之间直接比较相同参考稿的 `summary`，并同时检查 `contract.passed`。

若要验证四川方言模式，在同一条完整实时流程上增加：

```powershell
uv run python scripts/replay_generated_meeting_realtime.py `
  tests/fixtures/manufacturing_role_meeting_v3/manifest.json `
  --output result/benchmarks/manufacturing_role_meeting_v3/realtime_sichuan_mode `
  --chunk-seconds 0.5 `
  --playback-rate 1.0 `
  --speech-variant-mode sichuan
```

该模式是中文方言软提示，不会把整场会议强制锁定为中文，也不会改变英语/德语路由；方言标签只有在当前段文本出现高置信四川词汇时才生成。
