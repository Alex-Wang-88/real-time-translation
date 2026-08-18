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
