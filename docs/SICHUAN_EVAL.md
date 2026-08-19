# 四川方言真实评测集

项目使用 [WSC-Eval-ASR](https://huggingface.co/datasets/ASLP-lab/WSC-Eval/blob/main/WSC-Eval-ASR/readme.md)
作为四川方言 ASR 的外部标准评测集。它来自 WenetSpeech-Chuan，官方发布了 Easy、Hard、Short、Long
子集以及 Kaldi 风格的 `text`、`wav.scp` 和 `key.txt` 文件。

第三方音频不进入 Git 仓库。下载目录 `data/external/` 已加入 `.gitignore`；仓库只保存下载脚本、
标注约定和评测代码。使用前仍需根据数据集和原始音频的授权条款确认用途。

## 双文本标注

每条四川话样本使用以下字段：

完整的机器可读约束见 [`sichuan_annotation.schema.json`](sichuan_annotation.schema.json)。

```json
{
  "text_sichuan": "好嘛，这个事情今天先整起",
  "text_mandarin": "好的，这件事今天先开始",
  "speech_variant": "sichuan",
  "reference_text_normalization": "wsc-eval-asr-v1"
}
```

`text_sichuan` 是说话人实际说出的四川话表面文本，用于方言 ASR 的 CER/字符 n-gram 评测；
`text_mandarin` 是人工整理的普通话释义，用于可选的语义保持评测。WSC-Eval-ASR 本身只提供前者，
适配器不会自动把方言文本改写成普通话，避免把机器改写当成参考答案。

官方标注中的括号、`*`、无标点文本会原样保存在 `text_sichuan` 中；评测时沿用 WSC 的归一化规则
去除标点和不确定标记，但不覆盖原始字段。当前实时段落若要提供普通话释义，应写入
`mandarin_text` 或 `semantic_text`，评测器会单独输出普通话字符级代理指标，并标明这不是语义模型判断。

## 下载和生成清单

先安装包含 `huggingface-hub` 的可选依赖：

```powershell
uv sync --extra audio
```

先做元数据 smoke test（不下载音频）：

```powershell
uv run python scripts/download_wsc_eval_asr.py `
  --output data/external/WSC-Eval `
  --subset Easy `
  --metadata-only

uv run python scripts/prepare_sichuan_wsc_manifest.py `
  data/external/WSC-Eval/WSC-Eval-ASR `
  --subset Easy `
  --limit 50 `
  --output data/evaluation/sichuan_wsc_easy_50.json
```

需要实际音频时，去掉 `--metadata-only`。完整 Easy 子集约 8.55 小时，建议先用 `--limit 50` 验证
模型和流程，再扩大范围：

```powershell
uv run python scripts/download_wsc_eval_asr.py `
  --output data/external/WSC-Eval `
  --subset Easy `
  --limit 50

uv run python scripts/prepare_sichuan_wsc_manifest.py `
  data/external/WSC-Eval/WSC-Eval-ASR `
  --subset Easy `
  --limit 50 `
  --output data/evaluation/sichuan_wsc_easy_50.json `
  --require-audio
```

## 真实实时回放

WSC 回放会把独立片段按静音间隔串入同一个 `LiveMeetingSession`，使用项目实际的 VAD、Qwen3-ASR
1.7B、语言/方言路由和段落保存流程：

```powershell
uv run python scripts/replay_sichuan_wsc_realtime.py `
  data/evaluation/sichuan_wsc_easy_50.json `
  --output result/benchmarks/sichuan_wsc_easy_50 `
  --playback-rate 0 `
  --speech-variant-mode sichuan
```

`--playback-rate 1` 可按真实时间回放。WSC 评测是孤立语音片段的 ASR 基准，不会为每个片段请求
三段结果 API；制造业月会 fixture 仍负责完整的单次会后 API 合同测试。

最终事件若为空、短于默认 1.8 秒或质量信号较弱，会在同一个 1.7B 模型上清空上一段上下文重试一次。
回放报告的 `pipeline_metrics.asr_segment_diagnostics` 保存每个最终段的质量、触发原因和是否替换，
`pipeline_metrics.segmenter_diagnostics` 保存 VAD 是否开段、是否通过 admission 以及事件计数，便于区分
“模型返回空文本”和“VAD 没有形成事件”。

单独评测已有报告：

```powershell
uv run python scripts/evaluate_sichuan_wsc.py `
  data/evaluation/sichuan_wsc_easy_50.json `
  result/benchmarks/sichuan_wsc_easy_50/realtime_replay_report.json
```

报告重点查看：

- `sichuan_surface_error_rate`：四川话原文识别错误率；
- `sichuan_surface_chrf_mean`：四川话表面文本字符重叠；
- `sichuan_variant_accuracy`：四川方言标签准确率；
- `sichuan_mandarin_*`：只有补齐 `text_mandarin` 且输出包含 `mandarin_text`/`semantic_text` 后才会计分；
- `contract.postprocess_api_required`：WSC ASR 基准为 `false`，不替代完整会议流程测试。

## 参考来源

- [WenetSpeech-Chuan 官方仓库](https://github.com/ASLP-lab/WenetSpeech-Chuan)
- [WSC-Eval-ASR 标注和划分说明](https://huggingface.co/datasets/ASLP-lab/WSC-Eval/blob/main/WSC-Eval-ASR/readme.md)
- [WSC-Eval 数据集](https://huggingface.co/datasets/ASLP-lab/WSC-Eval)
