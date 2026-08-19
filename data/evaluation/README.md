# 外部评测清单

这里保存不含第三方音频的评测 manifest。音频路径指向被 `.gitignore` 排除的
`data/external/`，请先按 [`docs/SICHUAN_EVAL.md`](../../docs/SICHUAN_EVAL.md) 下载数据。

当前的 `sichuan_wsc_easy_50.json` 是 WSC-Eval-ASR Easy 子集前 50 条的双文本清单：

- `text_sichuan` 已填入官方四川话表面转写；
- `text_mandarin` 暂为空，需要人工释义后才会启用语义保持指标；
- 不包含第三方音频，也不替代完整制造业会议 fixture。
