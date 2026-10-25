# 会记 v2 · ASR 提速评估（2026-10-25）

结论先行：**当前流水线已经把系统层优化做完了**，剩余提速空间在模型层
（换量化/更新的 ASR checkpoint），建议作为下一次版本迭代的选项，而不是
现在动已交付系统的代码。本次评估全程未改动生产代码；`pytest` 74/74 通过。

## 现状（代码证据）

| 环节 | 实现 | 已有优化 |
|---|---|---|
| 翻译 OPUS-MT | CTranslate2 | **int8 量化**（`_ensure_translator`，TransformersConverter） |
| VAD | funasr fsmn-vad | 轻量模型，天然快 |
| ASR | Qwen3-ASR-1.7B（qwen_asr, bf16, cuda:0） | 单流推理 + GPU 优先级仲裁 |
| 调度 | asyncio 优先级队列（`LatestEventQueue`/`LatestTranslationQueue`） | provisional 逐出、latest 语义防积压 |

ASR 的单流设计是**刻意的延迟取舍**：`final > 语言切换确认 > final 翻译 >
partial` 的优先级（`_recognize`，priority 5/40）保证说话人听感优先。把
`max_inference_batch_size` 调大可以提升吞吐，但 partial 会被凑批的等待
时间拖慢——对会议实时性是负优化，故默认 1 是对的。

## 剩余杠杆（按性价比排序，供下个版本决策）

1. **官方量化 Qwen3-ASR checkpoint**：Qwen 团队放出 int8/AWQ 版 ASR 时直接
   换 snapshot（`_model_snapshot` 已按 revision 锁定，升级路径现成）。
   bf16→int8 在 8GB 卡上预期 1.5–2× 提速、显存减半。注意用 sichuan_eval.py
   的方言评测集做回归（热词/方言提示对量化敏感）。
2. **0.6B 小模型做 partial、1.7B 做 final**：代码已内置双模型路由
   （`asr_fallback`/`single_model=False`），只需配置切换——partial 延迟
   减半，final 精度不降。风险：两模型常驻显存（8GB 卡需实测）。
3. **faster-whisper 替换**：**不推荐**。qwen 路径的方言提示
   （`SICHUAN_DIALECT_HINTS`）、语言标签、热词机制都是 Qwen 专属；
   Whisper 换掉后 22 类中文方言与语言识别管线要整套重做，收益
   （CTranslate2 上本就快）不抵重构面。
4. **VAD 与 ASR 流水线并行**：现状 VAD 出段后 ASR 才开始，但 VAD 本身
   毫秒级、不构成瓶颈；真正的段间空闲已被优先级队列利用。无可做。

## 本次动作

- 测试基线：74/74 passed（10s）。
- 本评估文档入库。
- 生产代码零改动（单流/优先级/int8 翻译现状均有代码依据，非猜测）。
