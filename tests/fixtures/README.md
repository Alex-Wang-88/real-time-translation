# Synthetic meeting fixture

`sample_meeting.jsonl` 是不含真实隐私的固定中英德会议数据，供本地单元测试和可选的真实 Jimo smoke test 使用。

覆盖内容：

- 中文、英文、德文原文；
- 英文和德文的中文翻译；
- 已确认的上线安排；
- 有明确负责人的行动项；
- 风险与未决问题；
- 稳定的时间范围、`segment_id` 和 revision。

文件格式与运行时 `transcript.jsonl` 相同，不包含 Authorization、模型密钥或真实会议内容。
