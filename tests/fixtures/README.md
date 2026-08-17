# Synthetic meeting fixture

`sample_meeting.jsonl` 是不含真实隐私的固定段落数据，供本地单元测试和可选的真实 Jimo smoke test 使用。

覆盖内容：

- 普通话、官方类别中的粤语/四川话示例、英文、德文原文；
- 英文和德文的中文翻译；
- 稳定的时间范围、`segment_id`、`source_revision` 和 `revision`；
- schema 2.0 段落字段，不含人员身份和旧后台处理字段。

文件格式与运行时 `transcript.jsonl` 相同，使用 `schema_version`、`event_type` 和
`paragraph` 事件。不包含 Authorization、模型密钥或真实会议内容。
