# 平台会议智能体提示词

平台内配置了“完整逐句转写（精修）→会议纪要、待办事项”三个节点。客户端只发送一次完整 JSON 请求，平台结束节点必须返回三个可解析的固定区块：

```text
@@JIMO_SECTION:DATA:BEGIN@@
完整逐句转写输出
@@JIMO_SECTION:DATA:END@@
@@JIMO_SECTION:SUMMARY:BEGIN@@
会议纪要输出
@@JIMO_SECTION:SUMMARY:END@@
@@JIMO_SECTION:TODOLIST:BEGIN@@
待办事项输出
@@JIMO_SECTION:TODOLIST:END@@
```

客户端侧的请求提示词位于 `realtime_meeting/prompts.py` 的
`MEETING_AGENT_REQUEST_PROMPT`。平台节点负责内容质量和三个区块的生成；客户端只负责发送音频链接或本地实时转写上下文、解析区块、渲染和保存结果。

不要返回 JSON、代码围栏或区块之外的说明文字。`DATA`、`SUMMARY`、`TODOLIST` 的内部格式可以继续使用平台节点已经配置的 Markdown 模板。
