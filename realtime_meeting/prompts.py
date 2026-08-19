"""Request-side instructions for the configured platform meeting agent."""


# The platform agent owns the three-node workflow and its end node emits three
# machine-separable Markdown sections. Keep this prompt short so it does not
# compete with the platform-configured node prompts.
MEETING_AGENT_REQUEST_PROMPT = r"""当前请求调用的是一个已经在平台配置完成的多节点会议智能体。

请严格执行平台中配置的“完整逐句转写（精修）→会议纪要、待办事项”流程。最终回答必须直接使用平台结束节点配置的三个分隔区块：

@@JIMO_SECTION:DATA:BEGIN@@
完整逐句转写输出
@@JIMO_SECTION:DATA:END@@
@@JIMO_SECTION:SUMMARY:BEGIN@@
会议纪要输出
@@JIMO_SECTION:SUMMARY:END@@
@@JIMO_SECTION:TODOLIST:BEGIN@@
待办事项输出
@@JIMO_SECTION:TODOLIST:END@@

不要输出 JSON、代码围栏、额外说明或分隔区块之外的内容。"""
