# 本地配置与未提交文件说明

本文件用于说明公开仓库中没有提交的本地文件。文件本身不包含真实密钥、录音或会议内容，可以安全放在公开仓库中。

## 为什么没有提交

| 文件或目录 | 未提交原因 | 如何获得 |
| --- | --- | --- |
| .env | 包含积墨 API 地址、Authorization 和本机参数，不能公开 | 从 .env.example 复制后在本机填写 |
| result/ | 包含会议录音、逐句稿、翻译稿和会议纪要，可能包含敏感会议内容 | 启动应用并完成一场会议后自动生成 |
| .venv/ | Windows、Python、显卡驱动相关的本机虚拟环境，体积大且不可跨机器复用 | 按 README 的安装命令自动创建 |
| Hugging Face 模型缓存 | 模型文件体积很大，不属于源代码 | 首次启动时自动下载并缓存 |
| __pycache__/、*.pyc、.pytest_cache/、*.egg-info/ | Python 运行时和测试产生的临时文件 | 运行应用或测试时自动生成 |

## 恢复本机配置

在项目根目录执行：

    Copy-Item .env.example .env
    notepad .env

至少填写以下两个变量：

    JIMO_API_URL=请填写积墨接口地址
    JIMO_AUTHORIZATION=请填写完整的原始Authorization值

不要自动添加 Basic 或 Bearer 前缀，程序会直接使用填写的原始值。完整配置项和启动方式见 README.md。

如果确实需要在本机保留一份“含实际密钥的 Markdown”，可以把已经填写好的 .env 复制为 LOCAL_CONFIG_PRIVATE.md。该文件已加入 .gitignore，只能留在本机，不能上传到公开仓库。

推荐的本地模型配置如下：

    MEETING_DEVICE=auto
    MEETING_ASR_MODEL=large-v3-turbo
    MEETING_TRANSLATION_MODEL=JustFrederik/nllb-200-distilled-1.3B-ct2-int8

## 安全注意事项

- 不要把 .env、Authorization、录音、result/ 或会议原稿上传到公开仓库。
- 如果密钥曾经出现在聊天、日志、截图或屏幕录制中，应在积墨服务端轮换后再正式使用。
- 公开仓库只保存源代码、示例配置和不含会议内容的测试素材。
- 提交前可以执行 git status --ignored，确认敏感文件仍被 .gitignore 忽略。
