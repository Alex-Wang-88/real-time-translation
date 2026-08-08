# real-time-translation：本机实时会议转译

这是一个独立的 Windows 桌面应用：用 PyQt6 采集麦克风，在本机 GPU 上运行语音识别、语言识别、说话人编号和中文翻译。会议过程中实时显示“原文 + 简体中文翻译”，停止录音后先保存完整原稿，再由用户手动点击按钮请求积墨 AI 生成中文会议纪要。

音频默认只在本机处理；积墨 AI 在用户点击“生成会议纪要”后才会收到文本。项目面向单机单会议场景，不依赖浏览器，也不需要公网部署。

## 主要功能

- 实时麦克风采集、音量反馈和 16 kHz 单声道转换；
- Whisper `large-v3-turbo` 语音识别，保留原文语言；
- 全语种自动切换，支持 Whisper 官方 100 种输入语言，中文、英文、德文为必选语言；
- NLLB 1.3B INT8 本地翻译，目标语言固定为简体中文；
- 匿名说话人编号（演讲人1、演讲人2……），不推断真实姓名；
- 约 2–6 秒典型稳定输出延迟，连续发言自动分段；
- 录音按 30 分钟轮转保存，逐句稿立即写入 JSONL，异常退出后可恢复；
- 停止后手动生成积墨 AI 中文会议纪要，并实时显示 SSE 输出；
- 生成 Markdown、JSON、JSONL、音频清单和各语言原稿，方便下载和二次处理。

## 输入语言范围

识别模型使用的是多语种 Whisper `large-v3-turbo`，不是 `*.en` 英语专用模型。它可以识别官方语言列表中的 100 种语言，包括中文（含粤语）、英文、德文、日文、韩文、俄文、乌克兰文、法文、西班牙文、葡萄牙文、意大利文、荷兰文、北欧语言、阿拉伯文、波斯文、印地文、孟加拉文、泰文、越南文、印尼文、土耳其文、希伯来文、希腊文、波兰文、捷克文以及更多语言。

语言判定由三层共同完成：

1. Whisper 按每个稳定音频片段重新检测语言，启用 `multilingual=True` 和明确的 `task=transcribe`；
2. 全语种 Lingua 对识别文本进行第二次判定，补充短句、噪声和拉丁字母语言；
3. 中文、日文、韩文、阿拉伯文、斯拉夫文字、印度文字、泰文、老挝文、缅甸文、希腊文、亚美尼亚文、格鲁吉亚文等脚本使用确定性规则纠偏。

本地 NLLB 翻译映射覆盖 Whisper 100 种语言中的 97 种，翻译目标固定为简体中文。布列塔尼文（`br`）、夏威夷文（`haw`）和拉丁文（`la`）在当前 NLLB 词表中没有可用源语言标签；这些语言仍会保留正确的原文和语言标记，但中文翻译行会原样保留，避免误翻成英文。增加其他语言时只需在 `realtime_meeting/runtime.py` 的 `NLLB_CODES` 中加入对应 NLLB 标签。

## 环境要求

### 推荐配置

- Windows 10/11 64 位；
- Python 3.11（支持 3.10–3.12）；
- NVIDIA RTX 5060 或同级显卡，显存建议至少 8 GB；
- CUDA 12.x 驱动，能够被 CTranslate2 和 PyTorch 识别；
- 16 GB 以上内存，至少 8 GB 可用磁盘空间；
- 可用麦克风，并允许 Python 访问麦克风。

### CPU 模式

没有 NVIDIA GPU 时可以把 `.env` 中的 `MEETING_DEVICE` 改为 `cpu`。功能仍然可用，但模型加载和逐句翻译会明显变慢，不建议用于长时间实时会议。

首次启动会从 Hugging Face 下载模型：ASR 模型和约 1.4 GB 的 NLLB 1.3B INT8 翻译模型。请预留下载时间、磁盘空间，并确保网络可以访问 Hugging Face。

## 安装与启动

在 PowerShell 中执行：

```powershell
git clone https://github.com/Alex-Wang-88/real-time-translation.git
cd real-time-translation
Copy-Item .env.example .env
notepad .env
.\start.ps1
```

第一次运行 `start.ps1` 会自动创建 `.venv` 并安装依赖。脚本默认启动 PyQt6 桌面窗口，并在本机启动 FastAPI 后端；不需要手动打开浏览器。

只启动后端进行接口调试：

```powershell
.\start.ps1 -ServerOnly
```

旧参数 `-NoBrowser` 是 `-ServerOnly` 的兼容别名。

## 配置 `.env`

`.env` 只存在于本机，禁止提交 GitHub。至少需要填写积墨接口的地址和完整原始 Authorization 值：

```dotenv
JIMO_API_URL=https://jimoai-bot-api.xiaohuodui.cn/v2/chat/completions/share?shareId=你的shareId
JIMO_AUTHORIZATION=完整的原始Authorization值
```

常用配置如下：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MEETING_DEVICE` | `auto` | `auto`、`cuda` 或 `cpu` |
| `MEETING_ASR_MODEL` | `large-v3-turbo` | faster-whisper 模型名 |
| `MEETING_TRANSLATION_MODEL` | `JustFrederik/nllb-200-distilled-1.3B-ct2-int8` | 本地中文翻译模型 |
| `MEETING_HOST` | `127.0.0.1` | 只监听本机 |
| `MEETING_PORT` | `8765` | 本地 API 端口 |
| `MEETING_API_TOKEN` | 空 | 非本机监听时必填；浏览器使用 `?token=...` |
| `MEETING_RESULTS_DIR` | `result/live` | 会议输出目录 |
| `MEETING_MAX_UTTERANCE_SECONDS` | `5` | 连续发言的最大稳定分段长度 |
| `MEETING_AUDIO_SEGMENT_MINUTES` | `30` | 音频轮转分片时长 |
| `MEETING_MAX_AUDIO_PACKET_BYTES` | `262144` | WebSocket 单个 PCM 音频包上限 |
| `MEETING_INFERENCE_QUEUE_SIZE` | `64` | 实时推理队列上限 |
| `JIMO_MAX_REQUEST_CHARS` | `12000` | 单次积墨请求字符上限 |

正式使用前请轮换曾经出现在聊天、日志或截图中的密钥。仓库的 `.gitignore` 已排除 `.env`、`.venv`、`result`、缓存和编译文件。

关于这些未提交文件的原因、恢复方式和安全注意事项，见 [LOCAL_CONFIG.md](LOCAL_CONFIG.md)。

## 使用流程

1. 等待窗口顶部显示“后端：已就绪”，并确认 GPU 与积墨状态。
2. 在输入设备下拉框选择实际麦克风。说话时音量条应明显变化。
3. 点击“开始会议”。系统会按稳定语音片段显示：

   ```text
   [00:00:01.170 - 00:00:03.820] 演讲人1（德文）：“Guten Morgen.”
   [00:00:01.170 - 00:00:03.820] 演讲人1（中文翻译）：“早上好。”
   ```

4. 点击“停止会议”。尾部音频会先处理并保存，应用不会自动请求 AI。
5. 检查实时逐句稿后，点击“生成会议纪要”，积墨 AI 的中文 SSE 内容会流式显示。

### 音量为 0 的排查顺序

1. 在下拉框切换同一麦克风的 WASAPI、DirectSound 或其他输入端点；
2. 检查 Windows“设置 → 隐私和安全 → 麦克风”中的 Python 权限；
3. 检查硬件静音键、系统输入音量和应用独占模式；
4. 查看窗口中的“前端帧数 / 后端包数”，确认音频是否送达后端；
5. 如果设备不支持 16 kHz，应用会自动重采样 44.1/48 kHz 输入，不应因此被过滤。

## 输出文件

每场会议保存在 `result/live/<时间>-<会话 ID>/`：

- `transcript.jsonl`：逐句追加的完整稳定记录；
- `transcript.json`：完整会议结构化数据，翻译字段为 `translation_zh`；
- `meeting_transcript.md`：原文和中文翻译交替排列的完整稿；
- `translated_zh.md`：简体中文完整译稿；
- `original_*.md`：按检测语言导出的原稿；
- `meeting_minutes.md`：手动请求积墨 AI 后生成的中文会议纪要；
- `audio/`、`audio_manifest.json`：轮转音频分片和清单；
- `manifest.json`：本场会议状态和文件清单。

没有检测到有效语音时仍会保存录音，但不会把空内容发送给积墨 AI。

## 本地 API

后端默认监听 `http://127.0.0.1:8765`，主要接口如下：

- `GET /api/health`：模型、GPU、磁盘和会议状态；
- `POST /api/meetings`：创建唯一活动会议；
- `WS /api/meetings/{id}/stream`：发送 PCM16 音频并接收状态、逐句稿和翻译事件；
- `POST /api/meetings/{id}/stop`：停止录音并保存原稿；
- `POST /api/meetings/{id}/retry-summary`：手动重试会议纪要；
- `GET /api/meetings/{id}`：恢复会议状态；
- `GET /api/meetings/{id}/files/{name}`：下载白名单文件。

浏览器静态页面仅保留作 API 调试资源，正式入口是 PyQt6 桌面程序。

## 测试与性能

```powershell
.venv\Scripts\python.exe -m pytest -q
```

当前测试覆盖 VAD 分段、语言切换、脚本语言纠偏、中文翻译目标、SSE 解析、导出文件和 WebSocket 会话流程。RTX 5060 实测可在 2–6 秒内得到稳定原文和中文翻译；噪声、多人重叠和麦克风距离会影响识别质量与延迟。

## 安全提示

- 不要把 `.env`、Authorization 值、模型缓存、录音和 `result/` 提交到 GitHub；
- 如果密钥曾经出现在聊天、截图或日志中，请立即在服务端轮换；
- 应用默认只监听 `127.0.0.1`；如果修改为其他监听地址，必须配置 `MEETING_API_TOKEN`，并使用带 `?token=...` 的 Web 页面地址。
