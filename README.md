# real-time-translation：本机实时会议转译

这是一个由 FastAPI 提供的本机 Web 应用：浏览器使用 Web Audio/AudioWorklet 采集麦克风，在本机 GPU 上运行语音识别、语言识别、说话人编号和中文翻译。会议过程中先显示稳定原文，再异步补充简体中文翻译；停止录音后先保存完整原稿和精修结果，再由用户手动点击按钮请求积墨 AI 生成中文会议纪要。

音频默认只在本机处理；积墨 AI 在用户点击“生成会议纪要”后才会收到文本。项目支持 Windows/Linux 浏览器访问，也提供面向企业并发场景的 Linux NVIDIA 容器部署方式。

## 主要功能

- 浏览器麦克风设备选择、刷新、权限提示、音量反馈和带抗混叠滤波的 16 kHz 单声道转换；
- Fun-ASR-Nano 优先处理中英、中文方言和区域口音；德语使用 faster-whisper `large-v3-turbo`，实时模式暂只开放中文、英文和德语；
- `large-v3` 只作为低优先级停止后精修模型，不与实时 ASR 争抢 GPU；
- 本地 OPUS-MT 英文/德文对中文模型使用 CTranslate2 INT8 异步微批翻译；中文原文直接保留，其他语种暂不进入实时稿；
- 匿名说话人编号（演讲人1、演讲人2……），不推断真实姓名；
- 20 ms 音频帧、递增序号、丢包/乱序统计和稳定前缀 partial，连续发言自动分段；
- 录音按 30 分钟轮转保存，逐句稿立即写入 JSONL，异常退出后可恢复；
- 停止后手动生成积墨 AI 中文会议纪要，并实时显示 SSE 输出；
- 生成 Markdown、JSON、JSONL、音频清单和中英德原稿，方便下载和二次处理。

## 输入语言范围

实时模式只处理中文（含粤语等中文方言）、英文和德语。中文优先走 Fun-ASR-Nano，德语和主模型异常时使用 faster-whisper `large-v3-turbo`；上一句的语言不会强行锁定下一句，避免中文被错误解码成其他语言。其他语言暂时过滤，不会显示在实时逐句稿中。

语言判定由三层共同完成：

1. 首段和低置信度片段由语言门控决定 FunASR 或 Whisper 路由，正常连续语音不会对每个包运行双模型；
2. FunASR/Whisper 输出再由 Lingua 和脚本规则校验，补充短句、噪声和语言切换；
3. 中文字符和 Whisper 的 `yue` 粤语代码统一归为中文；英文和德语使用词汇规则与 Whisper 结果交叉校验。

生产翻译只使用经过许可证清单审核的 OPUS-MT 语言对：`en→zh`、`de→zh`。中文原文标记为 `not_needed`；其他语言暂时不进入实时翻译。模型目录、版本 SHA、来源和许可证见部署文档中的模型清单要求。

## 环境要求

### 推荐配置

- Windows 10/11 或 Linux；
- Python 3.11（依赖锁文件按 3.11 生成）；
- NVIDIA RTX 5060 或同级显卡，显存建议至少 8 GB；
- CUDA 12.x 驱动，能够被 CTranslate2 和 PyTorch 识别；
- 16 GB 以上内存，至少 8 GB 可用磁盘空间；
- Chrome、Edge 或其他支持 AudioWorklet 的现代浏览器；
- 可用麦克风，并允许浏览器访问麦克风。

浏览器麦克风通常只允许在 `localhost` 或 HTTPS 安全上下文中使用；通过局域网地址访问时请配置 HTTPS 反向代理。

### CPU 模式

没有 NVIDIA GPU 时可以把 `.env` 中的 `MEETING_DEVICE` 改为 `cpu`。功能仍然可用，但模型加载和逐句翻译会明显变慢，不建议用于长时间实时会议。

首次启动会加载 FunASR/Whisper ASR 模型。翻译默认只读取 `models/opus-mt` 下已审核的本地 CTranslate2 语言对，不会自动下载未经审核的翻译模型；请预留模型缓存空间。若在隔离环境中允许下载，必须显式设置 `MEETING_TRANSLATION_AUTODOWNLOAD=1`，并在上线前完成许可证审核。

## 安装与启动

在 PowerShell 中执行：

```powershell
git clone https://github.com/Alex-Wang-88/real-time-translation.git
cd real-time-translation
Copy-Item .env.example .env
notepad .env
.\start.ps1
```

第一次运行 `start.ps1` 会自动创建 `.venv` 并安装依赖，启动 FastAPI 服务并自动打开 Web 页面。

只启动 Web 服务、不自动打开浏览器：

```powershell
.\start.ps1 -NoBrowser
```

旧参数 `-ServerOnly` 仍作为 `-NoBrowser` 的兼容别名；也可以运行 `scripts/start_web.ps1`。

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
| `MEETING_ASR_PRIMARY` | `FunAudioLLM/Fun-ASR-Nano-2512` | 中英、中文方言优先模型；旧名 `MEETING_ASR_MODEL` 仍兼容 |
| `MEETING_ASR_FALLBACK` | `large-v3-turbo` | 德语、长尾语种和主模型异常时的回退模型 |
| `MEETING_ASR_REFINE` | `large-v3` | 停止后低优先级精修模型；旧名 `MEETING_REFINE_ASR_MODEL` 仍兼容 |
| `MEETING_ENABLE_REFINEMENT` | `1` | 是否启用异步精修 |
| `MEETING_VAD` | `fsmn-vad` | FunASR FSMN-VAD；加载失败时自动使用能量 VAD |
| `MEETING_GPU_WORKERS` | `1` | 单 GPU 全局优先队列 worker 数 |
| `MEETING_GPU_MEMORY_BUDGET_MB` | `7200` | GPU 模型预算；超限时优先释放精修模型 |
| `MEETING_MAX_CONCURRENT_MEETINGS` | `1` | 单实例同时录音会议上限 |
| `MEETING_MAX_PENDING_REFINEMENTS` | `10000` | 持久精修任务配额 |
| `MEETING_MAX_REFINEMENT_SPOOL_BYTES` | `21474836480` | 待处理 PCM 分片磁盘配额 |
| `MEETING_TRANSLATION_PROFILE` | `opusmt-local` | 本地 OPUS-MT 翻译配置 |
| `MEETING_TRANSLATION_MODEL_ROOT` | `models/opus-mt` | 审核过的 CTranslate2 语言对目录 |
| `MEETING_TRANSLATION_AUTODOWNLOAD` | `0` | 是否允许按官方模型 ID 下载；生产默认关闭 |
| `MEETING_TRANSLATION_TARGET` | `zh-CN` | 固定简体中文目标 |
| `MEETING_HOST` | `127.0.0.1` | 只监听本机 |
| `MEETING_PORT` | `8765` | 本地 API 端口 |
| `MEETING_API_TOKEN` | 空 | 非本机监听时必填；仅用于 REST 获取短期 stream ticket |
| `MEETING_RESULTS_DIR` | `result/live` | 会议输出目录 |
| `MEETING_MAX_UTTERANCE_SECONDS` | `8` | 连续发言的最大稳定分段长度 |
| `MEETING_AUDIO_SEGMENT_MINUTES` | `30` | 音频轮转分片时长 |
| `MEETING_MAX_AUDIO_PACKET_BYTES` | `262144` | WebSocket 单个 PCM 音频包上限 |
| `MEETING_INFERENCE_QUEUE_SIZE` | `64` | 实时推理队列上限 |
| `JIMO_MAX_REQUEST_CHARS` | `12000` | 单次积墨请求字符上限 |

企业部署、可信代理身份传递、容量测量和横向扩容边界见 [DEPLOYMENT.md](DEPLOYMENT.md)。

正式使用前请轮换曾经出现在聊天、日志或截图中的密钥。仓库的 `.gitignore` 已排除 `.env`、`.venv`、`result`、缓存和编译文件。

关于这些未提交文件的原因、恢复方式和安全注意事项，见 [LOCAL_CONFIG.md](LOCAL_CONFIG.md)。

## 使用流程

1. 等待 Web 页面顶部显示“后端：已就绪”，并确认 GPU、模型和积墨状态。
2. 在“输入设备”下拉框选择实际麦克风，必要时点击“刷新”。说话时音量条应明显变化。
3. 点击“开始会议”。浏览器会先通过 REST API 换取 60 秒、单次使用的 WebSocket stream ticket，再发送 `audio_config` 和带序号的 PCM16 音频包。系统会按稳定语音片段显示原文，并稍后异步补充翻译：

   ```text
   [00:00:01.170 - 00:00:03.820] 演讲人1（德文）：“Guten Morgen.”
   [翻译完成] 演讲人1（中文翻译）：“早上好。”
   ```

4. 点击“停止会议”。尾部音频会先处理并保存，应用不会自动请求 AI。
5. 检查实时逐句稿后，点击“生成会议纪要”，积墨 AI 的中文 SSE 内容会流式显示。

### 音量为 0 的排查顺序

1. 在 Web 页面的输入设备下拉框选择实际麦克风并点击“刷新”；
2. 检查浏览器站点权限和 Windows“设置 → 隐私和安全 → 麦克风”；
3. 检查硬件静音键、系统输入音量和浏览器是否被其他应用占用；
4. 查看页面中的“前端帧数 / 后端包数”，确认音频是否送达后端；
5. 如果设备不支持 16 kHz，浏览器会自动重采样 44.1/48 kHz 输入，不应因此被过滤。

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
- `GET /api/metrics`：音频包、推理队列、翻译、精修和 GPU 指标；
- `POST /api/meetings`：创建唯一活动会议；
- `POST /api/meetings/{id}/stream-ticket`：换取一次性短期 WebSocket ticket；
- `WS /api/meetings/{id}/stream`：发送 PCM16 音频并接收状态、逐句稿和翻译事件；
- `POST /api/meetings/{id}/stop`：停止录音并保存原稿；
- `POST /api/meetings/{id}/retry-summary`：手动重试会议纪要；
- `GET /api/meetings/{id}`：恢复会议状态；
- `GET /api/meetings/{id}/files/{name}`：下载白名单文件。

浏览器页面是唯一正式客户端；桌面端不再单独打包，后端启动后访问根路径即可使用。

## 测试与性能

```powershell
.venv\Scripts\python.exe -m pytest -q
```

当前测试覆盖 VAD 分段、语言切换、脚本语言纠偏、中文翻译目标、SSE 解析、导出文件、短期 WebSocket 认证和事件 revision 合并。历史 2–6 秒数据属于重构前单模型基线，不再作为当前性能承诺；请按照 [BENCHMARK.md](BENCHMARK.md) 记录 CER/WER、首个 partial、稳定句、翻译、队列和显存指标。噪声、多人重叠和麦克风距离会影响识别质量与延迟。

## 安全提示

- 不要把 `.env`、Authorization 值、模型缓存、录音和 `result/` 提交到 GitHub；
- 如果密钥曾经出现在聊天、截图或日志中，请立即在服务端轮换；
- 应用默认只监听 `127.0.0.1`；如果修改为其他监听地址，必须配置 `MEETING_API_TOKEN` 或可信反向代理身份。浏览器不会把长期 token 放进 WebSocket URL，而是先换取一次性 stream ticket；旧的 URL token 解析仅为兼容旧客户端。
