# 企业部署与容量规划

## 首期拓扑

首期以单个 Linux NVIDIA GPU 容器运行 API、实时 ASR、低优先级精修和本地翻译。所有模型任务进入同一个优先队列：实时 ASR > 翻译 > 说话人识别 > 精修 > 导出。单 RTX 5060 默认只允许一个 GPU worker 和一个活动会议。

```
企业 OIDC/SSO -> 可信反向代理 -> REST API（换取短期 ticket）
                                      -> WebSocket 音频流
                                      -> 单 GPU 优先队列
                                      |-> SQLite WAL + JSONL + PCM/FLAC spool
                                      |-> WAV/JSONL/Markdown 持久卷
```

反向代理必须移除客户端自带的 `x-meeting-user` 与 `x-meeting-service-token`，完成 SSO 后重新写入，并限制后端只接受配置在 `MEETING_TRUSTED_PROXY_CIDRS` 中的来源。生产客户端不把长期 API token 放在 WebSocket URL；服务端签发 60 秒、单次使用的 stream ticket，并要求连接后 5 秒内发送 `auth` 消息。旧的 header/query token 解析只用于滚动升级兼容。

## 模型与许可证清单

生产包必须保存一份机器可读模型清单（建议 `models/MODEL_MANIFEST.json`），每个模型至少记录：名称、版本或 commit SHA、来源 URL、本地路径、许可证、商业使用状态和审核人/日期。CI 或发布脚本应阻止缺少清单项的模型进入生产镜像。

仓库提供 [MODEL_MANIFEST.example.json](MODEL_MANIFEST.example.json) 和校验脚本。复制并补齐真实 revision、许可证审核和商业使用结论后执行：

```bash
python scripts/check_model_manifest.py models/MODEL_MANIFEST.json --production
```

首期默认模型：

| 角色 | 默认模型 | 运行方式 |
| --- | --- | --- |
| 主 ASR | `FunAudioLLM/Fun-ASR-Nano-2512` | 中英、中文方言优先；运行时失败回退 Whisper |
| ASR 回退 | `large-v3-turbo` | 德语、长尾语种、低置信度和异常结果 |
| 精修 ASR | `large-v3` | 停止后低优先级；首次需要时懒加载 |
| VAD | `fsmn-vad` | FunASR FSMN-VAD；失败时能量 VAD |
| 翻译 | 审核后的 OPUS-MT pair | CTranslate2 INT8，CPU 或低优先级 GPU |

生产翻译只部署已审核的常用语言对：`en→zh`、`de→zh`、`ja→zh`、`ko→zh`、`fr→zh`、`es→zh`、`ru→zh`。目标语言固定为 `zh-CN`。未部署的语言必须返回原文和 `unsupported`。当前 NLLB 生产路径已移除；任何历史 NLLB 缓存都不能直接打包为商业生产模型，需按其模型卡许可证重新审核。

## 容器启动

```bash
docker build -f Dockerfile.server -t realtime-meeting:latest .
docker run --rm --gpus all -p 8765:8765 \
  -v meeting-data:/data -v meeting-models:/models \
  --env-file .env realtime-meeting:latest
```

宿主机需要 NVIDIA 驱动和 NVIDIA Container Toolkit。Docker 基础镜像和 PyTorch wheel 固定在 CUDA 12.8 系列；启动前应检查 `torch.version.cuda`、`nvidia-smi` 和 CTranslate2 CUDA 可用性。`/models` 保存 ASR/VAD/OPUS-MT 缓存，`/data` 保存录音、转写、SQLite 队列与恢复状态；两者都必须使用持久卷。

## 容量与扩展

“1000 人企业、峰值 50 路音频”不是单张 GPU 的承载承诺。先用目标音频和目标 GPU 测量实时、精修阶段的处理比（处理秒数 / 音频秒数），在不超过约 70% 持续 GPU 利用率和 7.2 GB 模型预算的前提下确定每副本安全流数。

首期通过以下配置执行硬限流：

- `MEETING_MAX_ACTIVE_MEETINGS`（兼容 `MEETING_MAX_CONCURRENT_MEETINGS`）；
- `MEETING_GPU_WORKERS=1`、`MEETING_GPU_MEMORY_BUDGET_MB=7200`；
- `MEETING_INFERENCE_QUEUE_SIZE`、`MEETING_REFINEMENT_QUEUE_SIZE`；
- `MEETING_MAX_PENDING_REFINEMENTS`、`MEETING_MAX_REFINEMENT_SPOOL_BYTES`。

实时 ASR 不得等待精修；停止录音后才排空全部精修和翻译任务。达到会议、任务或磁盘配额时，新会议返回 HTTP 429，已有会议继续保存和排空。需要多 GPU 或多节点时，保留 `ASRBackend`、`TranslationBackend`、`MeetingStore` 和 `InferenceScheduler` 接口，将本地存储替换为 Postgres、S3 兼容对象存储和 Redis/消息队列适配器；SQLite 不用于多副本共享。

## 运维检查

- `/health/live`：进程存活探针；
- `/health/ready`：模型、磁盘和 GPU 就绪探针；
- `/api/health`：需鉴权，包含模型路由、许可证配置状态、GPU 预算、队列和磁盘容量；
- `/api/metrics`：音频包、丢包/乱序、VAD、推理、翻译、精修和模型事件指标；
- 优雅终止应给精修和翻译任务留出时间；被中断的 `running` 任务会在下次启动时恢复为 `queued`。

生产部署应额外配置反向代理的 HTTPS、WebSocket 空闲超时、请求体上限、审计日志脱敏和结果目录加密/访问控制。
