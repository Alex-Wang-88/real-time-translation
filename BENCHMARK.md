# 实时会议翻译基准与验收记录

## 当前已知基线

历史测试日期：2026-08-07
GPU：NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB
输入：16 kHz、单声道、PCM16；中文/英文/德文连续切换，真实时间 WebSocket 回放
旧模型：Whisper `large-v3-turbo` + NLLB-200 distilled 1.3B CT2 int8 + Resemblyzer

旧基线只用于对比调度重构前后的变化，不代表当前 FunASR/OPUS-MT 配置的准确率或延迟承诺。此前关闭精修仍出现约 8–12 秒逐句延迟，启用双模型精修出现约 20–32 秒积压，因此当前验收必须分别记录 draft、稳定原文、翻译和 refine 阶段。

## 基准集

正式验收需要准备带人工参考文本的 WAV/FLAC，并在 `test_data/benchmarks/` 下按场景组织：

| 子集 | 必测内容 | 主要指标 |
| --- | --- | --- |
| `zh_mandarin` | 普通话、数字、专有名词 | CER、首个 partial、稳定句 p50/p95 |
| `zh_dialect` | 粤语、吴语、闽语等方言 | 分方言 CER、fallback 次数 |
| `en` | 清晰英语、口音英语 | WER、稳定句延迟 |
| `de` | 清晰德语、长句和复合词 | WER、Whisper 路由比例 |
| `mixed` | 中英德切换、同句混合 | 语言切换错误率、翻译延迟 |
| `noise_far_field` | 噪声、远场、回声 | CER/WER 下降、VAD speech ratio |
| `overlap` | 两人重叠发言 | 记录可用性和说话人临时 ID 稳定性 |

每条样本应保存 `reference.txt` 和元数据（语言、说话人数、信噪比、时长、采样率），禁止把真实会议录音提交到公开仓库。

## 必记录指标

- 音频包总数、丢包数、乱序数、音频时长和 VAD speech ratio；
- 首个 partial 延迟、稳定句 p50/p95、翻译 p50/p95；
- 实时 ASR 队列深度、精修队列长度/最老年龄、停止后排空耗时；
- FunASR/Whisper fallback 次数、模型加载/卸载/OOM 事件；
- GPU 峰值显存，必须不超过 `MEETING_GPU_MEMORY_BUDGET_MB=7200`；
- 会议结束到 JSON/Markdown/音频导出完成的时间。

## 验收阈值

- 1 倍速回放 30 分钟音频时，实时 ASR 队列不持续增长；
- 首个 partial p95 ≤ 1.5 秒；稳定句 p95 ≤ 2.5 秒；
- 已部署常用语种翻译在原文事件后 p95 ≤ 1.5 秒；
- 音频包丢失率 ≤ 0.1%，不能丢失最终句；
- RTX 5060 峰值显存 ≤ 7.2 GB，无 OOM、死锁、SQLite 锁死或 WebSocket 长时间无响应；
- 停止后精修完成时间不超过音频时长 + 60 秒；
- 未覆盖翻译语种必须显示原文和 `unsupported`，不能伪造翻译。

## 可复现命令

```powershell
.venv\Scripts\python.exe -m pytest -q
python -m compileall realtime_meeting scripts tests
node --check realtime_meeting/web/app.js
node --check realtime_meeting/web/audio-worklet.js
python scripts/replay_audio.py test_data/live_zh_en_de_30s.wav
```

回放工具会逐行输出 WebSocket 事件；正式基准还应把 `/api/metrics` 和会议导出目录保存为同一份实验记录。
