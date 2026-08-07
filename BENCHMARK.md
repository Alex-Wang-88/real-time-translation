# RTX 5060 本机验收记录

测试日期：2026-08-07
GPU：NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB
模型：Whisper `large-v3-turbo` + NLLB-200 distilled 1.3B CT2 int8 + Resemblyzer
输入：16 kHz、单声道、PCM16；中文/英文/德文连续切换，真实时间 WebSocket 回放

## 结果

- 模型常驻显存约 2.5 GB，8 GB 显存有足够余量，没有发生显存溢出。
- GPU 启动预热后，16 条稳定记录的端到端延迟：最小 2.118 秒、中位数 3.889 秒、P95 5.642 秒、最大 6.153 秒。
- 中文、英文、德文均被识别；目标翻译统一输出简体中文。
- 启动预热消除了首段约 9–11 秒的 CUDA 冷启动延迟。
- 连续发言默认每 5 秒强制稳定一次；1.5 秒临时字幕用于填补稳定结果前的等待。
- 4 小时加速测试验证了 30 分钟音频轮转、完整 JSONL 持久化和客户端/内存只保留最近 500 条。

结论：RTX 5060 Laptop GPU 适合这套单会议本机流程。更大的 1.3B INT8 翻译模型会牺牲少量延迟换取更可靠的中文翻译；清晰轮流发言通常能在 2–6 秒内得到稳定原文与中文译文。实际麦克风环境中的噪声、重叠发言和说话人距离仍会影响准确率与延迟。

## 可复现命令

服务启动后运行：

```powershell
python scripts/replay_audio.py test_data/live_zh_en_de_30s.wav
```

回放工具会逐行输出 WebSocket 事件，并在每条 `utterance` 上附加 `wall_latency_seconds`。
