# Model and pipeline benchmark

Create a JSON fixture with `samples`. Each sample can contain `audio`,
`duration_seconds`, `language`, `text`, `hypothesis`, `translation`, and
`translation_hypothesis`.

Run the dependency-free regression report with:

```powershell
\.venv\Scripts\python.exe scripts\benchmark.py tests\fixtures\benchmark.json --output result\benchmark.json
```

The report records WER/CER-style error rate, chrF-style translation overlap,
P50/P95 latency, and RTF. Real model runs can pass recognizer/translator
callbacks from a Python harness. Diarization fixtures should add DER/JER and
speaker-label stability measurements alongside this report so model choices
are based on the same fixed corpus rather than a single recording. Reports
should also record the fixed Resemblyzer parameters (16 kHz input,
1.6-second embedding context, 0.8-second hop, cosine threshold 0.68), model
weight size, and whether overlap metadata was produced. Resemblyzer is the only
speaker-separation implementation; it does not provide a dedicated overlap
model, so DER/JER fixtures must make that product boundary explicit.
