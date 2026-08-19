"""Replay WSC-Eval-ASR clips through the real-time meeting session.

This is an ASR benchmark runner, so it does not call the meeting-minutes and
todo APIs for every isolated clip.  The existing manufacturing meeting replay
remains the full end-to-end contract test.  WSC replay focuses on real audio,
speech segmentation, Sichuan routing and the dual-text evaluator.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from collections import Counter
from pathlib import Path

from realtime_meeting.config import load_settings
from realtime_meeting.evaluation import evaluate_realtime_replay
from realtime_meeting.runtime import LiveModelRuntime
from realtime_meeting.session import LiveMeetingSession
from realtime_meeting.sichuan_eval import validate_wsc_eval_manifest
from realtime_meeting.storage import LocalMeetingStore


SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2


async def replay(
    manifest_path: Path,
    output_root: Path,
    chunk_seconds: float,
    playback_rate: float,
    pause_seconds: float,
    speech_variant_mode: str,
) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = validate_wsc_eval_manifest(manifest)
    if errors:
        raise ValueError("manifest 校验失败:\n- " + "\n- ".join(errors))
    samples = [item for item in manifest["samples"] if isinstance(item, dict)]
    if not samples:
        raise ValueError("manifest 没有 samples")

    settings = load_settings()
    settings.results_dir = output_root.resolve() / "meetings"
    settings.keep_audio = True
    runtime = LiveModelRuntime(
        settings.asr_primary,
        settings.asr_fallback,
        settings.device,
        language_id_model=settings.language_id_model,
        asr_autodownload=settings.asr_autodownload,
        translation_model_root=settings.translation_model_root,
        translation_autodownload=False,
        translation_warmup=settings.translation_warmup,
        vad_model=settings.vad_model,
        single_model=settings.single_asr_model,
    )
    runtime.load(print)
    if not runtime.ready or runtime.primary is None:
        raise RuntimeError(f"ASR runtime is not ready: {runtime.status}")

    subset = str((manifest.get("dataset") or {}).get("subset") or "wsc").casefold()
    meeting_id = f"sichuan-wsc-{subset}-realtime"
    store = LocalMeetingStore(settings.results_dir)
    store.delete(meeting_id)
    meeting = LiveMeetingSession(
        settings,
        runtime,
        store,
        meeting_id=meeting_id,
        title=f"WSC-Eval 四川方言 {subset} 实时评测",
    )
    meeting.configure_meeting_settings({"speech_variant_mode": speech_variant_mode})
    started = time.perf_counter()
    sequence = 0
    feed_started = 0.0
    next_chunk_deadline = 0.0
    chunk_frames = max(1, round(SAMPLE_RATE * chunk_seconds))

    async def feed(pcm: bytes) -> None:
        nonlocal sequence, next_chunk_deadline
        if not pcm:
            return
        await meeting.feed_audio(pcm, sequence=sequence, source_id="wsc-eval")
        sequence += 1
        if playback_rate > 0:
            duration = len(pcm) / (SAMPLE_WIDTH * SAMPLE_RATE)
            next_chunk_deadline += duration / playback_rate
            delay = next_chunk_deadline - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

    try:
        await meeting.start()
        meeting.configure_audio(
            {"sample_rate": SAMPLE_RATE, "channels": CHANNELS, "encoding": "pcm_s16le"},
            source_id="wsc-eval",
        )
        feed_started = time.perf_counter()
        next_chunk_deadline = feed_started
        silence_frames = max(1, round(SAMPLE_RATE * pause_seconds))
        silence = b"\0" * (silence_frames * SAMPLE_WIDTH)
        for sample_index, sample in enumerate(samples):
            audio_path = (manifest_path.parent / str(sample["audio_path"])).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            with wave.open(str(audio_path), "rb") as source:
                if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (
                    SAMPLE_RATE,
                    CHANNELS,
                    SAMPLE_WIDTH,
                ):
                    raise ValueError(f"WSC 音频必须是 16kHz/单声道/PCM16: {audio_path}")
                while True:
                    pcm = source.readframes(chunk_frames)
                    if not pcm:
                        break
                    await feed(pcm)
            if sample_index < len(samples) - 1:
                for offset in range(0, silence_frames, chunk_frames):
                    await feed(silence[offset * SAMPLE_WIDTH : min(silence_frames, offset + chunk_frames) * SAMPLE_WIDTH])

        feed_wall_seconds = time.perf_counter() - feed_started
        await meeting.request_stop("wsc_eval_replay")
        if meeting.stop_task:
            await meeting.stop_task
        paragraphs = meeting.load_transcript()
        output_root.mkdir(parents=True, exist_ok=True)
        output_dir = store.meeting_dir(meeting_id)
        payload = {
            "schema_version": "1.0-sichuan-wsc-realtime-replay",
            "source_manifest": str(manifest_path),
            "meeting_id": meeting_id,
            "recording_state": meeting.recording_state,
            "recording_seconds": round(meeting.audio_samples_received / SAMPLE_RATE, 3),
            "replay_wall_seconds": round(time.perf_counter() - started, 3),
            "input_chunk_seconds": chunk_seconds,
            "input_chunk_count": sequence,
            "playback_rate": playback_rate,
            "pause_seconds": pause_seconds,
            "pacing_mode": "realtime" if playback_rate == 1.0 else ("max_speed" if playback_rate <= 0 else "accelerated"),
            "speech_variant_mode": speech_variant_mode,
            "feed_wall_seconds": round(feed_wall_seconds, 3),
            "expected_speech_segments": len(samples),
            "paragraph_count": len(paragraphs),
            "languages": dict(Counter(item.language for item in paragraphs)),
            "speech_variants": dict(Counter(str(item.speech_variant or "none") for item in paragraphs)),
            "translation_statuses": dict(Counter(item.translation_status for item in paragraphs)),
            "pipeline_metrics": meeting.pipeline_metrics,
            "runtime_metrics": runtime.metrics,
            "postprocess_api": None,
            "paragraphs": [item.to_dict() for item in paragraphs],
            "meeting_output_dir": str(output_dir),
        }
        payload["automatic_evaluation"] = evaluate_realtime_replay(manifest, payload)
        report_path = output_root / "realtime_replay_report.json"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {report_path}")
        print(
            json.dumps(
                {
                    key: payload[key]
                    for key in (
                        "recording_state",
                        "recording_seconds",
                        "replay_wall_seconds",
                        "paragraph_count",
                        "languages",
                        "speech_variants",
                        "automatic_evaluation",
                    )
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return payload
    finally:
        runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("result/benchmarks/sichuan_wsc_realtime"))
    parser.add_argument("--chunk-seconds", type=float, default=0.5)
    parser.add_argument("--playback-rate", type=float, default=0.0, help="1.0 为真实时间，0 为最快回放")
    parser.add_argument("--pause-seconds", type=float, default=1.0, help="样本之间的静音，用于模拟说话间停顿")
    parser.add_argument("--speech-variant-mode", choices=("auto", "sichuan"), default="sichuan")
    args = parser.parse_args()
    if args.chunk_seconds <= 0 or args.chunk_seconds > 8:
        parser.error("--chunk-seconds must be > 0 and <= 8")
    if args.playback_rate < 0 or args.playback_rate > 8:
        parser.error("--playback-rate must be >= 0 and <= 8")
    if args.pause_seconds < 0 or args.pause_seconds > 10:
        parser.error("--pause-seconds must be >= 0 and <= 10")
    asyncio.run(
        replay(
            args.manifest,
            args.output.resolve(),
            args.chunk_seconds,
            args.playback_rate,
            args.pause_seconds,
            args.speech_variant_mode,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
