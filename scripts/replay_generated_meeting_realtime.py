"""Replay a generated meeting through the project's LiveMeetingSession."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from collections import Counter
from pathlib import Path

import httpx

from realtime_meeting.config import load_settings
from realtime_meeting.evaluation import evaluate_realtime_replay
from realtime_meeting.runtime import LiveModelRuntime
from realtime_meeting.server import create_app
from realtime_meeting.session import LiveMeetingSession
from realtime_meeting.storage import LocalMeetingStore


def _response_payload(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return response.text[:2000]


async def _post_api(client: httpx.AsyncClient, path: str) -> dict[str, object]:
    started = time.perf_counter()
    try:
        response = await client.post(path)
        return {
            "method": "POST",
            "path": path,
            "status_code": response.status_code,
            "response": _response_payload(response),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    except Exception as exc:  # noqa: BLE001 - preserve API failures in the report
        return {
            "method": "POST",
            "path": path,
            "status_code": None,
            "response": None,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


async def _wait_for_task(task: asyncio.Task[object] | None, timeout: float) -> dict[str, object]:
    if task is None:
        return {"present": False, "status": "not_created"}
    started = time.perf_counter()
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=max(1.0, timeout))
        return {
            "present": True,
            "status": "complete",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    except asyncio.TimeoutError:
        return {
            "present": True,
            "status": "timeout",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    except Exception as exc:  # noqa: BLE001 - preserve task failures in the report
        return {
            "present": True,
            "status": "failed",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


async def _request_postprocess_apis(
    settings: object,
    runtime: LiveModelRuntime,
    store: LocalMeetingStore,
    meeting_id: str,
) -> dict[str, object]:
    """Simulate the user's single manual post-recording request."""
    app = create_app(settings, runtime=runtime, load_models=False, store=store)
    api_meeting = app.state.manager.get(meeting_id)
    if api_meeting is None:
        raise RuntimeError(f"回放完成后无法从 API 会话管理器恢复会议: {meeting_id}")

    headers: dict[str, str] = {}
    if settings.api_auth_required and settings.api_token.strip():
        headers["Authorization"] = f"Bearer {settings.api_token.strip()}"
    task_timeout = max(30.0, settings.jimo_timeout_seconds * max(2, settings.jimo_max_retries))
    summary_path = f"/api/v2/meetings/{meeting_id}/summary"
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", headers=headers) as client:
        summary_request = await _post_api(client, summary_path)
        summary_wait = await _wait_for_task(api_meeting.summary_task, task_timeout)

    snapshot = api_meeting.snapshot()
    return {
        "transport": "in_process_httpx_asgi",
        "request_count": 1,
        "request": {
            "request": summary_request,
            "task": summary_wait,
        },
        "final_state": {
            "recording_state": snapshot.get("recording_state"),
            "summary_state": snapshot.get("summary_state"),
            "summary_revision": snapshot.get("summary_revision"),
            "summary_chars": len(str(snapshot.get("summary") or "")),
            "todo_state": snapshot.get("todo_state"),
            "todo_items": len((snapshot.get("todo") or {}).get("items") or []),
            "summary_error": snapshot.get("summary_error"),
            "todo_error": snapshot.get("todo_error"),
            "agent_error": snapshot.get("agent_error"),
        },
    }


async def replay(
    manifest_path: Path,
    output_root: Path,
    chunk_seconds: float,
    playback_rate: float,
    speech_variant_mode: str = "auto",
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audio_path = (manifest_path.parent / str(manifest["full_audio_path"])).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

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

    meeting_id = "manufacturing-role-meeting-v3-realtime"
    store = LocalMeetingStore(settings.results_dir)
    store.delete(meeting_id)
    meeting = LiveMeetingSession(
        settings,
        runtime,
        store,
        meeting_id=meeting_id,
        title="制造业月会 v3 实时回放测试",
    )
    meeting.configure_meeting_settings({"speech_variant_mode": speech_variant_mode})
    started = time.perf_counter()
    try:
        await meeting.start()
        meeting.configure_audio({"sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"}, source_id="replay")
        chunk_frames = max(1, round(16_000 * chunk_seconds))
        sequence = 0
        feed_started = time.perf_counter()
        next_chunk_deadline = feed_started
        with wave.open(str(audio_path), "rb") as source:
            while True:
                pcm = source.readframes(chunk_frames)
                if not pcm:
                    break
                await meeting.feed_audio(pcm, sequence=sequence, source_id="replay")
                sequence += 1
                if playback_rate > 0:
                    chunk_duration = len(pcm) / (2 * 16_000)
                    next_chunk_deadline += chunk_duration / playback_rate
                    delay = next_chunk_deadline - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(delay)
        feed_wall_seconds = time.perf_counter() - feed_started
        await meeting.request_stop("generated_audio_replay")
        if meeting.stop_task:
            await meeting.stop_task
        paragraphs = meeting.load_transcript()
        output_dir = store.meeting_dir(meeting_id)
        postprocess_api = await _request_postprocess_apis(settings, runtime, store, meeting_id)
        payload = {
            "schema_version": "1.0-generated-realtime-replay",
            "source_manifest": str(manifest_path),
            "source_audio": str(audio_path),
            "meeting_id": meeting_id,
            "recording_state": meeting.recording_state,
            "recording_seconds": round(meeting.audio_samples_received / 16_000, 3),
            "replay_wall_seconds": round(time.perf_counter() - started, 3),
            "input_chunk_seconds": chunk_seconds,
            "input_chunk_count": sequence,
            "playback_rate": playback_rate,
            "pacing_mode": "realtime" if playback_rate == 1.0 else ("max_speed" if playback_rate <= 0 else "accelerated"),
            "speech_variant_mode": meeting.meeting_settings.get("speech_variant_mode", "auto"),
            "feed_wall_seconds": round(feed_wall_seconds, 3),
            "expected_speech_segments": len(manifest.get("samples") or []),
            "paragraph_count": len(paragraphs),
            "languages": dict(Counter(item.language for item in paragraphs)),
            "speech_variants": dict(Counter(str(item.speech_variant or "none") for item in paragraphs)),
            "translation_statuses": dict(Counter(item.translation_status for item in paragraphs)),
            "pipeline_metrics": meeting.pipeline_metrics,
            "runtime_metrics": runtime.metrics,
            "postprocess_api": postprocess_api,
            "paragraphs": [item.to_dict() for item in paragraphs],
            "meeting_output_dir": str(output_dir),
        }
        payload["automatic_evaluation"] = evaluate_realtime_replay(manifest, payload)
        output_root.mkdir(parents=True, exist_ok=True)
        report_path = output_root / "realtime_replay_report.json"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {report_path}")
        print(json.dumps({key: payload[key] for key in ("recording_state", "recording_seconds", "replay_wall_seconds", "paragraph_count", "languages", "speech_variants", "translation_statuses", "postprocess_api")}, ensure_ascii=False, indent=2))
        return payload
    finally:
        runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("result/benchmarks/manufacturing_role_meeting_v3/realtime"))
    parser.add_argument("--chunk-seconds", type=float, default=0.5)
    parser.add_argument(
        "--playback-rate",
        type=float,
        default=1.0,
        help="输入回放速度；1.0 为真实时间，>1 加速，0 为不等待的压力回放",
    )
    parser.add_argument(
        "--speech-variant-mode",
        choices=("auto", "sichuan"),
        default="auto",
        help="中文方言识别偏好；sichuan 只对中文启用四川方言提示，不改变英语/德语路由",
    )
    args = parser.parse_args()
    if args.chunk_seconds <= 0 or args.chunk_seconds > 8:
        parser.error("--chunk-seconds must be > 0 and <= 8")
    if args.playback_rate < 0 or args.playback_rate > 8:
        parser.error("--playback-rate must be >= 0 and <= 8")
    asyncio.run(
        replay(
            args.manifest.resolve(),
            args.output.resolve(),
            args.chunk_seconds,
            args.playback_rate,
            args.speech_variant_mode,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
