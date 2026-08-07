"""Replay a 16 kHz mono PCM16 WAV file through the live WebSocket API."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import httpx
import websockets


async def replay(base_url: str, wav_path: Path, speed: float) -> None:
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        meeting = (await client.post("/api/meetings", json={"hotwords": ""})).raise_for_status().json()
        meeting_id = meeting["id"]
        started = time.perf_counter()

        async with websockets.connect(f"{ws_url}/api/meetings/{meeting_id}/stream") as socket:
            async def receive_events() -> None:
                async for message in socket:
                    event = json.loads(message)
                    if event.get("type") in {"partial", "utterance", "status", "error", "summary_complete"}:
                        if event.get("type") == "utterance":
                            item = event["utterance"]
                            event["wall_latency_seconds"] = round(
                                time.perf_counter() - started - float(item["end"]), 3
                            )
                        print(json.dumps(event, ensure_ascii=False), flush=True)
                    if event.get("type") == "summary_complete" or (
                        event.get("type") == "error" and event.get("code") == "summary_failed"
                    ):
                        return

            receiver = asyncio.create_task(receive_events())
            with wave.open(str(wav_path), "rb") as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                    raise ValueError("Replay input must be 16 kHz, mono, PCM16 WAV")
                frames_per_chunk = 320
                while chunk := wav.readframes(frames_per_chunk):
                    await socket.send(chunk)
                    await asyncio.sleep((frames_per_chunk / 16000) / speed)

            await asyncio.sleep(0.8)
            (await client.post(f"/api/meetings/{meeting_id}/stop", json={})).raise_for_status()
            try:
                await asyncio.wait_for(receiver, timeout=120)
            except asyncio.TimeoutError:
                receiver.cancel()

        snapshot = (await client.get(f"/api/meetings/{meeting_id}")).raise_for_status().json()
        print(json.dumps({"type": "snapshot", "meeting": snapshot}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    asyncio.run(replay(args.base_url.rstrip("/"), args.wav.resolve(), args.speed))


if __name__ == "__main__":
    main()
