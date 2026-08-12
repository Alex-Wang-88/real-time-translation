from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from realtime_meeting.config import load_settings
from realtime_meeting.jimo import MeetingSummarizer, TodoGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real Jimo summary + todo smoke test without printing secrets.")
    parser.add_argument("--env-file", default=".env", help="dotenv file to load; .env.example is supported")
    parser.add_argument(
        "--fixture",
        default="tests/fixtures/sample_meeting.jsonl",
        help="synthetic transcript JSONL fixture",
    )
    return parser.parse_args()


async def run(env_file: Path, fixture: Path) -> dict[str, object]:
    load_dotenv(env_file, override=True)
    settings = load_settings()
    if not settings.jimo_configured:
        raise RuntimeError("会议纪要 Jimo 配置不完整")
    if not settings.todo_configured:
        raise RuntimeError("To-do-list Jimo 配置不完整")
    if not fixture.is_file():
        raise FileNotFoundError(f"测试数据不存在: {fixture}")

    meeting_id = f"smoke-{uuid4().hex}"
    started_at = datetime(2026, 8, 12, 9, 0, tzinfo=timezone.utc).isoformat()
    ended_at = datetime(2026, 8, 12, 9, 1, tzinfo=timezone.utc).isoformat()
    summary = await MeetingSummarizer(settings).summarize(
        fixture,
        meeting_id,
        started_at,
        ended_at,
        attempt_id=f"attempt-{uuid4().hex}",
    )
    if not summary.strip():
        raise RuntimeError("会议纪要节点返回空内容")
    todo = await TodoGenerator(settings).generate(meeting_id, 1, summary)
    return {
        "status": "ok",
        "meeting_id": meeting_id,
        "summary_chars": len(summary),
        "todo_items": len(todo.items),
        "fixture": str(fixture),
    }


def main() -> int:
    args = parse_args()
    env_file = Path(args.env_file).resolve()
    fixture = Path(args.fixture).resolve()
    try:
        result = asyncio.run(run(env_file, fixture))
    except Exception as exc:  # noqa: BLE001 - smoke test reports a concise safe failure
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
