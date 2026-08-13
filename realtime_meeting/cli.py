from __future__ import annotations

import argparse

import uvicorn

from .config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="实时会议记录 v2")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    reload_group = parser.add_mutually_exclusive_group()
    reload_group.add_argument("--reload", dest="reload", action="store_true")
    reload_group.add_argument("--no-reload", dest="reload", action="store_false")
    parser.set_defaults(reload=False)
    args = parser.parse_args()
    settings = load_settings()
    uvicorn.run(
        "realtime_meeting.server:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
    )
