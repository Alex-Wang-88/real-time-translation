from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn

from .config import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动本机实时多语言会议转写与中文翻译 API 服务")
    parser.add_argument("--host", help="监听地址；默认读取 MEETING_HOST")
    parser.add_argument("--port", type=int, help="监听端口；默认读取 MEETING_PORT")
    parser.add_argument(
        "--browser",
        action="store_true",
        help="启动服务后自动打开 Web 页面",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="启动 Web 服务但不自动打开浏览器",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    host = args.host or settings.host
    port = args.port or settings.port
    if args.browser and not args.no_browser:
        display_host = "localhost" if host in {"127.0.0.1", "0.0.0.0"} else host
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{display_host}:{port}")).start()
    uvicorn.run(
        "realtime_meeting.server:create_default_app",
        factory=True,
        host=host,
        port=port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
