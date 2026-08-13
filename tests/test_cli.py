from __future__ import annotations

import sys

from realtime_meeting import cli


def test_cli_disables_reload_by_default(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(sys, "argv", ["meeting-v2"])
    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: captured.update(kwargs))

    cli.main()

    assert captured["reload"] is False


def test_cli_reload_is_explicit_opt_in(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(sys, "argv", ["meeting-v2", "--reload"])
    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: captured.update(kwargs))

    cli.main()

    assert captured["reload"] is True
