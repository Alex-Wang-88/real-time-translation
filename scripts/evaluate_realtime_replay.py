"""Evaluate a saved real-time replay report against its reference manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from realtime_meeting.evaluation import evaluate_realtime_replay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("replay_report", type=Path)
    parser.add_argument("--output", type=Path, help="输出 JSON 文件；默认写到回放报告旁边")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    report_path = args.replay_report.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evaluation = evaluate_realtime_replay(manifest, report)
    output_path = (args.output or report_path.with_name("automatic_evaluation.json")).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    print(json.dumps({"summary": evaluation["summary"], "contract": evaluation["contract"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
