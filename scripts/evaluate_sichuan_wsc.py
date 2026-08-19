"""Evaluate a saved WSC-Eval-ASR real-time replay report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from realtime_meeting.evaluation import evaluate_realtime_replay
from realtime_meeting.sichuan_eval import validate_wsc_eval_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("replay_report", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    report_path = args.replay_report.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = validate_wsc_eval_manifest(manifest)
    if errors:
        raise SystemExit("manifest 校验失败:\n- " + "\n- ".join(errors))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evaluation = evaluate_realtime_replay(manifest, report)
    output_path = (args.output or report_path.with_name("automatic_evaluation.json")).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    print(json.dumps({"summary": evaluation["summary"], "contract": evaluation["contract"]}, ensure_ascii=False, indent=2))
    return 0 if evaluation["contract"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
