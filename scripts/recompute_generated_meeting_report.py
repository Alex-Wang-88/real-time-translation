"""Repair derived summary fields in an already completed generated-meeting report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_generated_meeting import summarize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    path = args.report.resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    records = report.get("records") or []
    for item in records:
        item["translation_expected"] = item.get("translation_expected") or item.get("reference_translation")
    report["summary"] = summarize(records)
    groups = sorted({str(item.get("group")) for item in records})
    report["by_group"] = {
        group: summarize([item for item in records if str(item.get("group")) == group])
        for group in groups
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"updated {path}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
