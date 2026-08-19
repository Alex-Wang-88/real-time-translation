"""Create the project's dual-text manifest from WSC-Eval-ASR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from realtime_meeting.sichuan_eval import WSC_SUBSETS, WscEvalError, validate_wsc_eval_manifest, write_wsc_eval_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="WSC-Eval-ASR 目录，例如 data/external/WSC-Eval/WSC-Eval-ASR",
    )
    parser.add_argument("--subset", choices=WSC_SUBSETS, default="Easy")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 manifest；默认写入 data/evaluation/sichuan_wsc_<subset>.json",
    )
    parser.add_argument("--limit", type=int, default=None, help="只取前 N 条，便于先做小规模 smoke test")
    parser.add_argument("--require-audio", action="store_true", help="要求每条音频已下载")
    args = parser.parse_args()

    output = args.output or Path("data/evaluation") / f"sichuan_wsc_{args.subset.casefold()}.json"
    try:
        manifest = write_wsc_eval_manifest(args.dataset_root, args.subset, output, limit=args.limit)
    except WscEvalError as exc:
        raise SystemExit(str(exc)) from exc

    errors = validate_wsc_eval_manifest(manifest)
    if args.require_audio:
        root = output.expanduser().resolve().parent
        for sample in manifest["samples"]:
            if not (root / str(sample["audio_path"])).is_file():
                errors.append(f"音频不存在: {sample['audio_path']}")
    if errors:
        output.unlink(missing_ok=True)
        raise SystemExit("manifest 校验失败:\n- " + "\n- ".join(errors))

    print(f"wrote {output.expanduser().resolve()}")
    print(
        json.dumps(
            {
                "dataset": manifest["dataset"],
                "sample_count": manifest["sample_count"],
                "recording_seconds": manifest["recording_seconds"],
                "surface_text_field": manifest["reference_schema"]["surface_text_field"],
                "mandarin_text_field": manifest["reference_schema"]["mandarin_text_field"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
