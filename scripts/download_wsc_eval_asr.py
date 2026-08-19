"""Download selected WSC-Eval-ASR files outside the Git repository.

The audio is third-party data and is deliberately stored under
``data/external`` (which is ignored by Git).  Use ``--metadata-only`` when
only the reference text and manifest structure are needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from realtime_meeting.sichuan_eval import WSC_EVAL_REPO_ID, WSC_SUBSETS


def _patterns(
    subsets: list[str],
    metadata_only: bool,
    source_ids_by_subset: dict[str, list[str]] | None = None,
) -> list[str]:
    patterns: list[str] = []
    for subset in subsets:
        root = f"WSC-Eval-ASR/{subset}"
        patterns.extend(
            [
                f"{root}/duration.txt",
                f"{root}/key.txt",
                f"{root}/text",
                f"{root}/wav.scp",
            ]
        )
        if not metadata_only:
            source_ids = (source_ids_by_subset or {}).get(subset)
            if source_ids:
                patterns.extend(f"{root}/wav/{source_id}.wav" for source_id in source_ids)
            else:
                patterns.append(f"{root}/wav/*")
    return patterns


def _ordered_keys(path: Path, limit: int) -> list[str]:
    keys = [line.strip().split(maxsplit=1)[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return keys[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/WSC-Eval"),
        help="下载目录；默认不会进入 Git",
    )
    parser.add_argument(
        "--subset",
        choices=[*WSC_SUBSETS, "all"],
        default="Easy",
        help="要下载的评测子集；默认 Easy",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="只下载 key/text/wav.scp，不下载音频",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="每个子集只下载前 N 条音频；不指定时下载所选子集全部音频",
    )
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be > 0")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on optional audio extra
        raise SystemExit("需要 huggingface-hub，请先运行: uv sync --extra audio") from exc

    subsets = list(WSC_SUBSETS) if args.subset == "all" else [args.subset]
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_downloaded = snapshot_download(
        repo_id=WSC_EVAL_REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(output),
        allow_patterns=_patterns(subsets, True),
    )
    downloaded = metadata_downloaded
    if not args.metadata_only:
        selected_ids: dict[str, list[str]] | None = None
        if args.limit is not None:
            selected_ids = {}
            for subset in subsets:
                key_path = output / "WSC-Eval-ASR" / subset / "key.txt"
                selected_ids[subset] = _ordered_keys(key_path, args.limit)
                if len(selected_ids[subset]) < args.limit:
                    raise SystemExit(f"{key_path} 只有 {len(selected_ids[subset])} 条，无法满足 --limit {args.limit}")
        downloaded = snapshot_download(
            repo_id=WSC_EVAL_REPO_ID,
            repo_type="dataset",
            revision=args.revision,
            local_dir=str(output),
            allow_patterns=_patterns(subsets, False, selected_ids),
        )
    print(f"WSC-Eval-ASR 已准备: {downloaded}")
    print(f"子集: {', '.join(subsets)}")
    if args.metadata_only:
        print("模式: 仅元数据")
    elif args.limit is None:
        print("模式: 含所选子集全部音频")
    else:
        print(f"模式: 每个子集前 {args.limit} 条音频")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
