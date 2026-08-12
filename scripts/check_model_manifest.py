"""Validate the model license manifest before building a production image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_MODEL_FIELDS = {
    "role",
    "name",
    "revision",
    "source",
    "license",
    "commercial_use",
    "license_verified",
}


def validate(path: Path, *, production: bool = False) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"无法读取模型清单：{exc}"]
    if payload.get("schema_version") != 1:
        errors.append("schema_version 必须为 1")
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        errors.append("models 必须是非空数组")
        return errors
    names: set[str] = set()
    for index, model in enumerate(models):
        prefix = f"models[{index}]"
        if not isinstance(model, dict):
            errors.append(f"{prefix} 必须是对象")
            continue
        missing = REQUIRED_MODEL_FIELDS - model.keys()
        if missing:
            errors.append(f"{prefix} 缺少字段：{', '.join(sorted(missing))}")
        name = str(model.get("name", "")).strip()
        if not name:
            errors.append(f"{prefix}.name 不能为空")
        if name in names:
            errors.append(f"重复模型：{name}")
        names.add(name)
        if production:
            if str(model.get("revision", "")).startswith("PIN_REQUIRED"):
                errors.append(f"{prefix}.revision 仍未固定：{name}")
            if model.get("commercial_use") is not True:
                errors.append(f"{prefix}.commercial_use 未明确批准：{name}")
            if model.get("license_verified") is not True:
                errors.append(f"{prefix}.license_verified 未通过：{name}")
    if production and payload.get("production_approved") is not True:
        errors.append("production_approved 必须为 true")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()
    errors = validate(args.manifest, production=args.production)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"model manifest OK: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
