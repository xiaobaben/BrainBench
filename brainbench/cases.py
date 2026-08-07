"""Case loading and explicit benchmark-file discovery."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


_CASE_FILE = re.compile(r"case\d+_\d+\.json$")
_CASE_DIR = re.compile(r"case\d+$")


def load_case_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def is_case_json(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(isinstance(value.get(section), Mapping) for section in (
        "meta_info",
        "agent_input",
        "eval_config",
    ))


def validate_case_json(value: Mapping[str, Any]) -> None:
    if not is_case_json(value):
        raise ValueError("Case JSON must contain meta_info, agent_input, and eval_config objects")
    eval_config = value["eval_config"]
    metrics = eval_config.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("Case JSON eval_config.metrics must be a non-empty list")
    if "instruction" not in value["agent_input"]:
        raise ValueError("Case JSON agent_input.instruction is required")


def iter_case_paths(subset_root: str | Path) -> Iterable[Path]:
    """Yield only canonical case files, excluding backups and validation JSON."""

    root = Path(subset_root)
    for path in sorted(root.glob("case*/*.json")):
        if _CASE_DIR.fullmatch(path.parent.name) and _CASE_FILE.fullmatch(path.name):
            value = load_case_json(path)
            if is_case_json(value):
                yield path
