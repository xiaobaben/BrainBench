"""Model-agnostic validation units used by BrainBench cases."""

from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional


JudgeFn = Callable[..., Any]


@dataclass
class ScoreContext:
    workspace_root: Path | str = field(default_factory=Path.cwd)
    semantic_judge: Optional[JudgeFn] = None
    vlm_judge: Optional[JudgeFn] = None
    artifact_root: Optional[Path | str] = None
    path_violations: list[str] = field(default_factory=list)

    def resolve_path(self, value: Any) -> Optional[Path]:
        if not isinstance(value, str) or not value.strip():
            return None
        root = Path(self.artifact_root or self.workspace_root).resolve()
        raw = value.strip().replace("\\", "/")
        if raw.startswith("/workspace/"):
            candidate = root / raw[len("/workspace/"):]
        elif Path(raw).is_absolute():
            candidate = Path(raw)
        else:
            candidate = root / raw
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError:
            self.path_violations.append(f"artifact path escapes workspace: {value}")
            return None
        return resolved


@dataclass(frozen=True)
class ScoreResult:
    raw_score: float
    weighted_score: float
    error: Optional[str] = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "raw_score": self.raw_score,
            "weighted_score": self.weighted_score,
        }
        if self.error:
            result["error"] = self.error
        if self.evidence:
            result["evidence"] = self.evidence
        return result


def _clamp01(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return max(0.0, min(1.0, result))


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _canonical(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value).strip().upper()


def _tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [_canonical(item).replace(" ", "") for item in value]
    return [item.strip().strip("'\"").upper().replace(" ", "")
            for item in re.split(r"\s*,\s*", str(value).strip("[]()")) if item.strip()]


def _judge_score(judge: JudgeFn, *args: Any) -> float:
    try:
        result = judge(*args)
    except TypeError:
        parameters = inspect.signature(judge).parameters.values()
        required = [item for item in parameters if item.default is inspect.Parameter.empty]
        if len(required) != 1:
            raise
        result = judge(args[-1])
    if isinstance(result, Mapping):
        if "score" in result:
            return _clamp01(result["score"])
        if "status" in result:
            return 1.0 if bool(result["status"]) else 0.0
        if "passed" in result:
            return 1.0 if bool(result["passed"]) else 0.0
    return _clamp01(result)


def _raw_score(predicted: Any, agent_response: Any, metric: Mapping[str, Any], context: ScoreContext) -> float:
    metric_type = str(metric.get("type", ""))
    params = metric.get("params", {})
    if not isinstance(params, Mapping):
        params = {}

    if metric_type == "numeric_check":
        pred = _number(predicted)
        gt = _number(params.get("gt_value"))
        tolerance = _number(params.get("tolerance", 0.0))
        return float(pred is not None and gt is not None and tolerance is not None and tolerance >= 0 and abs(pred - gt) <= tolerance)

    if metric_type == "categorical_check":
        return float(_canonical(predicted) == _canonical(params.get("gt_value")))

    if metric_type == "set_match_check":
        predicted_set = set(_tokens(predicted))
        expected_set = set(_tokens(params.get("gt_value")))
        top_k = params.get("top_k")
        if top_k is not None:
            predicted_set = set(_tokens(predicted)[:int(top_k)])
        if str(params.get("match_mode", "exact")).lower() == "per_hit_fraction":
            return _clamp01(len(predicted_set & expected_set) * float(params.get("score_per_hit", 1.0 / max(1, len(expected_set)))))
        return float(predicted_set == expected_set)

    if metric_type == "sequence_match_check":
        predicted_seq = _tokens(predicted)
        expected_seq = _tokens(params.get("gt_value"))
        if str(params.get("match_mode", "position")).lower() in {"position", "per_hit_fraction_order"}:
            if not expected_seq:
                return float(not predicted_seq)
            return sum(index < len(predicted_seq) and predicted_seq[index] == item for index, item in enumerate(expected_seq)) / len(expected_seq)
        return float(predicted_seq == expected_seq)

    if metric_type in {"semantic_consistency", "safety_compliance", "factual_qa_check"}:
        if context.semantic_judge is None:
            return 0.0
        return _judge_score(context.semantic_judge, agent_response, params.get("judge_prompt", ""))

    if metric_type == "file_status_check":
        path = context.resolve_path(predicted)
        if path is None or not path.is_file() or path.stat().st_size == 0:
            return 0.0
        return 1.0

    raise ValueError(f"Unsupported metric type: {metric_type}")


def score_metric(
    parsed_json: Mapping[str, Any],
    agent_response: Any,
    metric: Mapping[str, Any],
    context: Optional[ScoreContext] = None,
) -> ScoreResult:
    context = context or ScoreContext()
    target_key = metric.get("target_key")
    predicted = agent_response if target_key == "full_text" else parsed_json.get(target_key)
    try:
        raw = _raw_score(predicted, agent_response, metric, context)
        weight = float(metric.get("weight", 0.0) or 0.0)
        return ScoreResult(_clamp01(raw), _clamp01(raw) * weight)
    except Exception as exc:
        return ScoreResult(0.0, 0.0, error=f"{metric.get('type', 'unknown')} scorer failed: {exc}")
