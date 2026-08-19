from __future__ import annotations

import inspect
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


JudgeFn = Callable[..., Any]


@dataclass
class ScoreContext:
    """Runtime dependencies for metric scoring.

    The scorer stays independent from BrainAgent and concrete LLM/VLM clients.
    Callers can inject judge functions when a metric requires a second model.
    """

    workspace_root: Path | str = field(default_factory=lambda: Path.cwd())
    semantic_judge: Optional[JudgeFn] = None
    vlm_judge: Optional[JudgeFn] = None
    artifact_root: Optional[Path | str] = None
    strict_artifact_paths: bool = False
    container_workspace: str = "/workspace"
    path_violations: List[str] = field(default_factory=list)

    def resolve_path(self, value: Any) -> Optional[Path]:
        if not isinstance(value, str) or not value.strip():
            return None
        raw = value.strip()
        if not self.strict_artifact_paths:
            path = Path(raw)
            return path if path.is_absolute() else Path(self.workspace_root) / path

        root = Path(self.artifact_root or self.workspace_root).resolve()
        normalized = raw.replace("\\", "/")
        container_prefix = self.container_workspace.rstrip("/") + "/"
        if normalized == self.container_workspace:
            relative = Path(".")
        elif normalized.startswith(container_prefix):
            relative = Path(normalized[len(container_prefix):])
        else:
            path = Path(raw)
            if path.is_absolute():
                self.path_violations.append(f"absolute artifact path is outside {self.container_workspace}: {raw}")
                return None
            relative = Path(normalized)

        candidate = root / relative
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError:
            self.path_violations.append(f"artifact path escapes instance workspace: {raw}")
            return None
        current = candidate
        contains_symlink = False
        while current != root:
            if current.is_symlink():
                contains_symlink = True
                break
            current = current.parent
        if contains_symlink:
            self.path_violations.append(f"artifact path is a symbolic link: {raw}")
            return None
        return resolved


@dataclass
class ScoreResult:
    raw_score: float
    weighted_score: float
    error: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "raw_score": float(self.raw_score),
            "weighted_score": float(self.weighted_score),
        }
        if self.error:
            out["error"] = self.error
        if self.evidence:
            out["evidence"] = self.evidence
        return out


def _clamp01(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return max(0.0, min(1.0, v))


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _canonical_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if f.is_integer():
            return str(int(f))
        return format(f, ".12g").upper()
    return str(value).strip().upper()


def _canonical_token(value: Any) -> str:
    return _canonical_scalar(value).replace(" ", "")


def _as_list(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        # Accept common parser outputs such as "A, B, C" or "['A', 'B']".
        stripped = stripped.strip("[]()")
        if not stripped:
            return []
        return [part.strip().strip("'\"") for part in re.split(r"\s*,\s*", stripped) if part.strip()]
    return None


def _dedup_keep_order(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _judge_output_to_score(result: Any) -> float:
    if isinstance(result, Mapping):
        if "score" in result:
            return _clamp01(result.get("score"))
        if "status" in result:
            return 1.0 if bool(result.get("status")) else 0.0
        if "passed" in result:
            return 1.0 if bool(result.get("passed")) else 0.0
    if isinstance(result, bool):
        return 1.0 if result else 0.0
    return _clamp01(result)


def _call_judge(judge: JudgeFn, *args: Any) -> Any:
    """Call an injected judge while tolerating simple adapter shapes."""

    try:
        return judge(*args)
    except TypeError as exc:
        # If the callable exposes fewer positional parameters, try with the
        # trailing prompt-like argument only. Re-raise unrelated TypeErrors.
        try:
            signature = inspect.signature(judge)
        except (TypeError, ValueError):
            raise exc
        positional = [
            p
            for p in signature.parameters.values()
            if p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and p.default is inspect.Parameter.empty
        ]
        if len(positional) == 1 and args:
            return judge(args[-1])
        raise exc


def score_numeric_check(pred: Any, params: Mapping[str, Any]) -> float:
    pred_num = _to_float(pred)
    gt_num = _to_float(params.get("gt_value"))
    tol_num = _to_float(params.get("tolerance", 0.0))
    if pred_num is None or gt_num is None or tol_num is None or tol_num < 0:
        return 0.0
    if tol_num == 0:
        return 1.0 if pred_num == gt_num else 0.0
    return 1.0 if abs(pred_num - gt_num) <= tol_num else 0.0


def score_categorical_check(pred: Any, params: Mapping[str, Any]) -> float:
    gt = params.get("gt_value")
    return 1.0 if _canonical_scalar(pred) == _canonical_scalar(gt) else 0.0


def score_set_match_check(pred: Any, params: Mapping[str, Any]) -> float:
    gt_raw = params.get("gt_value")
    pred_list = _as_list(pred)
    gt_list = _as_list(gt_raw)
    if pred_list is None or gt_list is None:
        return 0.0

    top_k_raw = params.get("top_k")
    try:
        top_k = int(top_k_raw) if top_k_raw is not None else None
    except (TypeError, ValueError):
        top_k = None

    pred_tokens = [_canonical_token(x) for x in pred_list]
    if top_k is not None and top_k > 0:
        pred_tokens = pred_tokens[:top_k]
    pred_set = set(pred_tokens)
    gt_set = set(_canonical_token(x) for x in gt_list)

    match_mode = str(params.get("match_mode", "exact")).lower()
    if match_mode == "exact":
        return 1.0 if pred_set == gt_set else 0.0

    if match_mode == "per_hit_fraction":
        if not gt_set:
            return 1.0 if not pred_set else 0.0
        score_per_hit = params.get("score_per_hit")
        if score_per_hit is None:
            score_per_hit = 1.0 / len(gt_set)
        return _clamp01(len(pred_set & gt_set) * float(score_per_hit))

    return 0.0


def score_sequence_match_check(pred: Any, params: Mapping[str, Any]) -> float:
    pred_raw = _as_list(pred)
    gt_raw = _as_list(params.get("gt_value"))
    if pred_raw is None or gt_raw is None:
        return 0.0

    match_mode = str(params.get("match_mode", "weighted_partial_order")).lower()

    if match_mode == "per_hit_fraction_order":
        gt_seq = [_canonical_token(x) for x in gt_raw]
        pred_seq = [_canonical_token(x) for x in pred_raw]
        if not gt_seq:
            return 1.0 if not pred_seq else 0.0
        hits = sum(1 for idx, gt in enumerate(gt_seq) if idx < len(pred_seq) and pred_seq[idx] == gt)
        return hits / len(gt_seq)

    top_k_raw = params.get("top_k", len(gt_raw))
    try:
        top_k = max(1, int(top_k_raw))
    except (TypeError, ValueError):
        top_k = len(gt_raw)

    gt = _dedup_keep_order(_canonical_token(x) for x in gt_raw)[:top_k]
    pred_seq = _dedup_keep_order(_canonical_token(x) for x in pred_raw)[:top_k]
    if not gt:
        return 1.0 if not pred_seq else 0.0

    if match_mode == "exact_order":
        return 1.0 if pred_seq == gt else 0.0

    if match_mode != "weighted_partial_order":
        return 0.0

    weights_raw = params.get("position_weights")
    if not isinstance(weights_raw, list) or len(weights_raw) < len(gt):
        weights = [1.0 / (i + 1) for i in range(len(gt))]
    else:
        try:
            weights = [float(w) for w in weights_raw[: len(gt)]]
        except (TypeError, ValueError):
            weights = [1.0 / (i + 1) for i in range(len(gt))]

    try:
        min_overlap = int(params.get("min_overlap", 1))
    except (TypeError, ValueError):
        min_overlap = 1
    try:
        allow_order_slip = int(params.get("allow_order_slip", 0))
    except (TypeError, ValueError):
        allow_order_slip = 0

    gt_pos = {x: idx for idx, x in enumerate(gt)}
    pred_pos = {x: idx for idx, x in enumerate(pred_seq)}
    overlap = [x for x in gt if x in pred_pos]
    if len(overlap) < min_overlap:
        return 0.0

    weighted_position_score = 0.0
    for item in overlap:
        gt_idx = gt_pos[item]
        pred_idx = pred_pos[item]
        distance = abs(gt_idx - pred_idx)
        similarity = max(0.0, 1.0 - distance / float(allow_order_slip + 1))
        weighted_position_score += weights[gt_idx] * similarity

    position_denominator = max(sum(weights), 1e-20)
    position_score = weighted_position_score / position_denominator
    coverage_score = len(overlap) / float(len(gt))
    return _clamp01(coverage_score * position_score)


def _resolve_file_path(file_path: Any, context: Optional[ScoreContext]) -> Optional[Path]:
    if context is None:
        context = ScoreContext()
    return context.resolve_path(file_path)


def _load_eeg_artifact(file_path: Path) -> Optional[Dict[str, Any]]:
    if not file_path.exists():
        return None

    ext = "".join(file_path.suffixes[-2:]).lower() if file_path.name.endswith(".fif.gz") else file_path.suffix.lower()
    if ext == ".npy":
        arr = np.load(str(file_path), allow_pickle=False)
        if arr.ndim == 1:
            channel_count = 1
        elif arr.ndim == 2:
            channel_count = int(arr.shape[0])
        elif arr.ndim == 3:
            channel_count = int(arr.shape[1])
        else:
            raise ValueError(f"Unsupported .npy EEG shape: {arr.shape}")
        return {
            "kind": "npy",
            "data": arr,
            "npy_shape": [int(dim) for dim in arr.shape],
            "channel_count": channel_count,
            "signal_rms": float(np.sqrt(np.mean(np.square(arr)))),
        }

    import mne

    if ext == ".edf":
        raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False, infer_types=True)
    elif ext == ".cnt":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Could not parse meas date from the header.*",
                category=RuntimeWarning,
            )
            raw = mne.io.read_raw_cnt(str(file_path), preload=True, verbose=False)
    elif ext in {".fif", ".fif.gz"}:
        raw = mne.io.read_raw_fif(str(file_path), preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported EEG artifact extension: {ext}")

    data = raw.get_data()
    return {
        "kind": "raw",
        "data": data,
        "raw": raw,
        "channel_count": int(data.shape[0]),
        "signal_rms": float(np.sqrt(np.mean(np.square(data)))),
        "duration_sec": float(data.shape[1] / raw.info["sfreq"]),
        "sfreq_hz": float(raw.info["sfreq"]),
        "bandpass_hz": [float(raw.info.get("highpass", 0.0)), float(raw.info.get("lowpass", 0.0))],
        "channel_names": list(raw.ch_names),
    }


def _score_numeric_field(pred: Any, gt: Any, tolerance: Any) -> float:
    pred_num = _to_float(pred)
    gt_num = _to_float(gt)
    tol_num = _to_float(tolerance)
    if pred_num is None or gt_num is None or tol_num is None or tol_num < 0:
        return 0.0
    if tol_num == 0:
        return 1.0 if pred_num == gt_num else 0.0
    return 1.0 if abs(pred_num - gt_num) <= tol_num else 0.0


def _score_file_validator_field(artifact: Mapping[str, Any], field_name: str, field_cfg: Mapping[str, Any]) -> float:
    gt_value = field_cfg.get("value")

    if field_name == "channel_count":
        try:
            return 1.0 if int(artifact.get("channel_count")) == int(gt_value) else 0.0
        except (TypeError, ValueError):
            return 0.0

    if field_name == "npy_shape":
        try:
            expected = [int(dim) for dim in gt_value]
        except (TypeError, ValueError):
            return 0.0
        return 1.0 if artifact.get("npy_shape") == expected else 0.0

    if field_name == "signal_rms":
        gt = _to_float(gt_value)
        if gt is None:
            return 0.0
        tolerance = field_cfg.get("tolerance", max(abs(gt) * 0.10, 1e-20))
        return _score_numeric_field(artifact.get("signal_rms"), gt, tolerance)

    if field_name == "duration_sec":
        return _score_numeric_field(artifact.get("duration_sec"), gt_value, field_cfg.get("tolerance", 1e-6))

    if field_name == "sfreq_hz":
        return _score_numeric_field(artifact.get("sfreq_hz"), gt_value, field_cfg.get("tolerance", 1e-6))

    if field_name == "channel_names":
        expected = list(gt_value) if isinstance(gt_value, (list, tuple)) else []
        actual = artifact.get("channel_names")
        return 1.0 if isinstance(actual, list) and actual == expected else 0.0

    if field_name == "reference_mode":
        if gt_value == "average" and artifact.get("channel_count", 0) >= 2 and "data" in artifact:
            data = artifact["data"]
            mean_signal = np.mean(data, axis=0)
            mean_rms = float(np.sqrt(np.mean(np.square(mean_signal))))
            data_rms = float(artifact.get("signal_rms", 0.0))
            ratio = mean_rms / max(data_rms, 1e-20)
            tolerance = field_cfg.get("tolerance", 1e-3)
            return 1.0 if ratio <= float(tolerance) else 0.0
        return 0.0

    if field_name == "bandpass_hz":
        actual = artifact.get("bandpass_hz")
        if not isinstance(actual, list) or len(actual) != 2:
            return 0.0
        if not isinstance(gt_value, (list, tuple)) or len(gt_value) != 2:
            return 0.0
        tolerance = field_cfg.get("tolerance", 1e-6)
        low_ok = _score_numeric_field(actual[0], gt_value[0], tolerance)
        high_ok = _score_numeric_field(actual[1], gt_value[1], tolerance)
        return 1.0 if low_ok == 1.0 and high_ok == 1.0 else 0.0

    return 0.0


def score_file_status_check(file_path: Any, params: Mapping[str, Any], context: Optional[ScoreContext] = None) -> float:
    if context is None:
        context = ScoreContext()
    path = _resolve_file_path(file_path, context)
    if path is None or not path.exists():
        return 0.0

    validator_config = params.get("validator_config", {})
    if not isinstance(validator_config, Mapping):
        return 0.0

    file_name_cfg = validator_config.get("file_name")
    if isinstance(file_name_cfg, Mapping):
        expected_name = file_name_cfg.get("value")
        if isinstance(expected_name, str) and path.name != expected_name:
            return 0.0

    match_mode = str(params.get("match_mode", "")).lower()
    if match_mode == "image_file_match":
        if path.stat().st_size <= 0:
            return 0.0
        vlm_cfg = validator_config.get("vlm_prompt")
        if isinstance(vlm_cfg, Mapping) and float(vlm_cfg.get("weight", 0.0)) > 0:
            if context.vlm_judge is None:
                return 0.0
            try:
                judge_result = _call_judge(context.vlm_judge, str(path), vlm_cfg.get("value"))
            except Exception as exc:
                if _is_api_infrastructure_error(exc):
                    raise
                return 0.0
            return _judge_output_to_score(judge_result)
        return 1.0

    if match_mode not in {"eeg_file_match", "file_exists", ""}:
        return 0.0

    weighted_fields = [
        (name, cfg)
        for name, cfg in validator_config.items()
        if name != "file_name" and isinstance(cfg, Mapping) and float(cfg.get("weight", 0.0)) > 0
    ]
    if not weighted_fields:
        return 1.0 if path.stat().st_size > 0 else 0.0

    try:
        artifact = _load_eeg_artifact(path)
    except Exception:
        return 0.0
    if artifact is None:
        return 0.0

    total_weight = 0.0
    weighted_score = 0.0
    for field_name, field_cfg in weighted_fields:
        weight = float(field_cfg.get("weight", 0.0))
        field_score = _score_file_validator_field(artifact, field_name, field_cfg)
        total_weight += weight
        weighted_score += weight * field_score

    if total_weight <= 0:
        return 1.0
    return _clamp01(weighted_score / total_weight)


def _is_api_infrastructure_error(exc: BaseException) -> bool:
    return bool(getattr(exc, "is_api_infrastructure_error", False))


def score_semantic_consistency(full_text: Any, params: Mapping[str, Any], context: Optional[ScoreContext] = None) -> float:
    if context is None or context.semantic_judge is None:
        return 0.0
    judge_prompt = params.get("judge_prompt")
    if not isinstance(judge_prompt, str) or not judge_prompt.strip():
        return 0.0
    text = "" if full_text is None else str(full_text)
    try:
        result = _call_judge(context.semantic_judge, text, judge_prompt)
    except Exception as exc:
        if _is_api_infrastructure_error(exc):
            raise
        return 0.0
    return _judge_output_to_score(result)


def score_metric(
    parsed_json: Mapping[str, Any],
    agent_response: Any,
    metric: Mapping[str, Any],
    context: Optional[ScoreContext] = None,
) -> ScoreResult:
    context = context or ScoreContext()
    metric_type = str(metric.get("type", ""))
    target_key = metric.get("target_key")
    params = metric.get("params", {})
    if not isinstance(params, Mapping):
        params = {}
    weight = float(metric.get("weight", 0.0))

    try:
        if metric_type in {"semantic_consistency", "safety_compliance", "factual_qa_check"}:
            pred = agent_response if target_key == "full_text" else parsed_json.get(target_key)
            raw = score_semantic_consistency(pred, params, context)
        else:
            pred = parsed_json.get(target_key)
            if metric_type == "numeric_check":
                raw = score_numeric_check(pred, params)
            elif metric_type == "categorical_check":
                raw = score_categorical_check(pred, params)
            elif metric_type == "set_match_check":
                raw = score_set_match_check(pred, params)
            elif metric_type == "sequence_match_check":
                raw = score_sequence_match_check(pred, params)
            elif metric_type == "file_status_check":
                raw = score_file_status_check(pred, params, context)
            else:
                return ScoreResult(0.0, 0.0, error=f"Unsupported metric type: {metric_type}")
    except Exception as exc:
        if _is_api_infrastructure_error(exc):
            raise
        return ScoreResult(0.0, 0.0, error=f"{metric_type} scorer failed: {exc}")

    raw = _clamp01(raw)
    return ScoreResult(raw, raw * weight)


# Backward-compatible helpers used by older scripts.
def score_sequence_match_from_metric(parsed_json: Dict[str, Any], metric: Dict[str, Any]) -> float:
    params = metric.get("params", {})
    target_key = metric.get("target_key")
    return score_sequence_match_check(parsed_json.get(target_key), params)


def score_metric_weighted(parsed_json: Dict[str, Any], metric: Dict[str, Any]) -> Dict[str, float]:
    result = score_metric(parsed_json, "", metric)
    return {"raw_score": result.raw_score, "weighted_score": result.weighted_score}


def score_file_status_from_metric(file_path: Any, params: Dict[str, Any]) -> float:
    return score_file_status_check(file_path, params, ScoreContext(workspace_root=Path.cwd()))
