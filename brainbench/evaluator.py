"""Minimal evaluator independent of BrainAgent."""

from __future__ import annotations

import json
import logging
import hashlib
import time
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from .agent import AgentRunResult, AgentRunner, build_agent_query
from .cases import is_case_json, iter_case_paths, load_case_json, validate_case_json
from .scoring import ScoreContext, score_metric


ParserAgent = Callable[[str, str], Dict[str, Any]]
logger = logging.getLogger("BrainBench")


class NeuroBenchEvaluator:
    """Run target Agents through Parser and validation metrics."""

    def __init__(
        self,
        agent_runner: AgentRunner | Callable[..., Any],
        parser_agent: ParserAgent,
        *,
        workspace_root: str | Path = ".",
        semantic_judge: Optional[Callable[..., Any]] = None,
        vlm_judge: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.agent_runner = agent_runner
        self.parser_agent = parser_agent
        self.workspace_root = Path(workspace_root).resolve()
        self.context = ScoreContext(
            workspace_root=self.workspace_root,
            artifact_root=self.workspace_root,
            semantic_judge=semantic_judge,
            vlm_judge=vlm_judge,
        )
        self._roles = {
            "target": agent_runner,
            "parser": parser_agent,
            "semantic": semantic_judge,
            "vlm": vlm_judge,
        }

    def run_case(
        self,
        case_json: Mapping[str, Any],
        *,
        source_path: Optional[str | Path] = None,
        instance_id: Optional[str] = None,
    ) -> dict[str, Any]:
        validate_case_json(case_json)
        meta = dict(case_json["meta_info"])
        agent_input = dict(case_json["agent_input"])
        query = build_agent_query(agent_input)
        eval_config = dict(case_json["eval_config"])
        source = Path(source_path) if source_path is not None else None
        resolved_id = instance_id or self._instance_id(meta, source)
        recording_id = self._recording_id(meta.get("case_id"), agent_input)
        started = datetime.now(timezone.utc).isoformat()
        total_start = time.perf_counter()
        token_before = self._token_snapshot()
        timing = {
            "target": 0.0,
            "parser": 0.0,
            "judge": 0.0,
            "vlm": 0.0,
            "model_total": 0.0,
            "scoring": 0.0,
            "total": 0.0,
        }
        target_start = time.perf_counter()
        errors: list[str] = []
        target_tokens = 0
        agent_output = ""
        sandbox_audit: Optional[Mapping[str, Any]] = None
        target_ok = False
        try:
            if isinstance(self.agent_runner, AgentRunner):
                run_context: dict[str, Any] = {"instance_id": resolved_id}
                if getattr(self.agent_runner, "evaluation_mode", "agent") == "codeact":
                    run_context["agent_input"] = agent_input
                result = self.agent_runner.run_with_usage(
                    query,
                    run_context,
                )
            else:
                result = self.agent_runner(query)
            if isinstance(result, AgentRunResult):
                agent_output = result.response
                target_tokens = int(result.tokens)
                sandbox_audit = result.audit
            else:
                agent_output = str(result)
            target_ok = True
        except Exception as exc:
            errors.append(f"agent_error: {exc}")
            sandbox_audit = getattr(self.agent_runner, "last_audit", None)
        timing["target"] = time.perf_counter() - target_start

        parser_output: dict[str, Any] = {}
        if target_ok:
            parser_start = time.perf_counter()
            try:
                parser_result = self.parser_agent(agent_output, str(eval_config.get("parser_prompt", "")))
                if not isinstance(parser_result, dict):
                    raise TypeError("parser must return a dict")
                parser_output = parser_result
            except Exception as exc:
                errors.append(f"parser_error: {exc}")
            timing["parser"] = time.perf_counter() - parser_start

        metrics = [item for item in eval_config.get("metrics", []) if isinstance(item, Mapping)]
        max_score = sum(float(metric.get("weight", 0.0) or 0.0) for metric in metrics)
        total_score = 0.0
        details: dict[str, Any] = {}
        score_start = time.perf_counter()
        case_context = ScoreContext(
            workspace_root=self.context.workspace_root,
            artifact_root=self.context.artifact_root,
            semantic_judge=self._timed_judge(self.context.semantic_judge, timing, "judge"),
            vlm_judge=self._timed_judge(self.context.vlm_judge, timing, "vlm"),
        )
        if target_ok:
            for metric in metrics:
                metric_id = str(metric.get("metric_id", "unknown_metric"))
                result = score_metric(parser_output, agent_output, metric, case_context)
                details[metric_id] = {
                    **result.as_dict(),
                    "metric_type": metric.get("type"),
                    "target_key": metric.get("target_key"),
                    "weight": metric.get("weight", 0.0),
                }
                total_score += result.weighted_score
                if result.error:
                    errors.append(f"{metric_id}: {result.error}")
        timing["scoring"] = time.perf_counter() - score_start
        timing["total"] = time.perf_counter() - total_start
        timing["model_total"] = sum(
            timing[role] for role in ("target", "parser", "judge", "vlm")
        )

        token_after = self._token_snapshot()
        token_usage = {
            role: max(0, token_after[role] - token_before[role])
            for role in ("target", "parser", "semantic", "vlm")
        }
        if target_tokens:
            token_usage["target"] = target_tokens
        token_usage["total"] = sum(token_usage.values())
        artifact_manifest = self._artifact_manifest(parser_output, eval_config, case_context)
        errors.extend(
            f"artifact_path_violation: {message}"
            for message in dict.fromkeys(case_context.path_violations)
        )

        return {
            "instance_id": resolved_id,
            "recording_id": recording_id,
            "case_id": meta.get("case_id"),
            "bench_subset": meta.get("bench_subset"),
            "difficulty": meta.get("difficult", meta.get("difficulty")),
            "source_json": str(source) if source else None,
            "total_score": float(total_score),
            "max_score": float(max_score),
            "vrr": 1.0 if target_ok else 0.0,
            "metric_details": details,
            "agent_output": agent_output,
            "parser_output": parser_output,
            "artifact_manifest": artifact_manifest,
            "sandbox": dict(sandbox_audit) if sandbox_audit is not None else None,
            "token_usage": token_usage,
            "timing_sec": {key: round(value, 6) for key, value in timing.items()},
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "errors": errors,
        }

    def run_subset(
        self,
        subset_root: str | Path,
        *,
        output_path: Optional[str | Path] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        paths = iter_case_paths(subset_root)
        if limit is not None:
            paths = islice(paths, limit)
        paths = list(paths)
        if not paths:
            raise ValueError(f"No valid case JSON files found under {subset_root}")
        case_records = [(path, load_case_json(path)) for path in paths]
        instances: list[dict[str, Any]] = []
        target = Path(output_path) if output_path is not None else None
        started_at = datetime.now(timezone.utc)
        wall_start = time.perf_counter()
        models = self.model_configs()
        evaluation_mode = str(getattr(self.agent_runner, "evaluation_mode", "agent"))
        subset_name = Path(subset_root).name
        if case_records:
            subset_name = str(
                case_records[0][1].get("meta_info", {}).get("bench_subset")
                or subset_name
            )

        def build_payload(finished_at: datetime) -> tuple[dict[str, Any], dict[str, Any]]:
            summary = self._summarize_instances(
                instances,
                time.perf_counter() - wall_start,
            )
            summary["result_file"] = str(target) if target is not None else None
            summary["planned_instance_count"] = len(case_records)
            summary["completed_instance_count"] = len(instances)
            experiment = {
                "subset": subset_name,
                "subset_root": str(Path(subset_root)),
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "models": models,
                "max_workers": 1,
                "evaluation_mode": evaluation_mode,
                "rerun": False,
            }
            if getattr(self.agent_runner, "execution_mode", None) == "docker":
                experiment["sandbox"] = self._sandbox_summary(instances)
            return {"experiment": experiment, "instances": instances, "summary": summary}, summary

        def write_checkpoint() -> dict[str, Any]:
            payload, summary = build_payload(datetime.now(timezone.utc))
            if target is None:
                return summary
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            json.loads(temporary.read_text(encoding="utf-8"))
            temporary.replace(target)
            return summary

        write_checkpoint()
        for completed, (path, case_json) in enumerate(case_records, 1):
            result = self.run_case(case_json, source_path=path)
            instances.append(result)
            logger.info(
                "Completed %s (%d/%d): score=%.4f, time=%.3fs, tokens=%d",
                result["instance_id"],
                completed,
                len(paths),
                result["total_score"],
                result["timing_sec"]["total"],
                result["token_usage"]["total"],
            )
            write_checkpoint()

        return write_checkpoint()

    def model_configs(self) -> dict[str, dict[str, Any]]:
        configs: dict[str, dict[str, Any]] = {}
        for role, runner in self._roles.items():
            config = getattr(runner, "model_config", None) if runner is not None else None
            configs[role] = dict(
                config
                or {"role": role, "model": None, "endpoint_mode": "unknown"}
            )
        return configs

    def _token_snapshot(self) -> dict[str, int]:
        return {
            role: int(getattr(runner, "total_tokens", 0) or 0)
            if runner is not None
            else 0
            for role, runner in self._roles.items()
        }

    @staticmethod
    def _timed_judge(
        judge: Optional[Callable[..., Any]],
        timing: dict[str, float],
        role: str,
    ) -> Optional[Callable[..., Any]]:
        if judge is None:
            return None

        def call(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return judge(*args, **kwargs)
            finally:
                timing[role] += time.perf_counter() - started

        return call

    @staticmethod
    def _recording_id(case_id: Any, agent_input: Mapping[str, Any]) -> str:
        candidate = (
            case_id
            or agent_input.get("data_path")
            or agent_input.get("label_path")
            or "unknown"
        )
        return Path(str(candidate)).stem

    @staticmethod
    def _artifact_manifest(
        parser_output: Mapping[str, Any],
        eval_config: Mapping[str, Any],
        context: ScoreContext,
    ) -> list[dict[str, Any]]:
        manifest: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for metric in eval_config.get("metrics", []):
            if not isinstance(metric, Mapping) or metric.get("type") != "file_status_check":
                continue
            metric_id = str(metric.get("metric_id", "unknown_metric"))
            target_key = metric.get("target_key")
            reported_path = (
                parser_output.get(target_key) if isinstance(target_key, str) else None
            )
            if not isinstance(reported_path, str):
                continue
            path = context.resolve_path(reported_path)
            if path is None or not path.is_file() or path.is_symlink():
                continue
            identity = (metric_id, str(path))
            if identity in seen:
                continue
            seen.add(identity)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            manifest.append(
                {
                    "metric_id": metric_id,
                    "reported_path": reported_path,
                    "file_name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                    "score_completed": True,
                    "retained": False,
                }
            )
        return manifest

    def _sandbox_summary(
        self, instances: list[Mapping[str, Any]]
    ) -> dict[str, Any]:
        runner = self.agent_runner
        return {
            "backend": "docker",
            "image": getattr(runner, "docker_image", None),
            "memory_limit": getattr(runner, "docker_memory_limit", None),
            "cpu_count": getattr(runner, "docker_cpu_count", None),
            "execution_timeout_sec": getattr(runner, "execution_timeout_sec", None),
            "case_timeout_sec": getattr(runner, "case_timeout_sec", None),
            "allowed_input_roots": [str((self.workspace_root / "data").resolve())],
            "image_digests": sorted(
                {
                    str(item.get("sandbox", {}).get("image_digest"))
                    for item in instances
                    if isinstance(item.get("sandbox"), Mapping)
                    and item.get("sandbox", {}).get("image_digest")
                }
            ),
        }

    @staticmethod
    def _summarize_instances(
        instances: list[Mapping[str, Any]], wall_time_sec: float
    ) -> dict[str, Any]:
        token_usage = {
            role: sum(
                int(item.get("token_usage", {}).get(role, 0)) for item in instances
            )
            for role in ("target", "parser", "semantic", "vlm")
        }
        token_usage["total"] = sum(token_usage.values())

        failure_summary: dict[str, int] = {}
        for item in instances:
            for error in item.get("errors", []):
                category = str(error).split(":", 1)[0]
                failure_summary[category] = failure_summary.get(category, 0) + 1

        metric_groups: dict[str, dict[str, Any]] = {}
        for item in instances:
            for detail in item.get("metric_details", {}).values():
                metric_type = str(detail.get("metric_type", "unknown"))
                group = metric_groups.setdefault(
                    metric_type,
                    {"count": 0, "raw_score_sum": 0.0, "weighted_score_sum": 0.0},
                )
                group["count"] += 1
                group["raw_score_sum"] += float(detail.get("raw_score", 0.0) or 0.0)
                group["weighted_score_sum"] += float(
                    detail.get("weighted_score", 0.0) or 0.0
                )
        for group in metric_groups.values():
            count = group["count"] or 1
            group["raw_score_mean"] = group["raw_score_sum"] / count
            group["weighted_score_mean"] = group["weighted_score_sum"] / count

        weighted_score_sum = 0.0
        weighted_max_score_sum = 0.0
        for item in instances:
            difficulty = float(item.get("difficulty", 1.0) or 1.0)
            weighted_score_sum += float(item.get("total_score", 0.0) or 0.0) * difficulty
            weighted_max_score_sum += float(item.get("max_score", 100.0) or 100.0) * difficulty
        total_score = (
            100.0 * weighted_score_sum / weighted_max_score_sum
            if weighted_max_score_sum > 0
            else 0.0
        )

        timing_summary = {
            role: sum(
                float(item.get("timing_sec", {}).get(role, 0.0))
                for item in instances
            )
            for role in (
                "target",
                "parser",
                "judge",
                "vlm",
                "model_total",
                "scoring",
                "total",
            )
        }
        instance_count = len(instances)
        instance_time = timing_summary["total"]
        return {
            "instance_count": instance_count,
            "successful_instance_count": sum(
                1 for item in instances if float(item.get("vrr", 0.0)) > 0
            ),
            "total_score": total_score,
            "mean_score": total_score,
            "difficulty_weighted_score_sum": weighted_score_sum,
            "difficulty_weighted_max_score_sum": weighted_max_score_sum,
            "vrr_mean": (
                sum(float(item.get("vrr", 0.0)) for item in instances)
                / instance_count
                if instance_count
                else 0.0
            ),
            "failure_summary": failure_summary,
            "token_usage": token_usage,
            "timing_sec": {
                "target": round(timing_summary["target"], 6),
                "parser": round(timing_summary["parser"], 6),
                "judge": round(timing_summary["judge"], 6),
                "vlm": round(timing_summary["vlm"], 6),
                "model_total": round(timing_summary["model_total"], 6),
                "scoring": round(timing_summary["scoring"], 6),
                "total": round(instance_time, 6),
                "wall_clock_total": round(float(wall_time_sec), 6),
                "mean_per_instance": (
                    round(instance_time / instance_count, 6)
                    if instance_count
                    else 0.0
                ),
            },
            "metric_type_summary": metric_groups,
        }

    @staticmethod
    def _instance_id(meta: Mapping[str, Any], source: Optional[Path]) -> str:
        subset = str(meta.get("bench_subset", "Unknown")).removeprefix("NeuroBench-")
        if source is not None:
            return f"{subset}-{source.parent.name}-{source.stem}"
        return f"{subset}-{Path(str(meta.get('case_id', 'unknown'))).stem}"
