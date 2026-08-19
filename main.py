"""BrainBench's single user entry point.

IDE users edit the small configuration block below and run this file directly.
Command-line users pass the equivalent ``smoke``, ``prepare``, or ``run``
command. API credentials are loaded from ``.env`` or system environment
variables and must never be written into this file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from dotenv import load_dotenv

from brainbench import AgentRunner, AgentRunResult, EndpointConfig, NeuroBenchEvaluator
from brainbench.cases import load_case_json
from brainbench.llm import make_json_parser, make_semantic_judge, make_vlm_judge
from brainbench.runners import CodeActAgentRunner

PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
load_dotenv(PROJECT_ROOT / ".env")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_custom_agent(query: str) -> AgentRunResult:
    """Send the complete query to the user's Agent and return its result."""

    class TargetAgent:
        @staticmethod
        def run(context):
            return context

    your_agent = TargetAgent()
    response = your_agent.run(query)

    return AgentRunResult(response=response, tokens=0)


def _subsets() -> dict[str, dict[str, Any]]:
    registry_path = BENCHMARK_ROOT / "subsets.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for entry in registry.get("subsets", []):
        manifest_path = BENCHMARK_ROOT / str(entry["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["manifest_path"] = manifest_path
        result[str(entry["id"])] = manifest
    return result


def _released_subsets() -> tuple[str, ...]:
    return tuple(
        subset_id
        for subset_id, manifest in _subsets().items()
        if manifest.get("status") == "released"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and run BrainBench.",
        epilog=(
            "Run without arguments to use the IDE configuration at the top of main.py. "
            "Model endpoint settings come from .env or system environment variables."
        ),
    )
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("smoke", help="Run the offline synthetic smoke test.")

    prepare = commands.add_parser("prepare", help="Prepare one subset from downloaded datasets.")
    prepare.add_argument("subset", choices=_released_subsets())
    prepare.add_argument("--data-root", required=True, type=Path)

    run = commands.add_parser("run", help="Evaluate one released subset or one case.")
    run.add_argument("subset", nargs="?", choices=_released_subsets())
    run.add_argument(
        "--case-path",
        type=Path,
        default=None,
        help="Run only the case JSON at this path; the subset is read from the case.",
    )
    run.add_argument("--agent", choices=("codeact", "custom"), default="codeact")
    run.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Path for the result JSON. Relative paths are resolved from the "
            "BrainBench project root. Defaults to runs/<subset>.json."
        ),
    )
    return parser


def _ide_arguments() -> list[str]:
    if MODE == "smoke":
        return ["smoke"]
    if MODE == "prepare":
        if not DATA_ROOT:
            raise ValueError("Set DATA_ROOT before running MODE='prepare'")
        return ["prepare", SUBSET, "--data-root", DATA_ROOT]
    if MODE == "run":
        arguments = ["run", SUBSET, "--agent", AGENT]
        if OUTPUT_PATH:
            arguments.extend(["--output-path", OUTPUT_PATH])
        return arguments
    raise ValueError("MODE must be 'smoke', 'prepare', or 'run'")


def _run_smoke() -> int:
    from examples.synthetic_smoke.generate_fixture import main as generate_fixture
    from examples.synthetic_smoke.run_smoke_test import main as run_smoke_test

    fixture = PROJECT_ROOT / "examples" / "synthetic_smoke" / "synthetic_signal.npy"
    if not fixture.exists():
        generate_fixture()
    run_smoke_test()
    return 0


def _prepare(args: argparse.Namespace) -> int:
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {data_root}")

    forwarded = ["--data-root", str(data_root)]

    if args.subset == "foundational_analysis":
        from benchmarks.foundational_analysis.prepare_core_test_inputs import main as prepare
    elif args.subset == "sleep_assessment":
        from benchmarks.sleep_assessment.prepare_sleep_test_inputs import main as prepare
    elif args.subset == "neurocognitive_assessment":
        from benchmarks.neurocognitive_assessment.prepare_emotion_test_inputs import (
            main as prepare,
        )
    elif args.subset == "physiological_integration":
        from benchmarks.physiological_integration.prepare_multi_test_inputs import (
            main as prepare,
        )
    else:
        raise ValueError(f"Unsupported subset: {args.subset}")
    return int(prepare(forwarded))


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _build_agent(
    agent_name: str,
    config: EndpointConfig,
) -> AgentRunner | Callable[[str], AgentRunResult]:
    if agent_name == "custom":
        return run_custom_agent

    mode = os.getenv("BRAINBENCH_CODEACT_MODE", "local").strip().lower()
    if mode == "local":
        print("CodeAct mode: local (model-generated code runs on this machine)")
    else:
        print("CodeAct mode: Docker (model-generated code runs on Docker: brainbench-codeact:latest)")
    return CodeActAgentRunner(
        config,
        work_dir=PROJECT_ROOT,
        execution_mode=mode,
        docker_image=os.getenv("BRAINBENCH_DOCKER_IMAGE", "brainbench-codeact:latest"),
        docker_memory_limit=os.getenv("BRAINBENCH_DOCKER_MEMORY", "8g"),
        docker_cpu_count=_float_env("BRAINBENCH_DOCKER_CPUS", 4.0),
    )


def _run(args: argparse.Namespace) -> int:
    if (args.subset is None) == (args.case_path is None):
        raise ValueError("Specify exactly one of <subset> or --case-path")

    subsets = _subsets()
    case_path = args.case_path.expanduser().resolve() if args.case_path else None
    case_json = None
    if case_path is not None:
        if not case_path.is_file():
            raise FileNotFoundError(f"case JSON does not exist: {case_path}")
        case_json = load_case_json(case_path)
        subset_id = _subset_for_case(case_path, case_json, subsets)
    else:
        subset_id = args.subset

    if case_path is not None and args.output_path is not None:
        raise ValueError("--output-path is only supported for a complete subset run")

    manifest = subsets[subset_id]
    manifest_path = Path(manifest["manifest_path"])
    subset_root = (manifest_path.parent / str(manifest["case_root"])).resolve()
    config = EndpointConfig.from_env()
    parser_config = EndpointConfig.from_env(prefix="PARSER")
    evaluator = NeuroBenchEvaluator(
        agent_runner=_build_agent(args.agent, config),
        parser_agent=make_json_parser(parser_config),
        workspace_root=PROJECT_ROOT,
        semantic_judge=make_semantic_judge(parser_config),
        vlm_judge=make_vlm_judge(parser_config),
    )
    if case_path is None:
        output_path = args.output_path.expanduser() if args.output_path else None
        if output_path is None:
            output_path = PROJECT_ROOT / "runs" / f"{subset_id}.json"
        elif not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path = output_path.resolve()
        summary = evaluator.run_subset(
            subset_root,
            output_path=output_path,
        )
        print(f"subset: {subset_id}")
        print(f"instances: {summary['instance_count']}")
        print(f"mean_score: {summary['mean_score']}")
    else:
        result = evaluator.run_case(case_json, source_path=case_path)
        instances = [result]
        summary = evaluator._summarize_instances(
            instances, float(result["timing_sec"].get("total", 0.0))
        )
        summary["planned_instance_count"] = 1
        summary["completed_instance_count"] = 1
        _print_single_case_details(
            subset_id=subset_id,
            case_path=case_path,
            result=result,
            summary=summary,
            models=evaluator.model_configs(),
        )
    if case_path is None:
        print(f"output: {output_path.resolve()}")
    return 0


def _print_single_case_details(
    *,
    subset_id: str,
    case_path: Path,
    result: dict[str, Any],
    summary: dict[str, Any],
    models: dict[str, dict[str, Any]],
) -> None:
    """Print the diagnostic fields for a single-case evaluation."""

    print("single_case:")
    print(f"  subset: {subset_id}")
    print(f"  case_path: {case_path}")
    print(f"  case_id: {result.get('case_id')}")
    print(f"  instance_id: {result.get('instance_id')}")
    print("  models:")
    print(json.dumps(models, ensure_ascii=False, indent=2))
    print(
        f"  score: {result.get('total_score', 0.0)} / "
        f"{result.get('max_score', 0.0)}"
    )
    print(f"  vrr: {result.get('vrr', 0.0)}")

    print("  metrics:")
    metric_details = result.get("metric_details", {})
    if isinstance(metric_details, dict) and metric_details:
        for metric_id, detail in metric_details.items():
            if not isinstance(detail, dict):
                print(f"    {metric_id}: {detail}")
                continue
            print(
                f"    {metric_id}: raw={detail.get('raw_score', 0.0)} "
                f"weighted={detail.get('weighted_score', 0.0)} "
                f"type={detail.get('metric_type')}"
            )
            if detail.get("error"):
                print(f"      error: {detail['error']}")
    else:
        print("    none")

    print("  agent_output:")
    print(result.get("agent_output", ""))
    print("  parser_output:")
    print(json.dumps(result.get("parser_output", {}), ensure_ascii=False, indent=2))
    print(
        "  token_usage: "
        + json.dumps(result.get("token_usage", {}), ensure_ascii=False, sort_keys=True)
    )
    print(
        "  timing_sec: "
        + json.dumps(result.get("timing_sec", {}), ensure_ascii=False, sort_keys=True)
    )
    print("  sandbox:")
    print(json.dumps(result.get("sandbox"), ensure_ascii=False, indent=2))
    print("  artifact_manifest:")
    print(json.dumps(result.get("artifact_manifest", []), ensure_ascii=False, indent=2))
    print("  errors:")
    errors = result.get("errors", [])
    if errors:
        for error in errors:
            print(f"    - {error}")
    else:
        print("    none")
    print("  result_json: not saved (single-case mode)")


def _subset_for_case(
    case_path: Path,
    case_json: dict[str, Any],
    subsets: dict[str, dict[str, Any]],
) -> str:
    """Resolve a case to a released subset from its path or metadata."""

    resolved_case = case_path.resolve()
    for subset_id, manifest in subsets.items():
        manifest_path = Path(manifest["manifest_path"])
        subset_root = (manifest_path.parent / str(manifest["case_root"])).resolve()
        try:
            resolved_case.relative_to(subset_root)
        except ValueError:
            continue
        return subset_id

    declared = str(case_json.get("meta_info", {}).get("bench_subset", ""))
    for subset_id, manifest in subsets.items():
        if declared in {
            subset_id,
            str(manifest.get("legacy_name", "")),
            str(manifest.get("display_name", "")),
        }:
            return subset_id
    raise ValueError(
        f"case does not belong to a released BrainBench subset: {case_path}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if not arguments:
            arguments = _ide_arguments()
        parser = _parser()
        args = parser.parse_args(arguments)
        if args.action == "smoke":
            return _run_smoke()
        if args.action == "prepare":
            return _prepare(args)
        return _run(args)
    except (FileNotFoundError, ImportError, NotImplementedError, ValueError) as exc:
        raise SystemExit(f"BrainBench error: {exc}") from exc


"""
IDE QUICK START CONFIGURATION

Run ``main.py`` directly without command-line arguments to use the four values
below. Command-line arguments, when provided, take precedence over this block.

MODE controls which workflow starts:
- "smoke": run the offline synthetic smoke test. SUBSET, AGENT, and DATA_ROOT
  are ignored.
- "prepare": prepare every required dataset for SUBSET. DATA_ROOT is required;
  AGENT is ignored.
- "run": evaluate every case in SUBSET. AGENT selects CodeAct or the user's
  custom Agent adapter. 

AGENT is used only by MODE="run":
- "codeact": use the built-in CodeAct adapter. Its local/docker execution mode
  is configured in .env with BRAINBENCH_CODEACT_MODE.
- "custom": call run_custom_agent(query) above, which the user must implement.

DATA_ROOT is always specific to the currently selected SUBSET. It is the root
of the downloaded raw datasets, not the BrainBench output directory. For IDE
preparation, set MODE="prepare", choose SUBSET, and provide the matching root:
"""

MODE = "run"  # "smoke", "prepare", or "run"
SUBSET = "foundational_analysis"  # or "sleep_assessment" or "neurocognitive_assessment" or "physiological_integration"
AGENT = "codeact"  # "codeact" or "custom"
DATA_ROOT = "/path/to/foundational_analysis_data"  # Direct parent of the raw dataset folders; used only by MODE="prepare".
OUTPUT_PATH = ""  # Optional result JSON path for MODE="run"; relative paths start at the BrainBench project root.


if __name__ == "__main__":
    raise SystemExit(main())
