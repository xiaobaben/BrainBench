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
from brainbench.llm import make_json_parser
from brainbench.runners import CodeActAgentRunner

PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
load_dotenv(PROJECT_ROOT / ".env")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_custom_agent(query: str) -> AgentRunResult:
    """Send the complete query to the user's Agent and return its result."""

    # response = your_agent.run(query)
    # return AgentRunResult(response=response, tokens=0)

    raise NotImplementedError(
        "Edit run_custom_agent() in main.py and return AgentRunResult(response=..., tokens=...)"
    )


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

    run = commands.add_parser("run", help="Evaluate one released subset.")
    run.add_argument("subset", choices=_released_subsets())
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
    else:
        from benchmarks.sleep_assessment.prepare_sleep_test_inputs import main as prepare
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
    manifest = _subsets()[args.subset]
    manifest_path = Path(manifest["manifest_path"])
    subset_root = manifest_path.parent / str(manifest["case_root"])
    config = EndpointConfig.from_env()
    evaluator = NeuroBenchEvaluator(
        agent_runner=_build_agent(args.agent, config),
        parser_agent=make_json_parser(config),
        workspace_root=PROJECT_ROOT,
    )
    output_path = args.output_path.expanduser() if args.output_path else None
    if output_path is None:
        output_path = PROJECT_ROOT / "runs" / f"{args.subset}.json"
    elif not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path = output_path.resolve()
    summary = evaluator.run_subset(
        subset_root,
        output_path=output_path,
    )
    print(f"subset: {args.subset}")
    print(f"instances: {summary['instance_count']}")
    print(f"mean_score: {summary['mean_score']}")
    print(f"output: {output_path.resolve()}")
    return 0


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
