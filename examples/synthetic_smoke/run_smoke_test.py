"""Run the evaluator without a real EEG file or remote model."""

import json
import re
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brainbench.agent import AgentRunResult, AgentRunner
from brainbench.evaluator import NeuroBenchEvaluator


CASE_PATH = Path(__file__).with_name("case.json")


class SyntheticAgent(AgentRunner):
    def run_with_usage(self, query, run_context=None):
        del run_context
        data_path = re.search(r"Load the data file:\s*(.+)", query)
        if data_path is None:
            raise ValueError("query does not contain a data path")
        signal = np.load(ROOT / data_path.group(1).strip())
        return AgentRunResult(
            response=f"mean_square = {float(np.mean(np.square(signal)))}",
            tokens=0,
        )


def parser(response: str, prompt: str):
    del prompt
    value = re.search(r"mean_square\s*=\s*([-+0-9.eE]+)", response)
    if value is None:
        raise ValueError("mean_square was not reported")
    return {"mean_square": float(value.group(1))}


def main() -> None:
    fixture = Path(__file__).with_name("synthetic_signal.npy")
    if not fixture.exists():
        raise SystemExit("Run generate_fixture.py first")
    case = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    evaluator = NeuroBenchEvaluator(SyntheticAgent(), parser, workspace_root=ROOT)
    result = evaluator.run_case(case, source_path=CASE_PATH)
    assert result["total_score"] == 100.0, result
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
