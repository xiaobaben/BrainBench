"""Minimal CodeAct runner backed by a dedicated Jupyter kernel.

The interaction protocol follows HEARTS' CodeAct implementation:
https://github.com/yang-ai-lab/HEARTS/tree/main/agents/codeact
"""

from __future__ import annotations

import logging
import hashlib
import os
import re
import shutil
import subprocess
import uuid
from importlib.util import find_spec
import time
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Optional

from jupyter_client import BlockingKernelClient, KernelManager
from jupyter_client.connect import write_connection_file
from openai import APIConnectionError, APIStatusError, APITimeoutError, BadRequestError, RateLimitError
from .sandbox import CaseRunContext, CodeActSandboxConfig
from .llm_requests import ReasoningEffort, build_chat_completion_kwargs, supports_stop_sequences
from .transport import (
    CaseDeadlineExceeded,
    is_retryable_api_error,
    make_llm_client,
    make_openai_client,
    request_chat_completion,
)


logger = logging.getLogger(__name__)

# At each turn, first give a concise plan enclosed in <thought>...</thought>.
def create_codeact_prompt() -> str:
    """Return the HEARTS-style CodeAct protocol prompt with a worked example."""

    return """\
You are a helpful assistant assigned a problem-solving task. You have access to
an interactive Python environment to inspect data and calculate the answer.

Return exactly one action block per turn:
<execute>...</execute> or <solution>...</solution>.
Do not output plans, explanations, or text outside the selected block.

Then choose exactly one of these actions:

1) Execute Python code by enclosing it in <execute>...</execute>. The code will
   run in a persistent Python kernel and the output will be returned as an
   Observation. Top-level variables from earlier snippets remain available.
2) When the task is complete, provide the requested final report enclosed in
   <solution>...</solution>. The text inside <solution> is returned to the user,
   so it must follow the task's requested output format and contain all results.

Use code to inspect the provided files instead of guessing. Paths named in the
task are accessible from the current workspace. To conserve context, never print
an entire long signal, label sequence, dataframe, or file.

---
Example task:
The file input/HR.npy contains a 1 Hz heart-rate signal in BPM. Calculate how
many seconds are in the inclusive range 60 to 100 BPM and return JSON with the
key time_in_range.

Assistant:
<thought>I will load the signal and inspect its shape.</thought>
<execute>
import numpy as np
hr = np.load("input/HR.npy")
print(hr.shape)
</execute>

Observation:
(300,)

Assistant:
<thought>I will count samples in range; at 1 Hz the count equals seconds.</thought>
<execute>
time_in_range = float(np.sum((hr >= 60) & (hr <= 100)))
print(time_in_range)
</execute>

Observation:
240.0

Assistant:
<thought>The calculation is complete, so I will return the requested JSON.</thought>
<solution>
{"time_in_range": 240.0}
</solution>

---
The actual task follows in the user message.
"""


CODEACT_SYSTEM_PROMPT = create_codeact_prompt()
FINAL_ANSWER_PROMPT = (
    "The execution budget is exhausted. You must return the best available final "
    "answer now inside <solution>...</solution>. Do not execute more code."
)
HISTORY_SUMMARY_HEADER = "Earlier CodeAct history:"


def _extract_blocks(text: str, tag: str) -> str:
    """Extract and combine all XML-style blocks with the requested tag."""

    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    if start_tag in text and end_tag not in text:
        text += end_tag
    pattern = re.compile(
        rf"{re.escape(start_tag)}(.*?){re.escape(end_tag)}",
        flags=re.DOTALL,
    )
    return "\n".join(block.strip() for block in pattern.findall(text)).strip()


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        texts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if texts:
            return "\n".join(texts)
    reasoning = getattr(message, "reasoning_content", "")
    return reasoning if isinstance(reasoning, str) else str(content or "")


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class _JupyterKernel:
    """One persistent local Python kernel for a single CodeAct request."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir.resolve()
        self.manager: Optional[KernelManager] = None
        self.client: Any = None

    def __enter__(self) -> "_JupyterKernel":
        if find_spec("ipykernel") is None:
            raise RuntimeError(
                "CodeAct requires ipykernel in the active Python environment. "
                "Install it with: python -m pip install ipykernel"
            )
        self.manager = KernelManager(kernel_name="python3")
        self.manager.start_kernel(cwd=str(self.work_dir))
        self.client = self.manager.blocking_client()
        self.client.start_channels()
        self.client.wait_for_ready(timeout=30)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self.client is not None:
                self.client.stop_channels()
        finally:
            if self.manager is not None:
                self.manager.shutdown_kernel(now=True)

    def execute(self, code: str, timeout_sec: float) -> str:
        msg_id = self.client.execute(code, allow_stdin=False, stop_on_error=False)
        outputs = []
        deadline = time.monotonic() + timeout_sec

        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise Empty
                message = self.client.get_iopub_msg(timeout=remaining)
                if message.get("parent_header", {}).get("msg_id") != msg_id:
                    continue

                msg_type = message.get("msg_type")
                content = message.get("content", {})
                if msg_type == "stream":
                    outputs.append(content.get("text", ""))
                elif msg_type in {"execute_result", "display_data"}:
                    text = content.get("data", {}).get("text/plain")
                    if text:
                        outputs.append(text)
                elif msg_type == "error":
                    outputs.append("\n".join(content.get("traceback", [])))
                elif msg_type == "status" and content.get("execution_state") == "idle":
                    break
        except Empty:
            self.manager.interrupt_kernel()
            outputs.append(f"[Execution timed out after {timeout_sec:g} seconds]")

        result = "".join(outputs)
        return _strip_ansi(result) if result else "[Code executed successfully with no output]"


class SandboxSetupError(RuntimeError):
    pass


class CodeActExecutionError(RuntimeError):
    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


def _failure_fingerprint(observation: str) -> Optional[str]:
    if "[Execution timed out after" in observation:
        return "execution_timeout"
    if "Traceback (most recent call last):" in observation:
        lines = [line.strip() for line in observation.splitlines() if line.strip()]
        return lines[-1] if lines else "traceback"
    match = re.search(r"(?:^|\n)([A-Za-z_][\w.]*(?:Error|Exception):[^\n]*)", observation)
    return match.group(1).strip() if match else None


def _truncate_observation(observation: str, max_chars: int) -> str:
    if len(observation) <= max_chars:
        return observation
    marker = f"\n...[{len(observation) - max_chars} characters truncated]...\n"
    content_chars = max(0, max_chars - len(marker))
    head_chars = content_chars // 2
    tail_chars = content_chars - head_chars
    return observation[:head_chars] + marker + observation[-tail_chars:]


def _remaining_time(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CodeActExecutionError("case_timeout", "CodeAct case exceeded its time budget")
    return remaining


def _compact_messages(
    messages: list[dict[str, str]],
    trace: list[dict[str, Any]],
    *,
    retained_pairs: int = 3,
) -> list[dict[str, str]]:
    conversation = messages[2:]
    observation_indexes = [
        index
        for index, message in enumerate(conversation)
        if message["role"] == "user"
        and message["content"].startswith("Observation:\n")
    ]
    if len(observation_indexes) <= retained_pairs:
        return messages

    first_observation = observation_indexes[-retained_pairs]
    tail_start = max(0, first_observation - 1)
    earlier = trace[:-retained_pairs]
    summary_lines = [HISTORY_SUMMARY_HEADER]
    for entry in earlier:
        line = (
            f"iteration={entry['iteration']} action={entry['action']} "
            f"status={entry.get('status', 'unknown')}"
        )
        if entry.get("code_sha256"):
            line += f" code_sha256={entry['code_sha256']}"
        if entry.get("observation_preview"):
            line += f" observation={entry['observation_preview']}"
        summary_lines.append(line)
    base_system = messages[0]["content"].split(
        f"\n\n{HISTORY_SUMMARY_HEADER}", 1
    )[0]
    summary_text = "\n".join(summary_lines)
    system = {
        "role": "system",
        "content": f"{base_system}\n\n{summary_text}",
    }
    return [system, messages[1]] + conversation[tail_start:]


def _validate_message_sequence(messages: list[dict[str, str]]) -> None:
    system_indexes = [
        index for index, message in enumerate(messages) if message["role"] == "system"
    ]
    if system_indexes != [0]:
        raise CodeActExecutionError(
            "invalid_message_sequence",
            f"System message must appear exactly once at index 0; got {system_indexes}",
        )


class DockerKernel:
    """Persistent ipykernel running in a file-isolated Docker container."""

    def __init__(self, context: CaseRunContext, config: CodeActSandboxConfig) -> None:
        self.context = context
        self.config = config
        safe_instance = re.sub(r"[^a-zA-Z0-9_.-]", "-", context.instance_id).lower()
        self.container_name = f"neurobench-{safe_instance[:40]}-{uuid.uuid4().hex[:8]}"
        self.client: Optional[BlockingKernelClient] = None
        self.started = False

    def build_run_command(self, connection_file: Path) -> list[str]:
        uid = os.getuid() if hasattr(os, "getuid") else 1000
        gid = os.getgid() if hasattr(os, "getgid") else 1000
        command = [
            "docker", "run", "-d", "--name", self.container_name,
            "--network", "none",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            "--pids-limit", "128",
            "--memory", self.config.memory_limit,
            "--cpus", str(self.config.cpu_count),
            "--user", f"{uid}:{gid}",
            "--env", "HOME=/tmp",
            "--env", "MPLCONFIGDIR=/tmp/matplotlib",
            "--env", "MPLBACKEND=Agg",
            "--env", "OPENBLAS_NUM_THREADS=1",
            "--env", "OMP_NUM_THREADS=1",
            "--env", "MKL_NUM_THREADS=1",
            "--env", "NUMEXPR_NUM_THREADS=1",
            "--env", "BLIS_NUM_THREADS=1",
            "--mount", (
                f"type=bind,src={self.context.host_workspace},"
                "dst=/workspace"
            ),
            "--tmpfs", f"/tmp:rw,nosuid,nodev,size=1g,uid={uid},gid={gid}",
            "--mount", (
                f"type=bind,src={self.context.runtime_dir},"
                f"dst={self.context.runtime_dir}"
            ),
            "--label", f"neurobench.run_id={self.context.run_id}",
            "--label", f"neurobench.instance_id={self.context.instance_id}",
            "--label", "neurobench.managed=true",
            "--label", f"neurobench.created_at={time.time()}",
        ]
        for source, destination in self.context.input_mounts.items():
            command.extend([
                "--mount",
                f"type=bind,src={source},dst={destination},readonly",
            ])
        command.extend([
            self.config.image,
            "python", "-m", "ipykernel_launcher", "-f", str(connection_file),
        ])
        return command

    def __enter__(self) -> "DockerKernel":
        if os.name != "posix" or shutil.which("docker") is None:
            raise SandboxSetupError("Docker CodeAct sandbox requires Linux and the docker CLI.")
        self._docker(["docker", "version", "--format", "{{.Server.Version}}"])
        digest = self._docker([
            "docker", "image", "inspect", self.config.image, "--format", "{{.Id}}",
        ]).stdout.strip()
        self.context.audit["image_digest"] = digest
        connection_file = self.context.runtime_dir / "connection.json"
        write_connection_file(
            str(connection_file),
            ip=str(self.context.runtime_dir / "kernel"),
            transport="ipc",
            key=os.urandom(32).hex().encode("ascii"),
        )
        try:
            self._docker(self.build_run_command(connection_file))
            self.started = True
            self.client = BlockingKernelClient(connection_file=str(connection_file))
            self.client.load_connection_file()
            self.client.start_channels()
            self.client.wait_for_ready(timeout=30)
            self.context.audit.update({
                "backend": "docker",
                "image": self.config.image,
                "container_name": self.container_name,
                "network": "none",
            })
            return self
        except Exception as exc:
            try:
                logs = self._docker(
                    ["docker", "logs", "--tail", "50", self.container_name],
                    check=False,
                ).stdout.strip()
            except SandboxSetupError:
                logs = ""
            self._remove_container()
            if isinstance(exc, SandboxSetupError):
                raise
            detail = f"; container logs: {logs}" if logs else ""
            raise SandboxSetupError(f"Failed to start Docker CodeAct kernel: {exc}{detail}") from exc

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self.client is not None:
                self.client.stop_channels()
            if self.started:
                self.context.audit["workspace_copy_succeeded"] = self.context.host_workspace.is_dir()
        finally:
            self._remove_container()

    def execute(self, code: str, timeout_sec: float) -> str:
        if self.client is None:
            raise RuntimeError("Docker kernel is not running")
        msg_id = self.client.execute(code, allow_stdin=False, stop_on_error=False)
        outputs = []
        deadline = time.monotonic() + timeout_sec
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise Empty
                message = self.client.get_iopub_msg(timeout=remaining)
                if message.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                msg_type = message.get("msg_type")
                content = message.get("content", {})
                if msg_type == "stream":
                    outputs.append(content.get("text", ""))
                elif msg_type in {"execute_result", "display_data"}:
                    text = content.get("data", {}).get("text/plain")
                    if text:
                        outputs.append(text)
                elif msg_type == "error":
                    outputs.append("\n".join(content.get("traceback", [])))
                elif msg_type == "status" and content.get("execution_state") == "idle":
                    break
        except Empty:
            self._docker(["docker", "kill", "--signal", "SIGINT", self.container_name], check=False)
            outputs.append(f"[Execution timed out after {timeout_sec:g} seconds]")
        result = "".join(outputs)
        return _strip_ansi(result) if result else "[Code executed successfully with no output]"

    @staticmethod
    def _docker(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                command,
                check=check,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SandboxSetupError(f"Docker command failed: {' '.join(command[:4])}: {exc}") from exc

    def _remove_container(self) -> None:
        if shutil.which("docker") is not None:
            try:
                subprocess.run(
                    ["docker", "rm", "-f", self.container_name],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (OSError, subprocess.SubprocessError):
                logger.exception("Failed to remove CodeAct container %s", self.container_name)
        self.started = False


class CodeActRunner:
    """Run an LLM/code-execution loop and return its final answer as ``str``.

    Generated code is executed locally with the permissions of this process. Run
    benchmarks only with models and input cases that are trusted for that access.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        llm_model: str,
        api_protocol: str = "openai",
        request_timeout_sec: float = 300.0,
        work_dir: str | Path = ".",
        execution_timeout_sec: float = 120.0,
        case_timeout_sec: float = 600.0,
        max_iterations: int = 8,
        max_completion_tokens: int = 4096,
        max_non_action_responses: int = 2,
        repeated_failure_limit: int = 2,
        observation_max_chars: int = 8000,
        reasoning_effort: Optional[ReasoningEffort] = None,
        kernel_factory: Optional[Callable[[], Any]] = None,
        eager_kernel: bool = False,
        audit: Optional[dict[str, Any]] = None,
        instance_id: Optional[str] = None,
    ) -> None:
        if api_protocol == "openai":
            self.client = make_openai_client(
                api_key=api_key,
                base_url=base_url,
                request_timeout_sec=request_timeout_sec,
            )
        else:
            self.client = make_llm_client(
                api_key=api_key,
                base_url=base_url,
                request_timeout_sec=request_timeout_sec,
                api_protocol=api_protocol,
            )
        self.llm_model = llm_model
        self.work_dir = Path(work_dir)
        self.execution_timeout_sec = execution_timeout_sec
        self.case_timeout_sec = case_timeout_sec
        self.max_iterations = max_iterations
        self.max_completion_tokens = max_completion_tokens
        self.max_non_action_responses = max_non_action_responses
        self.repeated_failure_limit = repeated_failure_limit
        self.observation_max_chars = observation_max_chars
        self.reasoning_effort = reasoning_effort
        self.kernel_factory = kernel_factory or (lambda: _JupyterKernel(self.work_dir))
        self.eager_kernel = eager_kernel
        self.audit = audit if audit is not None else {}
        self.instance_id = instance_id
        self.total_tokens = 0
        self.api_attempts: list[dict[str, Any]] = []

    def run(self, query: str) -> str:
        messages = [
            {"role": "system", "content": CODEACT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        last_response = ""
        non_action_responses = 0
        last_failure = None
        repeated_failures = 0
        last_code_hash = None
        repeated_code = 0
        kernel: Optional[_JupyterKernel] = None
        started_at = time.monotonic()
        deadline = started_at + self.case_timeout_sec
        trace: list[dict[str, Any]] = []
        self.audit["codeact_policy"] = {
            "case_timeout_sec": self.case_timeout_sec,
            "execution_timeout_sec": self.execution_timeout_sec,
            "max_iterations": self.max_iterations,
            "max_completion_tokens": self.max_completion_tokens,
            "max_non_action_responses": self.max_non_action_responses,
            "repeated_failure_limit": self.repeated_failure_limit,
            "observation_max_chars": self.observation_max_chars,
            "reasoning_effort": self.reasoning_effort or "none",
        }
        self.audit["codeact_trace"] = trace
        self.audit["target_api_attempts"] = self.api_attempts

        try:
            if self.eager_kernel:
                kernel = self.kernel_factory()
                kernel.__enter__()
            for iteration in range(self.max_iterations):
                remaining = _remaining_time(deadline)
                messages = _compact_messages(messages, trace)
                final_iteration = iteration == self.max_iterations - 1
                request_messages = messages
                if final_iteration:
                    request_messages = messages + [
                        {"role": "user", "content": FINAL_ANSWER_PROMPT}
                    ]
                _validate_message_sequence(request_messages)
                request = build_chat_completion_kwargs(
                    model=self.llm_model,
                    messages=request_messages,
                    enable_thinking=False,
                    reasoning_effort=self.reasoning_effort,
                    temperature=0.7,
                    stream=False,
                    # max_tokens=1024 if final_iteration else self.max_completion_tokens,
                    max_tokens=1024 if final_iteration else min(
                        self.max_completion_tokens,
                        2048,
                    )
                )

                if supports_stop_sequences(self.llm_model):
                    request["stop"] = ["</execute>", "</solution>"]

                response = request_chat_completion(
                    self.client,
                    request,
                    deadline=deadline,
                    request_timeout_cap_sec=300.0,
                    max_attempts=3,
                    context={
                        "case_id": self.instance_id or "unknown",
                        "agent_id": "codeact",
                        "model": self.llm_model,
                        "model_call_round": iteration + 1,
                    },
                    audit=self.api_attempts,
                )
                _remaining_time(deadline)
                usage = getattr(response, "usage", None)
                response_tokens = int(getattr(usage, "total_tokens", 0) or 0)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                self.total_tokens += response_tokens
                last_response = _message_text(response.choices[0].message)
                if last_response:
                    messages.append({"role": "assistant", "content": last_response})

                code = _extract_blocks(last_response, "execute")
                solution = _extract_blocks(last_response, "solution")
                entry = {
                    "iteration": iteration + 1,
                    "action": "solution" if solution else "execute" if code else "none",
                    "response_chars": len(last_response),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": response_tokens,
                    "elapsed_sec": round(time.monotonic() - started_at, 3),
                }
                if final_iteration:
                    if solution:
                        entry["status"] = "success"
                        trace.append(entry)
                        self.audit["codeact_termination"] = "success"
                        return str(solution)
                    entry["status"] = "rejected"
                    trace.append(entry)
                    raise CodeActExecutionError(
                        "max_iterations_exceeded",
                        f"CodeAct did not return a solution within {self.max_iterations} iterations",
                    )
                if code:
                    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
                    entry["code_sha256"] = code_hash
                    if code_hash == last_code_hash:
                        repeated_code += 1
                    else:
                        last_code_hash = code_hash
                        repeated_code = 1
                    if repeated_code >= self.repeated_failure_limit:
                        entry["status"] = "repeated_failure"
                        trace.append(entry)
                        raise CodeActExecutionError(
                            "repeated_failure",
                            "CodeAct repeated identical code without making progress",
                        )
                    if kernel is None:
                        kernel = self.kernel_factory()
                        kernel.__enter__()
                    execution_timeout = min(
                        self.execution_timeout_sec, _remaining_time(deadline)
                    )
                    observation = kernel.execute(code, execution_timeout)
                    _remaining_time(deadline)
                    entry["observation_chars"] = len(observation)
                    entry["observation_truncated"] = (
                        len(observation) > self.observation_max_chars
                    )
                    entry["observation_preview"] = " ".join(
                        observation[:500].split()
                    )
                    failure = _failure_fingerprint(observation)
                    if failure is not None and failure == last_failure:
                        repeated_failures += 1
                    elif failure is not None:
                        last_failure = failure
                        repeated_failures = 1
                    else:
                        last_failure = None
                        repeated_failures = 0
                    if repeated_failures >= self.repeated_failure_limit:
                        entry["status"] = "repeated_failure"
                        trace.append(entry)
                        raise CodeActExecutionError(
                            "repeated_failure",
                            f"CodeAct repeated the same execution failure {repeated_failures} times: {failure}",
                        )
                    compact_observation = _truncate_observation(
                        observation, self.observation_max_chars
                    )
                    messages.append({
                        "role": "user",
                        "content": f"Observation:\n{compact_observation}",
                    })
                    entry["status"] = "error" if failure else "success"
                    trace.append(entry)
                    non_action_responses = 0
                    continue
                if solution:
                    entry["status"] = "success"
                    trace.append(entry)
                    self.audit["codeact_termination"] = "success"
                    return str(solution)

                entry["status"] = "missing_action"
                trace.append(entry)
                non_action_responses += 1
                if non_action_responses >= self.max_non_action_responses:
                    raise CodeActExecutionError(
                        "missing_action",
                        "CodeAct repeatedly responded without <execute> or <solution>",
                    )
                messages.append({
                    "role": "user",
                    "content": (
                        "Your response did not contain an action. Return exactly one "
                        "<execute>...</execute> or <solution>...</solution> block."
                    ),
                })

            raise CodeActExecutionError(
                "max_iterations_exceeded",
                f"CodeAct did not return a solution within {self.max_iterations} iterations",
            )
        except SandboxSetupError:
            raise
        except CodeActExecutionError as exc:
            self.audit["codeact_termination"] = exc.category
            raise
        except CaseDeadlineExceeded as exc:
            self.audit["codeact_termination"] = "case_timeout"
            raise CodeActExecutionError("case_timeout", str(exc)) from exc
        except APITimeoutError as exc:
            self.audit["codeact_termination"] = "api_timeout"
            raise CodeActExecutionError("api_timeout", str(exc)) from exc
        except APIConnectionError as exc:
            self.audit["codeact_termination"] = "api_connection_error"
            raise CodeActExecutionError("api_connection_error", str(exc)) from exc
        except RateLimitError as exc:
            self.audit["codeact_termination"] = "api_rate_limit"
            raise CodeActExecutionError("api_rate_limit", str(exc)) from exc
        except APIStatusError as exc:
            status_code = int(getattr(exc, "status_code", 0) or 0)
            if status_code in {408, 409, 425, 429} or status_code >= 500:
                self.audit["codeact_termination"] = "api_server_error"
                raise CodeActExecutionError("api_server_error", str(exc)) from exc
            self.audit["codeact_termination"] = "api_request_error"
            raise CodeActExecutionError("api_request_error", str(exc)) from exc
        except BadRequestError as exc:
            self.audit["codeact_termination"] = "api_request_error"
            raise CodeActExecutionError("api_request_error", str(exc)) from exc
        except Exception as exc:
            if is_retryable_api_error(exc):
                self.audit["codeact_termination"] = "api_infrastructure_error"
                raise CodeActExecutionError(
                    "api_infrastructure_error", str(exc)
                ) from exc
            self.audit["codeact_termination"] = "kernel_error"
            logger.exception("CodeAct execution failed")
            raise CodeActExecutionError("kernel_error", str(exc)) from exc
        finally:
            if kernel is not None:
                kernel.__exit__(None, None, None)
            self.close()

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
