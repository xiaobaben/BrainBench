"""Hard process boundary for one CodeAct case."""

from __future__ import annotations

import multiprocessing
import time
from multiprocessing.connection import Connection
from typing import Any, Mapping, Optional

from .engine import (
    CodeActExecutionError,
    CodeActRunner,
    DockerKernel,
    SandboxSetupError,
)
from .sandbox import (
    CaseRunContext,
    CodeActSandboxConfig,
    remove_instance_containers,
)
from .transport import exception_diagnostics
from .llm_requests import ReasoningEffort


def _codeact_worker(
    connection: Connection,
    *,
    query: str,
    api_key: str,
    base_url: str,
    llm_model: str,
    api_protocol: str,
    request_timeout_sec: float,
    reasoning_effort: Optional[ReasoningEffort],
    run_context: CaseRunContext,
    sandbox_config: CodeActSandboxConfig,
) -> None:
    runner = CodeActRunner(
        api_key=api_key,
        base_url=base_url,
        llm_model=llm_model,
        api_protocol=api_protocol,
        request_timeout_sec=request_timeout_sec,
        reasoning_effort=reasoning_effort,
        work_dir=run_context.host_workspace,
        execution_timeout_sec=sandbox_config.execution_timeout_sec,
        case_timeout_sec=sandbox_config.case_timeout_sec,
        max_iterations=sandbox_config.max_iterations,
        max_completion_tokens=sandbox_config.max_completion_tokens,
        max_non_action_responses=sandbox_config.max_non_action_responses,
        repeated_failure_limit=sandbox_config.repeated_failure_limit,
        observation_max_chars=sandbox_config.observation_max_chars,
        kernel_factory=lambda: DockerKernel(run_context, sandbox_config),
        eager_kernel=True,
        audit=run_context.audit,
        instance_id=run_context.instance_id,
    )
    try:
        result = runner.run(query)
        payload = {
            "ok": True,
            "result": result,
            "tokens": runner.total_tokens,
            "audit": run_context.audit,
        }
    except SandboxSetupError as exc:
        payload = {
            "ok": False,
            "kind": "sandbox",
            "message": str(exc),
            "tokens": runner.total_tokens,
            "audit": run_context.audit,
            "diagnostics": exception_diagnostics(exc),
        }
    except CodeActExecutionError as exc:
        payload = {
            "ok": False,
            "kind": "codeact",
            "category": exc.category,
            "message": str(exc),
            "tokens": runner.total_tokens,
            "audit": run_context.audit,
            "diagnostics": exception_diagnostics(exc),
        }
    except BaseException as exc:
        payload = {
            "ok": False,
            "kind": "worker",
            "category": "worker_error",
            "message": str(exc),
            "tokens": runner.total_tokens,
            "audit": run_context.audit,
            "diagnostics": exception_diagnostics(exc),
        }
    try:
        connection.send(payload)
    finally:
        connection.close()


def _stop_process(process: multiprocessing.Process) -> None:
    if not process.is_alive():
        process.join(timeout=1.0)
        return
    process.terminate()
    process.join(timeout=5.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=5.0)


def run_codeact_case_isolated(
    *,
    query: str,
    api_key: str,
    base_url: str,
    llm_model: str,
    api_protocol: str = "openai",
    request_timeout_sec: float = 300.0,
    reasoning_effort: Optional[ReasoningEffort] = None,
    run_context: CaseRunContext,
    sandbox_config: CodeActSandboxConfig,
) -> tuple[str, int]:
    """Run one CodeAct case in a killable process with a hard wall-clock cap."""

    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_codeact_worker,
        kwargs={
            "connection": send,
            "query": query,
            "api_key": api_key,
            "base_url": base_url,
            "llm_model": llm_model,
            "api_protocol": api_protocol,
            "request_timeout_sec": request_timeout_sec,
            "reasoning_effort": reasoning_effort,
            "run_context": run_context,
            "sandbox_config": sandbox_config,
        },
        name=f"neurobench-codeact-{run_context.instance_id}",
    )
    started = time.monotonic()
    process.start()
    send.close()
    payload: Mapping[str, Any] | None = None
    try:
        if receive.poll(float(sandbox_config.case_timeout_sec)):
            try:
                payload = receive.recv()
            except EOFError:
                payload = None
        else:
            _stop_process(process)
            remove_instance_containers(run_context.instance_id)
            run_context.audit.update(
                {
                    "codeact_termination": "case_timeout",
                    "case_timeout_sec": float(sandbox_config.case_timeout_sec),
                    "hard_timeout_elapsed_sec": round(
                        time.monotonic() - started, 3
                    ),
                }
            )
            raise CodeActExecutionError(
                "case_timeout",
                "CodeAct case exceeded its "
                f"{float(sandbox_config.case_timeout_sec):g}-second hard deadline",
            )
    finally:
        receive.close()

    process.join(timeout=5.0)
    if process.is_alive():
        _stop_process(process)
    if payload is None:
        remove_instance_containers(run_context.instance_id)
        raise CodeActExecutionError(
            "worker_error",
            f"CodeAct worker exited without a result (exit_code={process.exitcode})",
        )

    run_context.audit.update(dict(payload.get("audit") or {}))
    tokens = int(payload.get("tokens") or 0)
    run_context.audit["codeact_total_tokens"] = tokens
    if payload.get("ok"):
        return str(payload.get("result", "")), tokens

    diagnostics = dict(payload.get("diagnostics") or {})
    if diagnostics:
        run_context.audit["codeact_worker_error"] = diagnostics
    message = str(payload.get("message") or "CodeAct worker failed")
    if payload.get("kind") == "sandbox":
        raise SandboxSetupError(message)
    raise CodeActExecutionError(
        str(payload.get("category") or "worker_error"),
        message,
    )
