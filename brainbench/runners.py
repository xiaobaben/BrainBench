"""Public target adapters. BrainAgent is intentionally absent."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import shutil
from typing import Any, Optional

from .agent import AgentRunResult, AgentRunner, build_agent_query
from .config import EndpointConfig
from .codeact.engine import CodeActRunner
from .codeact.process import run_codeact_case_isolated
from .codeact.sandbox import (
    CodeActSandboxConfig,
    cleanup_run_root,
    create_run_root,
    prepare_case_context,
)

class CodeActAgentRunner(AgentRunner):
    """CodeAct Adapter backed by the preserved CodeAct execution Module."""

    evaluation_mode = "codeact"

    def __init__(
        self,
        config: EndpointConfig,
        *,
        work_dir: str | Path = ".",
        execution_mode: str = "local",
        docker_image: str = "brainbench-codeact:latest",
        docker_memory_limit: str = "8g",
        docker_cpu_count: float = 4.0,
        execution_timeout_sec: float = 150.0,
        case_timeout_sec: float = 720.0,
        max_iterations: int = 20,
    ) -> None:
        self.config = config
        self.work_dir = Path(work_dir).resolve()
        self.execution_mode = execution_mode.strip().lower()
        if self.execution_mode not in {"local", "docker"}:
            raise ValueError("execution_mode must be 'local' or 'docker'")
        self.docker_image = docker_image
        self.docker_memory_limit = docker_memory_limit
        self.docker_cpu_count = docker_cpu_count
        self.execution_timeout_sec = execution_timeout_sec
        self.case_timeout_sec = case_timeout_sec
        self.max_iterations = max_iterations
        self.total_tokens = 0
        self.last_audit: Optional[Mapping[str, Any]] = None
        self.model_config = {
            "role": "target",
            "model": config.model,
            "endpoint_mode": "api",
            "base_url": config.base_url,
            "api_protocol": config.api_protocol,
            "request_timeout_sec": config.request_timeout_sec,
            "evaluation_mode": "codeact",
            "execution_mode": self.execution_mode,
        }

    def run_with_usage(
        self,
        query: str,
        run_context: Optional[Any] = None,
    ) -> AgentRunResult:
        self.last_audit = None
        self._clear_artifact_workspace()
        if self.execution_mode == "docker":
            return self._run_docker(query, run_context)

        runner = CodeActRunner(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            llm_model=self.config.model,
            api_protocol=self.config.api_protocol,
            request_timeout_sec=self.config.request_timeout_sec,
            work_dir=self.work_dir,
            execution_timeout_sec=self.execution_timeout_sec,
            case_timeout_sec=self.case_timeout_sec,
            max_iterations=self.max_iterations,
        )
        try:
            response = runner.run(query)
            self.total_tokens += runner.total_tokens
            return AgentRunResult(
                response=response,
                tokens=runner.total_tokens,
                audit=dict(getattr(runner, "audit", {})) or None,
            )
        finally:
            self.last_audit = dict(getattr(runner, "audit", {}))
            runner.close()

    def _run_docker(
        self,
        query: str,
        run_context: Optional[Any],
    ) -> AgentRunResult:
        instance_id = "brainbench-instance"
        agent_input: Mapping[str, Any] = {"instruction": query}
        if isinstance(run_context, Mapping):
            instance_id = str(run_context.get("instance_id") or instance_id)
            structured_input = run_context.get("agent_input")
            if isinstance(structured_input, Mapping):
                agent_input = structured_input

        sandbox_config = CodeActSandboxConfig(
            image=self.docker_image,
            runs_root=self.work_dir / "runs",
            memory_limit=self.docker_memory_limit,
            cpu_count=self.docker_cpu_count,
            execution_timeout_sec=self.execution_timeout_sec,
            case_timeout_sec=self.case_timeout_sec,
            max_iterations=self.max_iterations,
            allowed_input_roots=((self.work_dir / "data").resolve(),),
        )
        run_root = create_run_root(sandbox_config)
        context = None
        try:
            context, rewritten_input = prepare_case_context(
                config=sandbox_config,
                run_root=run_root,
                workspace_root=self.work_dir,
                instance_id=instance_id,
                agent_input=agent_input,
                allow_missing_data_path=bool(
                    isinstance(run_context, Mapping)
                    and run_context.get("expected_missing_data_path", False)
                ),
            )
            response, tokens = run_codeact_case_isolated(
                query=build_agent_query(rewritten_input),
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                llm_model=self.config.model,
                api_protocol=self.config.api_protocol,
                request_timeout_sec=self.config.request_timeout_sec,
                run_context=context,
                sandbox_config=sandbox_config,
            )
            artifact_source = context.host_workspace / "file_check"
            if artifact_source.is_dir():
                shutil.copytree(
                    artifact_source,
                    self.work_dir / "file_check",
                    dirs_exist_ok=True,
                )
            self.total_tokens += tokens
            return AgentRunResult(
                response=response,
                tokens=tokens,
                audit=dict(context.audit) or None,
            )
        finally:
            if context is not None:
                self.last_audit = dict(context.audit)
            cleanup_run_root(run_root)

    def _clear_artifact_workspace(self) -> None:
        """Prevent artifacts from a previous CodeAct case being reused."""

        artifact_root = self.work_dir / "file_check"
        if artifact_root.is_symlink() or artifact_root.is_file():
            artifact_root.unlink()
        elif artifact_root.is_dir():
            shutil.rmtree(artifact_root)
