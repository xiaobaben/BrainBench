from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class CodeActSandboxConfig:
    image: str = "neurobench-codeact:2026.07-anthropic"
    runs_root: Path = Path("runs")
    memory_limit: str = "8g"
    agent_memory_limit: str = "32g"
    cpu_count: float = 4.0
    workspace_size: str = "4g"
    execution_timeout_sec: float = 150.0
    case_timeout_sec: float = 720.0
    max_iterations: int = 20
    max_completion_tokens: int = 4096
    max_non_action_responses: int = 3
    repeated_failure_limit: int = 3
    observation_max_chars: int = 8000
    stale_ttl_sec: float = 86400.0
    allowed_input_roots: Tuple[Path, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CaseRunContext:
    run_id: str
    instance_id: str
    case_root: Path
    host_workspace: Path
    runtime_dir: Path
    input_mounts: Mapping[Path, str]
    container_workspace: str = "/workspace"
    audit: Dict[str, Any] = field(default_factory=dict, compare=False)


class SandboxInputError(ValueError):
    pass


def create_run_root(config: CodeActSandboxConfig, run_id: Optional[str] = None) -> Path:
    run_id = run_id or f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    temp_parent = Path(config.runs_root) / ".tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    root = temp_parent / run_id
    root.mkdir(parents=False, exist_ok=False)
    (root / "run_owner.json").write_text(
        json.dumps({"run_id": run_id, "pid": os.getpid(), "created_at": time.time()}),
        encoding="utf-8",
    )
    return root


def cleanup_run_root(run_root: Optional[Path]) -> None:
    if run_root is None:
        return
    shutil.rmtree(run_root, ignore_errors=True)
    parent = run_root.parent
    try:
        parent.rmdir()
    except OSError:
        pass


def cleanup_stale_run_roots(config: CodeActSandboxConfig) -> None:
    temp_parent = Path(config.runs_root) / ".tmp"
    if not temp_parent.exists():
        return
    now = time.time()
    for root in temp_parent.iterdir():
        marker = root / "run_owner.json"
        try:
            owner = json.loads(marker.read_text(encoding="utf-8"))
            age = now - float(owner["created_at"])
            pid = int(owner["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if age > config.stale_ttl_sec and not _pid_exists(pid):
            cleanup_run_root(root)


def cleanup_stale_containers(config: CodeActSandboxConfig) -> None:
    if shutil.which("docker") is None:
        return
    try:
        listed = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "label=neurobench.managed=true"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return
    if listed.returncode != 0:
        return
    now = time.time()
    for container_id in listed.stdout.split():
        inspected = subprocess.run(
            [
                "docker", "inspect", "--format",
                '{{ index .Config.Labels "neurobench.created_at" }}',
                container_id,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        try:
            created_at = float(inspected.stdout.strip())
        except ValueError:
            continue
        if now - created_at > config.stale_ttl_sec:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                text=True,
                timeout=30,
            )


def remove_instance_containers(instance_id: str) -> None:
    """Remove managed containers left behind by a hard-killed case process."""

    if shutil.which("docker") is None:
        return
    try:
        listed = subprocess.run(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                "label=neurobench.managed=true",
                "--filter",
                f"label=neurobench.instance_id={instance_id}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if listed.returncode != 0:
            return
        container_ids = listed.stdout.split()
        if container_ids:
            subprocess.run(
                ["docker", "rm", "-f", *container_ids],
                capture_output=True,
                text=True,
                timeout=30,
            )
    except (OSError, subprocess.SubprocessError):
        return


def prepare_case_context(
    *,
    config: CodeActSandboxConfig,
    run_root: Path,
    workspace_root: Path,
    instance_id: str,
    agent_input: Mapping[str, Any],
    allow_missing_data_path: bool = False,
) -> tuple[CaseRunContext, Dict[str, Any]]:
    safe_instance = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in instance_id)
    case_root = run_root / f"{safe_instance}-{uuid.uuid4().hex[:8]}"
    host_workspace = case_root / "workspace"
    runtime_dir = run_root / "runtime" / uuid.uuid4().hex[:8]
    host_workspace.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)

    allowed_roots = tuple(Path(path).resolve() for path in config.allowed_input_roots)
    if not allowed_roots:
        allowed_roots = ((workspace_root / "data").resolve(),)

    rewritten = dict(agent_input)
    mounts: Dict[Path, str] = {}
    used_names = set()
    for key in ("data_path", "label_path"):
        value = agent_input.get(key)
        if not value:
            continue

        # Multi cases may declare one logical input as several files (for
        # example, an EEG file and an fNIRS file).  Validate and mount every
        # member independently instead of stringifying the whole list into a
        # path such as "['data/multi/a.edf', 'data/multi/b.edf']".
        is_sequence = isinstance(value, (list, tuple))
        values = list(value) if is_sequence else [value]
        rewritten_values = []
        for item in values:
            if not isinstance(item, (str, os.PathLike)):
                raise SandboxInputError(f"{key} must be a path or a list of paths")
            source = Path(item)
            if not source.is_absolute():
                source = workspace_root / source
            source = source.resolve()
            if not any(_is_relative_to(source, root) for root in allowed_roots):
                raise SandboxInputError(f"{key} is outside allowed input roots: {source}")
            name = source.name
            if name in used_names:
                name = f"{key}_{name}"
            used_names.add(name)
            container_path = f"/input/{name}"
            if not source.is_file():
                if key != "data_path" or not allow_missing_data_path:
                    raise SandboxInputError(f"{key} does not reference an existing file: {source}")
                rewritten_values.append(container_path)
                continue
            mounts[source] = container_path
            rewritten_values.append(container_path)
        rewritten[key] = rewritten_values if is_sequence else rewritten_values[0]
    if "instruction" in rewritten:
        rewritten["instruction"] = (
            f"{rewritten.get('instruction')}\n\n"
            "Runtime file rule: input files under /input are read-only. "
            "Save any requested output artifacts under /workspace/file_check/ "
            "or as a relative path under file_check/."
        )

    context = CaseRunContext(
        run_id=run_root.name,
        instance_id=instance_id,
        case_root=case_root,
        host_workspace=host_workspace,
        runtime_dir=runtime_dir,
        input_mounts=mounts,
    )
    return context, rewritten


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
