from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .config import BASE_DIR, CODEX_EXECUTABLE, CODEX_MODEL, CODEX_TIMEOUT_SECONDS


class CodexUnavailableError(RuntimeError):
    pass


class CodexInvocationError(RuntimeError):
    pass


def codex_status() -> dict[str, Any]:
    executable = _find_codex()
    if not executable:
        return {"available": False, "version": None, "detail": "Codex CLI was not found on PATH."}
    try:
        completed = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "version": None, "detail": f"Codex CLI could not be started: {exc}"}
    version = (completed.stdout or completed.stderr).strip().splitlines()[-1] if (completed.stdout or completed.stderr).strip() else None
    return {
        "available": completed.returncode == 0,
        "version": version,
        "detail": "Codex CLI is available." if completed.returncode == 0 else "Codex CLI returned an error.",
    }


def ask_codex(question: str, project_context: dict[str, Any]) -> tuple[str, str | None]:
    executable = _find_codex()
    if not executable:
        raise CodexUnavailableError("Codex CLI is not installed or is not available on PATH")

    prompt = _build_prompt(question, project_context)
    with tempfile.TemporaryDirectory(prefix="corsair-codex-") as temporary_directory:
        answer_path = Path(temporary_directory) / "answer.txt"
        command = [
            executable,
            "--sandbox",
            "read-only",
            "--ask-for-approval",
            "never",
            "--cd",
            str(BASE_DIR),
        ]
        if CODEX_MODEL:
            command.extend(["--model", CODEX_MODEL])
        command.extend([
            "exec",
            "--ephemeral",
            "--ignore-rules",
            "--color",
            "never",
            "--output-last-message",
            str(answer_path),
            "-",
        ])
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=CODEX_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CodexInvocationError(f"Codex did not finish within {CODEX_TIMEOUT_SECONDS} seconds") from exc
        except OSError as exc:
            raise CodexUnavailableError(f"Codex CLI could not be started: {exc}") from exc

        if completed.returncode != 0:
            detail = _short_error(completed.stderr or completed.stdout)
            raise CodexInvocationError(f"Codex exited with code {completed.returncode}: {detail}")
        if not answer_path.is_file():
            raise CodexInvocationError("Codex completed without producing an answer")
        answer = answer_path.read_text(encoding="utf-8").strip()
        if not answer:
            raise CodexInvocationError("Codex returned an empty answer")

    status = codex_status()
    return answer, status.get("version")


def project_context(project: Any, spec: dict, dataset: Any | None, run: Any | None, load_json) -> dict[str, Any]:
    profile = load_json(dataset["profile_json"], {}) if dataset else None
    result = load_json(run["result_json"], {}) if run and run["result_json"] else None
    return {
        "project": {
            "name": project["name"],
            "status": project["status"],
            "problem_statement": project["problem_statement"],
        },
        "approved_specification": spec,
        "dataset_profile": profile,
        "latest_run": _compact_run(run, result),
    }


def _compact_run(run: Any | None, result: dict | None) -> dict[str, Any] | None:
    if not run:
        return None
    compact = {"status": run["status"], "error": run["error"]}
    if result:
        for key in (
            "task_type", "primary_metric", "metric_direction", "leaderboard", "best_model",
            "best_parameters", "rows_used", "features_used", "validation", "search",
            "cluster_summary", "cluster_profiles", "model_card",
        ):
            if key in result:
                compact[key] = result[key]
    return compact


def _build_prompt(question: str, context: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are the read-only ML advisor inside Corsair, a local tabular AutoML MVP.",
            "Answer the user's question using only the supplied project snapshot.",
            "Do not edit files, run commands, or claim that an experiment was performed.",
            "Clearly separate observed facts from recommendations and mention missing evidence.",
            "Keep the answer practical and concise for a non-expert user.",
            "",
            f"User question: {question}",
            "",
            "Corsair project snapshot:",
            json.dumps(context, indent=2, default=str)[:50_000],
        ]
    )


def _find_codex() -> str | None:
    if CODEX_EXECUTABLE:
        configured = Path(CODEX_EXECUTABLE).expanduser()
        return str(configured) if configured.is_file() else None
    # Prefer the native executable on Windows over npm's extensionless shim.
    return shutil.which("codex.exe") or shutil.which("codex.cmd") or shutil.which("codex")


def _short_error(value: str, limit: int = 800) -> str:
    cleaned = value.strip()
    return cleaned[-limit:] if cleaned else "no error output"
