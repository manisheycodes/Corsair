from pathlib import Path
from types import SimpleNamespace

import app.codex_agent as codex_agent


def test_codex_invocation_is_ephemeral_read_only_and_non_interactive(monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "--version" in command:
            return SimpleNamespace(returncode=0, stdout="codex-cli test\n", stderr="")
        answer_index = command.index("--output-last-message") + 1
        Path(command[answer_index]).write_text("No completed run exists.", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(codex_agent, "_find_codex", lambda: "codex.exe")
    monkeypatch.setattr(codex_agent.subprocess, "run", fake_run)

    answer, version = codex_agent.ask_codex(
        "Does a run exist?",
        {"project": {"status": "awaiting_data"}, "latest_run": None},
    )

    command = commands[0]
    assert command.index("--ask-for-approval") < command.index("exec")
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--ask-for-approval") + 1] == "never"
    assert "--ephemeral" in command
    assert "--ignore-rules" in command
    assert answer == "No completed run exists."
    assert version == "codex-cli test"
