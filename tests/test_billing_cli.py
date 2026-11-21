"""Billing guard and CLI adapter tests. No real CLI is ever invoked (BR-05)."""

import asyncio
import json
import os
import stat
import sys
import textwrap

import pytest

from arena.agents import billing
from arena.agents.base import Observation
from arena.agents.billing import FORBIDDEN_ENV, BillingGuardError, check_claude, check_codex, preflight, safe_env
from arena.agents.cli import ClaudeCodeAgent, CodexAgent
from arena.agents.prompt import build_prompt
from arena.engine import WHITE, Game


def obs(**kw):
    g = Game()
    g.play(4, 4)
    return Observation("t", 2, WHITE, [r[:] for r in g.board], g.to_text(), [(1, 4, 4)], **kw)


def test_safe_env_drops_api_keys_and_unlisted_vars():
    source = {k: "secret" for k in FORBIDDEN_ENV} | {"PATH": "/bin", "HOME": "/h", "CLAUDECODE": "1", "FOO": "x"}
    env = safe_env(source)
    assert env == {"PATH": "/bin", "HOME": "/h"}


def test_safe_env_strips_forbidden_even_if_allow_listed(monkeypatch):
    monkeypatch.setattr(billing, "ALLOWED_ENV", (*billing.ALLOWED_ENV, "OPENAI_API_KEY"))
    assert "OPENAI_API_KEY" not in safe_env({"OPENAI_API_KEY": "sk-x", "PATH": "/bin"})


@pytest.mark.parametrize(
    "status,ok",
    [
        ({"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty"}, True),
        ({"loggedIn": True, "authMethod": "api_key", "apiProvider": "firstParty"}, False),
        ({"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "bedrock"}, False),
        ({"loggedIn": False, "authMethod": "claude.ai", "apiProvider": "firstParty"}, False),
    ],
)
def test_check_claude(status, ok):
    assert check_claude(lambda cmd: json.dumps(status)).ok is ok


def test_check_claude_unreadable_output_fails():
    assert not check_claude(lambda cmd: "not json").ok


@pytest.mark.parametrize(
    "out,ok",
    [("Logged in using ChatGPT\n", True), ("Logged in using an API key - sk-***\n", False), ("Not logged in", False)],
)
def test_check_codex(out, ok):
    assert check_codex(lambda cmd: out).ok is ok


def test_preflight_refuses_when_any_check_fails(monkeypatch):
    monkeypatch.setitem(billing.CHECKS, "claude", lambda: billing.CheckResult(True, "ok"))
    monkeypatch.setitem(billing.CHECKS, "codex", lambda: billing.CheckResult(False, "api key"))
    with pytest.raises(BillingGuardError):
        preflight({"claude", "codex"})


def test_preflight_skips_dummy_agents(monkeypatch):
    def boom():
        raise AssertionError("should not run")

    monkeypatch.setitem(billing.CHECKS, "claude", boom)
    preflight({"heuristic", "random"})


def test_prompt_contains_rules_board_and_feedback():
    text = build_prompt(obs(last_error="occupied: (4, 4) is already occupied", owner_notes="play aggressively"))
    assert "White (O)" in text
    assert obs().board_text in text
    assert "occupied" in text
    assert "play aggressively" in text
    assert '{"row": <int>, "col": <int>}' in text


def test_claude_command_disables_tools_and_user_settings():
    cmd = ClaudeCodeAgent().command()
    assert cmd[:2] == ["claude", "-p"]
    assert "--restricted" in cmd and "--strict-mcp-config" in cmd
    assert cmd[cmd.index("--tools") + 1] == ""


def test_codex_command_is_sandboxed_and_ignores_user_config():
    cmd = CodexAgent().command("/tmp/x")
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"
    assert "--ignore-user-config" in cmd and "--ephemeral" in cmd
    assert cmd[-1] == "-"


def fake_cli(tmp_path, body):
    """Create an executable Python script standing in for a real CLI."""
    path = tmp_path / "fake-cli"
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


ENV_DUMP = """
    import json, os, sys
    sys.stdin.read()
    leaked = [k for k in os.environ if "API_KEY" in k or k.startswith("ANTHROPIC") or k.startswith("OPENAI")]
"""


def test_claude_adapter_parses_result_and_hides_api_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    binary = fake_cli(tmp_path, ENV_DUMP + """
    move = {"row": 3, "col": 3} if not leaked else {"row": 99, "col": 99}
    print(json.dumps({"type": "result", "is_error": False, "result": json.dumps(move), "duration_ms": 5}))
    """)
    reply = asyncio.run(ClaudeCodeAgent(binary=binary).act(obs()))
    assert (reply.row, reply.col) == (3, 3)
    assert reply.error is None


def test_claude_adapter_reports_cli_error(tmp_path):
    binary = fake_cli(tmp_path, """
    import json, sys
    sys.stdin.read()
    print(json.dumps({"type": "result", "is_error": True, "result": "usage limit reached"}))
    sys.exit(1)
    """)
    reply = asyncio.run(ClaudeCodeAgent(binary=binary).act(obs()))
    assert reply.error and "usage limit" in reply.raw


def test_codex_adapter_reads_last_message_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_API_KEY", "sk-should-not-leak")
    binary = fake_cli(tmp_path, ENV_DUMP + """
    out = sys.argv[sys.argv.index("-o") + 1]
    move = {"row": 5, "col": 6} if not leaked else {"row": 99, "col": 99}
    open(out, "w").write(json.dumps(move))
    """)
    reply = asyncio.run(CodexAgent(binary=binary).act(obs()))
    assert (reply.row, reply.col) == (5, 6)


def test_cli_process_is_killed_on_timeout(tmp_path):
    marker = tmp_path / "still-running"
    binary = fake_cli(tmp_path, f"""
    import time, sys
    sys.stdin.read()
    time.sleep(1.5)
    open({str(marker)!r}, "w").write("x")
    """)

    async def go():
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(ClaudeCodeAgent(binary=binary).act(obs()), timeout=0.3)
        await asyncio.sleep(2)

    asyncio.run(go())
    assert not marker.exists()
