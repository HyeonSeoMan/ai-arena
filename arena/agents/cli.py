"""Adapters for subscription-authenticated CLI agents (Claude Code, Codex).

Each turn runs a fresh process in an empty temporary directory with an
allow-listed environment (no API keys) and tools disabled as far as each
CLI allows.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import tempfile
from pathlib import Path

from dataclasses import dataclass, field

from arena.agents.base import AgentReply, Observation, parse_move
from arena.agents.billing import safe_env
from arena.agents.prompt import build_prompt

MOVE_SCHEMA = {
    "type": "object",
    "properties": {"row": {"type": "integer"}, "col": {"type": "integer"}},
    "required": ["row", "col"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Completion:
    text: str  # the model's final answer ("" on failure)
    raw: str  # what to show/log: the answer, or CLI output on failure
    error: str | None = None
    meta: dict = field(default_factory=dict)


async def run_cli(cmd: list[str], stdin: str, cwd: str) -> tuple[int, str, str]:
    """Run a command; if the caller is cancelled (turn timeout), kill its whole process group."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        env=safe_env(),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = await proc.communicate(stdin.encode())
    except asyncio.CancelledError:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()
        raise
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


class ClaudeCodeAgent:
    def __init__(self, name: str = "claude", binary: str = "claude", model: str | None = None):
        self.name = name
        self.binary = binary
        self.model = model

    def command(self) -> list[str]:
        cmd = [
            self.binary,
            "-p",
            "--output-format", "json",
            "--restricted",  # no code-running tools, ignores user/project/local settings (e.g. apiKeyHelper)
            "--strict-mcp-config",
            "--no-session-persistence",
            "--tools", "",
        ]
        if self.model:
            cmd += ["--model", self.model]
        return cmd

    async def complete(self, prompt: str, schema: dict | None = None) -> Completion:
        with tempfile.TemporaryDirectory(prefix="arena-claude-") as cwd:
            code, out, err = await run_cli(self.command(), prompt, cwd)
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            return Completion("", out or err, f"claude exited {code} without JSON output")
        text = str(data.get("result", ""))
        if data.get("is_error") or code != 0:
            return Completion("", text or out, f"claude reported an error (exit {code})")
        return Completion(text, text, meta={"duration_ms": data.get("duration_ms")})

    async def act(self, obs: Observation) -> AgentReply:
        return _move_reply(await self.complete(build_prompt(obs), MOVE_SCHEMA))


class CodexAgent:
    def __init__(self, name: str = "codex", binary: str = "codex", model: str | None = None):
        self.name = name
        self.binary = binary
        self.model = model

    def command(self, cwd: str, with_schema: bool = True) -> list[str]:
        cmd = [
            self.binary,
            "exec",
            "--sandbox", "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",  # config.toml can't switch provider; auth still comes from CODEX_HOME
            "--ignore-rules",
            "--color", "never",
            "-o", str(Path(cwd) / "answer.txt"),
            "-C", cwd,
        ]
        if with_schema:
            cmd += ["--output-schema", str(Path(cwd) / "schema.json")]
        if self.model:
            cmd += ["--model", self.model]
        return cmd + ["-"]

    async def complete(self, prompt: str, schema: dict | None = None) -> Completion:
        with tempfile.TemporaryDirectory(prefix="arena-codex-") as cwd:
            if schema is not None:
                (Path(cwd) / "schema.json").write_text(json.dumps(schema))
            code, out, err = await run_cli(self.command(cwd, schema is not None), prompt, cwd)
            answer = Path(cwd) / "answer.txt"
            text = answer.read_text() if answer.exists() else ""
        if not text:
            return Completion("", (err or out)[-2000:], f"codex exited {code} without a final message")
        return Completion(text, text)

    async def act(self, obs: Observation) -> AgentReply:
        return _move_reply(await self.complete(build_prompt(obs), MOVE_SCHEMA))


def _move_reply(c: Completion) -> AgentReply:
    if c.error:
        return AgentReply(raw=c.raw, error=c.error)
    reply = parse_move(c.text)
    return AgentReply(reply.raw, reply.row, reply.col, reply.error, c.meta)
