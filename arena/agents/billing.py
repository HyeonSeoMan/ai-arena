"""Subscription-only guard (PRD §6).

No token outside the Claude / ChatGPT subscriptions may be spent. Every agent
subprocess gets an allow-listed environment so API keys can never reach it,
and real matches refuse to start unless both CLIs report subscription login.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass

# BR-02: only these variables are passed to agent subprocesses.
ALLOWED_ENV = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TERM")

# Never passed, even if someone adds them to ALLOWED_ENV by mistake.
FORBIDDEN_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "CODEX_API_KEY",
)


class BillingGuardError(RuntimeError):
    pass


def safe_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if source is None else source
    env = {k: source[k] for k in ALLOWED_ENV if k in source}
    for key in FORBIDDEN_ENV:
        env.pop(key, None)
    return env


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    detail: str


Runner = Callable[[list[str]], str]


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, env=safe_env(), capture_output=True, text=True, timeout=30)
    return proc.stdout + proc.stderr


def check_claude(run: Runner = _run) -> CheckResult:
    """BR-03: `claude auth status` must report claude.ai login on the first-party provider."""
    try:
        status = json.loads(run(["claude", "auth", "status"]))
    except (json.JSONDecodeError, OSError, subprocess.SubprocessError) as exc:
        return CheckResult(False, f"could not read claude auth status: {exc}")
    method, provider = status.get("authMethod"), status.get("apiProvider")
    ok = status.get("loggedIn") is True and method == "claude.ai" and provider == "firstParty"
    return CheckResult(ok, f"authMethod={method}, apiProvider={provider}")


def check_codex(run: Runner = _run) -> CheckResult:
    """BR-03: `codex login status` must report ChatGPT login."""
    try:
        out = run(["codex", "login", "status"]).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult(False, f"could not read codex login status: {exc}")
    return CheckResult("Logged in using ChatGPT" in out, out.splitlines()[0] if out else "no output")


CHECKS: dict[str, Callable[[], CheckResult]] = {"claude": check_claude, "codex": check_codex}


def preflight(kinds: set[str]) -> None:
    """Raise BillingGuardError unless every real CLI in `kinds` is on subscription auth."""
    failures = []
    for kind in sorted(kinds & CHECKS.keys()):
        result = CHECKS[kind]()
        print(f"[billing] {kind}: {'OK' if result.ok else 'FAIL'} ({result.detail})")
        if not result.ok:
            failures.append(f"{kind}: {result.detail}")
    if failures:
        raise BillingGuardError("subscription login required; refusing to start: " + "; ".join(failures))
