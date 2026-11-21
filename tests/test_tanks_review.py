import asyncio
import json
from dataclasses import replace

from fastapi.testclient import TestClient

from arena.agents.cli import Completion
from arena.events import EventLog
from arena.tanks.coach import FixedCoach
from arena.tanks.director import Director
from arena.tanks.match import TanksConfig, run_tanks
from arena.tanks.review import MAX_TURNS, Review, match_context
from arena.tanks.strategy import Strategy
from arena.web.server import Arena, create_app

FAST = TanksConfig(timeout_seconds=0.5, realtime=False, yield_every=1, quiet_seconds=999, match_seconds=45)


class EchoAgent:
    name = "echo"

    def __init__(self):
        self.prompts = []

    async def complete(self, prompt, schema=None):
        self.prompts.append(prompt)
        return Completion(f"답변 {len(self.prompts)}", f"답변 {len(self.prompts)}")


def directed_match():
    d = Director(side=0)
    log = EventLog("rev")
    secret = replace(Strategy(), note="상대의 비밀 작전", aggression=0.111)

    async def main():
        async def feed():
            while d._tick() < 600 or not d.active:
                await asyncio.sleep(0)
            d.submit("공격적으로")
        task = asyncio.create_task(feed())
        await run_tanks(FixedCoach(name="me"), FixedCoach(secret, name="opp"), log, FAST, seed="rev", director=d)
        task.cancel()

    asyncio.run(main())
    return log


def test_context_has_own_side_and_effects_but_not_opponent_plan():
    log = directed_match()
    ctx = match_context(log.events, side=0)
    assert ctx["you"] == "me" and ctx["result"] in ("win", "loss")
    assert ctx["director_commands"][0]["text"] == "공격적으로"
    assert ctx["director_commands"][0]["effect"]["delta"]
    lines = ctx["timeline"]
    issued = next(i for i, line in enumerate(lines) if "DIRECTOR #1" in line)
    assert "applied" in lines[issued + 1]
    text = json.dumps(ctx, ensure_ascii=False)
    assert "상대의 비밀 작전" not in text and "0.111" not in text


def test_review_report_and_conversation_are_saved(tmp_path):
    log = directed_match()
    agent, notes = EchoAgent(), []
    review = Review(agent, 0, tmp_path / "m.review.json", notify=lambda t, d: notes.append(t))
    asyncio.run(review.start(log.events))
    assert review.report == "답변 1"
    assert "감독 지시별 효과" in agent.prompts[0] and "공격적으로" in agent.prompts[0]
    reply = asyncio.run(review.chat("왜 졌어?"))
    assert reply == {"ok": True, "reply": "답변 2"}
    assert "Director: 왜 졌어?" in agent.prompts[1] and "답변 1" in agent.prompts[1]
    saved = json.loads((tmp_path / "m.review.json").read_text())
    assert saved["report"] == "답변 1" and [m["role"] for m in saved["messages"]] == ["director", "coach"]
    assert notes[:2] == ["review_started", "review_report"] and notes.count("review_message") == 2


def test_review_chat_guards(tmp_path):
    review = Review(EchoAgent(), 0, tmp_path / "r.json")
    assert not asyncio.run(review.chat("hi"))["ok"]  # report not ready
    asyncio.run(review.start(directed_match().events))
    assert not asyncio.run(review.chat("   "))["ok"]
    for i in range(MAX_TURNS):
        assert asyncio.run(review.chat(f"q{i}"))["ok"]
    assert "limit" in asyncio.run(review.chat("one more"))["error"]


def test_server_hooks(tmp_path):
    calls = []

    async def chat(message):
        return {"ok": True, "reply": message.upper()}

    hooks = {"director_info": lambda: {"side": 0}, "command": lambda text: calls.append(text) or {"ok": True},
             "review_state": lambda: {"ready": False}, "review_chat": chat}
    arena = Arena(FixedCoach(), FixedCoach(), run_tanks, tmp_path, page="tanks.html", hooks=hooks)
    with TestClient(create_app(arena)) as client:
        assert client.get("/api/director").json() == {"side": 0}
        assert client.post("/api/command", json={"text": "공격적으로"}).json() == {"ok": True}
        assert calls == ["공격적으로"]
        assert client.post("/api/review", json={"message": "hi"}).json()["reply"] == "HI"
        assert "감독" in client.get("/").text

    plain = Arena(FixedCoach(), FixedCoach(), run_tanks, tmp_path, page="tanks.html")
    with TestClient(create_app(plain)) as client:
        assert client.get("/api/director").status_code == 404
        assert client.post("/api/command", json={"text": "x"}).status_code == 404
