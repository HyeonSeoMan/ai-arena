import time

from fastapi.testclient import TestClient

from arena.agents import HeuristicAgent, RandomAgent
from arena.events import verify
from arena.runner import run_match
from arena.web.server import Arena, create_app


def make_client(tmp_path, delay=0.0):
    arena = Arena(HeuristicAgent("h", seed=1, delay=delay), RandomAgent("r", seed=1), run_match, tmp_path)
    return arena, TestClient(create_app(arena))


def wait_finished(client, timeout=5):
    deadline = time.time() + timeout
    while client.get("/api/state").json()["running"]:
        assert time.time() < deadline
        time.sleep(0.02)


def test_index_served(tmp_path):
    _, client = make_client(tmp_path)
    with client:
        assert "AI Arena" in client.get("/").text


def test_live_stream_and_replay_endpoints(tmp_path):
    _, client = make_client(tmp_path, delay=0.01)
    with client:
        match_id = client.post("/api/matches").json()["match_id"]
        with client.websocket_connect("/ws") as ws:
            seen = []
            while not seen or seen[-1]["type"] != "match_ended":
                seen.append(ws.receive_json())
        assert seen[0]["type"] == "match_started"
        assert [e["seq"] for e in seen] == list(range(len(seen)))  # backlog + live, no gaps or dupes
        assert verify(seen).ok

        wait_finished(client)
        listing = client.get("/api/matches").json()
        assert listing[0]["match_id"] == match_id and listing[0]["finished"]
        events = client.get(f"/api/matches/{match_id}").json()
        assert events == seen


def test_rematch_alternates_colours_and_rejects_while_running(tmp_path):
    _, client = make_client(tmp_path, delay=0.05)
    with client:
        first = client.post("/api/matches").json()["match_id"]
        assert client.post("/api/matches").status_code == 409
        wait_finished(client, timeout=10)
        second = client.post("/api/matches").json()["match_id"]
        wait_finished(client, timeout=10)
        a = client.get(f"/api/matches/{first}").json()[0]
        b = client.get(f"/api/matches/{second}").json()[0]
        assert (a["black"], a["white"]) == (b["white"], b["black"])


def test_match_endpoint_rejects_path_traversal(tmp_path):
    _, client = make_client(tmp_path)
    with client:
        assert client.get("/api/matches/..%2F..%2Fetc%2Fpasswd").status_code == 404
        assert client.get("/api/matches/nope").status_code == 404
