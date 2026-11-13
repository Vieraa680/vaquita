import json
from typing import Any
from unittest.mock import MagicMock
import urllib.error
import urllib.request

from vaquita.core.agent import VaquitaAgent
from vaquita.core.router import IntentRoute
from vaquita.interface.server import VaquitaServer
from vaquita.memory import UserProfileStore


def _request_json(
    url: str,
    method: str = "GET",
    data: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    payload = json.dumps(data).encode("utf-8") if data is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=payload, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as response:
            status = response.status
            body = json.loads(response.read().decode("utf-8")) if response.length != 0 else {}
            return status, body
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode("utf-8")) if exc.length != 0 else {}
        return exc.code, body


def test_server_health_and_options_endpoints() -> None:
    mock_agent = MagicMock(spec=VaquitaAgent)
    server = VaquitaServer(agent_factory=lambda: mock_agent, port=0)
    server.start_background()

    try:
        base_url = f"http://127.0.0.1:{server.port}"

        # GET /api/health
        status, body = _request_json(f"{base_url}/api/health")
        assert status == 200
        assert body["status"] == "ok"
        assert body["version"] == "0.1.0"

        # OPTIONS /api/chat (CORS preflight)
        req = urllib.request.Request(f"{base_url}/api/chat", method="OPTIONS")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 204
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
    finally:
        server.stop()


def test_server_profile_endpoint() -> None:
    mock_agent = MagicMock(spec=VaquitaAgent)
    profile_store = UserProfileStore(path=None)
    profile_store.set_fact("editor", "nvim", section="preferences")

    server = VaquitaServer(
        agent_factory=lambda: mock_agent,
        profile_store=profile_store,
        port=0,
    )
    server.start_background()

    try:
        base_url = f"http://127.0.0.1:{server.port}"
        status, body = _request_json(f"{base_url}/api/profile")
        assert status == 200
        assert "editor: nvim" in str(body["profile_context"])
        assert body["facts"]["preferences"]["editor"] == "nvim"
    finally:
        server.stop()


def test_server_chat_endpoint_success_and_validations() -> None:
    mock_agent = MagicMock(spec=VaquitaAgent)
    mock_agent.run.return_value = "Espacio disponible: 150GB"
    mock_agent.last_route = IntentRoute(
        intent="terminal",
        confidence=0.97,
        is_confident=True,
    )

    server = VaquitaServer(agent_factory=lambda: mock_agent, port=0)
    server.start_background()

    try:
        base_url = f"http://127.0.0.1:{server.port}"

        # Successful chat
        status, body = _request_json(
            f"{base_url}/api/chat",
            method="POST",
            data={"prompt": "revisa el disco"},
        )
        assert status == 200
        assert body["response"] == "Espacio disponible: 150GB"
        assert body["intent"]["label"] == "terminal"
        assert body["intent"]["confidence"] == 0.97
        assert body["intent"]["is_confident"] is True

        # Validation: empty prompt
        status_empty, body_empty = _request_json(
            f"{base_url}/api/chat",
            method="POST",
            data={"prompt": ""},
        )
        assert status_empty == 400
        assert "empty" in str(body_empty["error"])

        # Not found endpoint
        status_404, _ = _request_json(f"{base_url}/api/unknown")
        assert status_404 == 404
    finally:
        server.stop()
