from collections.abc import Callable
from http import HTTPStatus
import http.server
import json
import threading
from typing import Any

from vaquita.core.agent import VaquitaAgent
from vaquita.core.providers import ProviderError
from vaquita.memory.base import BaseProfileStore


class VaquitaHTTPServer(http.server.ThreadingHTTPServer):
    """Custom HTTP server holding references to the agent and profile store."""

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[http.server.BaseHTTPRequestHandler],
        agent_factory: Callable[[], VaquitaAgent],
        profile_store: BaseProfileStore | None = None,
    ) -> None:
        super().__init__(server_address, request_handler_class)
        self.agent_factory = agent_factory
        self.agent = agent_factory()
        self.profile_store = profile_store


class VaquitaRequestHandler(http.server.BaseHTTPRequestHandler):
    """REST JSON request handler for local Vaquita daemon IPC."""

    server: VaquitaHTTPServer

    def _send_json(
        self,
        data: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._send_json({"status": "ok", "version": "0.1.0"})
            return

        if self.path == "/api/profile":
            store = self.server.profile_store
            facts = store.all_facts() if store else {}
            context = store.format_context() if store else ""
            self._send_json({"profile_context": context, "facts": facts})
            return

        self._send_json(
            {"error": f"Endpoint '{self.path}' not found."},
            status=HTTPStatus.NOT_FOUND,
        )

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self._send_json(
                {"error": f"Endpoint '{self.path}' not found."},
                status=HTTPStatus.NOT_FOUND,
            )
            return

        content_length_header = self.headers.get("Content-Length")
        if not content_length_header:
            self._send_json(
                {"error": "Missing Content-Length header."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        try:
            length = int(content_length_header)
            raw_body = self.rfile.read(length).decode("utf-8")
            body = json.loads(raw_body)
        except (ValueError, json.JSONDecodeError):
            self._send_json(
                {"error": "Invalid JSON body."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        prompt = str(body.get("prompt", "")).strip()
        if not prompt:
            self._send_json(
                {"error": "'prompt' field cannot be empty."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        try:
            response_text = self.server.agent.run(prompt)
            route = self.server.agent.last_route
            intent_data = {
                "label": route.intent if route else "unknown",
                "confidence": route.confidence if route else 0.0,
                "is_confident": route.is_confident if route else False,
            }
            self._send_json(
                {
                    "response": response_text,
                    "intent": intent_data,
                }
            )
        except ProviderError as exc:
            self._send_json(
                {"error": str(exc)},
                status=HTTPStatus.BAD_GATEWAY,
            )
        except Exception as exc:
            self._send_json(
                {"error": f"Internal agent error: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy standard HTTP access log
        pass


class VaquitaServer:
    """Manager for running the local HTTP IPC daemon."""

    def __init__(
        self,
        agent_factory: Callable[[], VaquitaAgent],
        profile_store: BaseProfileStore | None = None,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.host = host
        self.port = port
        self.agent_factory = agent_factory
        self.profile_store = profile_store
        self._httpd: VaquitaHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start_background(self) -> None:
        self._httpd = VaquitaHTTPServer(
            (self.host, self.port),
            VaquitaRequestHandler,
            agent_factory=self.agent_factory,
            profile_store=self.profile_store,
        )
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            daemon=True,
        )
        self._thread.start()

    def serve_forever(self) -> None:
        self._httpd = VaquitaHTTPServer(
            (self.host, self.port),
            VaquitaRequestHandler,
            agent_factory=self.agent_factory,
            profile_store=self.profile_store,
        )
        self.port = self._httpd.server_address[1]
        print(f"Vaquita daemon running on http://{self.host}:{self.port} (Ctrl+C to stop)")
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down Vaquita daemon...")
        finally:
            self.stop()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
