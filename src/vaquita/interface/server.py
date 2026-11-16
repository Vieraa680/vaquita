import base64
from collections.abc import Callable
from http import HTTPStatus
import http.server
import json
import threading
from typing import Any

from vaquita.core.agent import VaquitaAgent
from vaquita.core.providers import ProviderError
from vaquita.core.voice import (
    BaseAudioRecorder,
    BaseAudioTranscriber,
    VoiceError,
)
from vaquita.factory import create_audio_recorder, create_transcriber
from vaquita.memory.base import BaseProfileStore


class ServerError(Exception):
    """Raised when the local server fails to bind or initialize."""


class VaquitaHTTPServer(http.server.ThreadingHTTPServer):
    """Custom HTTP server holding references to the agent and profile store."""

    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[http.server.BaseHTTPRequestHandler],
        agent_factory: Callable[[], VaquitaAgent],
        profile_store: BaseProfileStore | None = None,
        transcriber: BaseAudioTranscriber | None = None,
        recorder: BaseAudioRecorder | None = None,
        provider_name: str | None = None,
    ) -> None:
        super().__init__(server_address, request_handler_class)
        self.agent_factory = agent_factory
        self.agent = agent_factory()
        self.profile_store = profile_store
        self.transcriber = transcriber or create_transcriber()
        self.recorder = recorder or create_audio_recorder()
        self.provider_name = provider_name


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
            payload: dict[str, Any] = {"status": "ok", "version": "0.1.0"}
            if self.server.provider_name is not None:
                payload["provider"] = self.server.provider_name
            self._send_json(payload)
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
        if self.path not in ("/api/chat", "/api/chat/stream", "/api/voice"):
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
            body: dict[str, Any] = json.loads(raw_body)
        except (ValueError, json.JSONDecodeError):
            self._send_json(
                {"error": "Invalid JSON body."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        if self.path == "/api/voice":
            self._handle_voice(body)
            return

        if self.path == "/api/chat/stream":
            self._handle_chat_stream(body)
            return

        self._handle_chat(body)

    def _send_sse_event(self, event: str, data: dict[str, Any]) -> None:
        chunk = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
        self.wfile.write(chunk)
        self.wfile.flush()

    def _handle_chat_stream(self, body: dict[str, Any]) -> None:
        prompt = str(body.get("prompt", "")).strip()
        if not prompt:
            self._send_json(
                {"error": "'prompt' field cannot be empty."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def on_token(token: str) -> None:
            self._send_sse_event("token", {"token": token})

        try:
            response_text = self.server.agent.run_stream(prompt, on_token=on_token)
            route = self.server.agent.last_route
            intent_data = {
                "label": route.intent if route else "unknown",
                "confidence": route.confidence if route else 0.0,
                "is_confident": route.is_confident if route else False,
            }
            self._send_sse_event(
                "done",
                {"response": response_text, "intent": intent_data},
            )
        except Exception as exc:
            self._send_sse_event("error", {"error": str(exc)})

    def _handle_chat(self, body: dict[str, Any]) -> None:
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

    def _handle_voice(self, body: dict[str, Any]) -> None:
        if self.server.transcriber is None:
            self._send_json(
                {"error": "Voice transcriber is not configured. Set GROQ_API_KEY or install a local Vosk model."},
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return

        try:
            if "audio_base64" in body:
                audio_bytes = base64.b64decode(str(body["audio_base64"]))
            else:
                record_seconds = float(body.get("record_seconds", 4.0))
                audio_bytes = self.server.recorder.record(duration_seconds=record_seconds)

            transcribed_prompt = self.server.transcriber.transcribe(audio_bytes)
            if not transcribed_prompt.strip():
                self._send_json(
                    {
                        "transcribed_prompt": "",
                        "response": "No se detectó comando de voz ni audio inteligible.",
                        "intent": None,
                    }
                )
                return

            response_text = self.server.agent.run(transcribed_prompt)
            route = self.server.agent.last_route
            intent_data = {
                "label": route.intent if route else "unknown",
                "confidence": route.confidence if route else 0.0,
                "is_confident": route.is_confident if route else False,
            }
            self._send_json(
                {
                    "transcribed_prompt": transcribed_prompt,
                    "response": response_text,
                    "intent": intent_data,
                }
            )
        except VoiceError as exc:
            self._send_json(
                {"error": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )
        except ProviderError as exc:
            self._send_json(
                {"error": str(exc)},
                status=HTTPStatus.BAD_GATEWAY,
            )
        except Exception as exc:
            self._send_json(
                {"error": f"Internal voice processing error: {exc}"},
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
        transcriber: BaseAudioTranscriber | None = None,
        recorder: BaseAudioRecorder | None = None,
        provider_name: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.agent_factory = agent_factory
        self.profile_store = profile_store
        self.transcriber = transcriber
        self.recorder = recorder
        self.provider_name = provider_name
        self._httpd: VaquitaHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def prepare(self) -> None:
        if self._httpd is not None:
            return
        try:
            self._httpd = VaquitaHTTPServer(
                (self.host, self.port),
                VaquitaRequestHandler,
                agent_factory=self.agent_factory,
                profile_store=self.profile_store,
                transcriber=self.transcriber,
                recorder=self.recorder,
                provider_name=self.provider_name,
            )
            self.port = self._httpd.server_address[1]
        except OSError as exc:
            if exc.errno == 98 or "address already in use" in str(exc).lower():
                raise ServerError(
                    f"El puerto {self.port} ya está en uso por otra instancia de Vaquita o proceso.\n"
                    f"Detené el proceso existente (ej. 'kill $(lsof -t -i :{self.port})') o especificá otro puerto con '--port'."
                ) from exc
            raise ServerError(f"Error al iniciar servidor HTTP en {self.host}:{self.port}: {exc}") from exc

    def start_background(self) -> None:
        self.prepare()
        if self._httpd is None:
            return
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            daemon=True,
        )
        self._thread.start()

    def serve_forever(self) -> None:
        self.prepare()
        if self._httpd is None:
            return
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
