import argparse
from pathlib import Path
import sys
from collections.abc import Sequence

from dotenv import load_dotenv

from vaquita.core.agent import VaquitaAgent
from vaquita.core.providers import (
    ProviderAuthError,
    ProviderError,
    StreamingLLMProvider,
    ToolCall,
)
from vaquita.core.router import IntentRoute
from vaquita.core.voice import BaseWakeWordListener, WakeWordEvent, WakeWordModelError
from vaquita.factory import create_agent


def _trigger_hud(action: str = "toggle") -> bool:
    import subprocess
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:8766/{action}",
            data=action.encode("utf-8"),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=0.8):
            return True
    except (urllib.error.URLError, OSError):
        root = Path(__file__).resolve().parents[3]
        for rel in (
            "desktop/src-tauri/target/debug/vaquita-desktop",
            "desktop/src-tauri/target/release/vaquita-desktop",
        ):
            binary = root / rel
            if binary.exists():
                subprocess.Popen([str(binary)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
        return False


def _log_intent(route: IntentRoute) -> None:
    if route.is_confident and route.intent != "unknown":
        print(f"[router] {route.intent} ({route.confidence:.0%})")


def _log_tool_call(call: ToolCall) -> None:
    cmd = call.arguments.get("command", "")
    if cmd:
        print(f"[{call.name}] {cmd}")
    else:
        print(f"[{call.name}] {call.arguments}")


def _confirm_command(command: str) -> bool:
    try:
        ans = input(f"Confirm execution of '{command}'? [y/N]: ").strip().lower()
        return ans in ("y", "yes", "s", "si", "sí")
    except (KeyboardInterrupt, EOFError):
        print()
        return False


def build_agent(
    provider_name: str | None = None,
    auto_approve: bool = False,
    enable_router: bool = True,
    enable_memory: bool = True,
    enable_fast_path: bool = True,
) -> VaquitaAgent:
    confirm_fn = None if auto_approve else _confirm_command
    return create_agent(
        provider_name=provider_name,
        confirm_fn=confirm_fn,
        on_tool_execute=_log_tool_call,
        enable_router=enable_router,
        on_intent=_log_intent,
        enable_memory=enable_memory,
        enable_fast_path=enable_fast_path,
    )


def _execute_and_print(agent: VaquitaAgent, prompt: str) -> None:
    if isinstance(agent, VaquitaAgent) and isinstance(agent._provider, StreamingLLMProvider):
        streamed = False

        def on_token(token: str) -> None:
            nonlocal streamed
            streamed = True
            print(token, end="", flush=True)

        response = agent.run_stream(prompt, on_token=on_token)
        if streamed:
            print()
        elif response:
            print(response)
        return

    response = agent.run(prompt)
    print(response)


def run_interactive(agent: VaquitaAgent) -> None:
    print("Vaquita - Terminal Assistant (type 'exit' to quit, 'clear' to reset history)")
    while True:
        try:
            user_input = input("vaquita> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() in ("clear", "reset"):
            agent.reset()
            print("Session history cleared.")
            continue

        try:
            _execute_and_print(agent, user_input)
        except ProviderError as exc:
            print(f"Error: {exc}", file=sys.stderr)


def load_configuration() -> None:
    load_dotenv()

    user_config = Path.home() / ".config" / "vaquita" / ".env"
    if user_config.is_file():
        load_dotenv(user_config)

    repo_env = Path(__file__).resolve().parent.parent.parent.parent / ".env"
    if repo_env.is_file():
        load_dotenv(repo_env)


def _init_wake_listener(args: argparse.Namespace) -> BaseWakeWordListener | None:
    from vaquita.factory import create_wake_word_listener

    listener = create_wake_word_listener(
        energy_threshold=args.energy_threshold,
        backend=args.wake_backend,
        model_path=args.wake_model,
    )
    if listener is None:
        raise WakeWordModelError(
            "No hay backend de Wake Word disponible (instalá vosk o configurá GROQ_API_KEY)."
        )
    return listener


def _run_serve(args: argparse.Namespace) -> int:
    from vaquita.factory import create_profile_store
    from vaquita.interface.server import ServerError, VaquitaServer

    profile_store = create_profile_store() if args.memory else None
    server = VaquitaServer(
        agent_factory=lambda: build_agent(
            provider_name=args.provider,
            auto_approve=args.yes,
            enable_router=args.router,
            enable_memory=args.memory,
        ),
        profile_store=profile_store,
        host=args.host,
        port=args.port,
    )

    try:
        server.prepare()
    except ServerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    wake_listener: BaseWakeWordListener | None = None
    if args.listen:
        try:
            wake_listener = _init_wake_listener(args)
        except WakeWordModelError as exc:
            print(f"Error de Wake Word: {exc}", file=sys.stderr)
            server.stop()
            return 1

        if wake_listener is not None:
            def on_wake_serve(evt: WakeWordEvent) -> None:
                print(f"\n[wake] Detectado: '{evt.wake_word}' (comando: '{evt.command}')")
                if not _trigger_hud("show"):
                    print(
                        "HUD desktop binary not found. Compile it with 'cargo build --manifest-path desktop/src-tauri/Cargo.toml'.",
                        file=sys.stderr,
                    )

            wake_listener.start(on_wake_serve)
            print("[wake] Daemon activo (escuchando 'Vaquita' en micrófono)...")

    try:
        server.serve_forever()
    finally:
        if wake_listener is not None:
            wake_listener.stop()
    return 0


def _run_listen(args: argparse.Namespace) -> int:
    import threading

    try:
        wake_listener = _init_wake_listener(args)
    except WakeWordModelError as exc:
        print(f"Error de Wake Word: {exc}", file=sys.stderr)
        return 1

    if wake_listener is None:
        return 1

    active_agent: VaquitaAgent | None = None

    def on_wake_cli(evt: WakeWordEvent) -> None:
        nonlocal active_agent
        print(f"\n[wake] Detectado: '{evt.wake_word}' (comando: '{evt.command or 'abriendo ventana...'}')")
        _trigger_hud("show")

        if evt.command:
            if active_agent is None:
                active_agent = build_agent(
                    provider_name=args.provider,
                    auto_approve=args.yes,
                    enable_router=args.router,
                    enable_memory=args.memory,
                )
            try:
                resp = active_agent.run(evt.command)
                print(f"\n[vaquita] {resp}\n")
            except Exception as exc:
                print(f"Error procesando comando: {exc}", file=sys.stderr)

    wake_listener.start(on_wake_cli)
    print("[wake] Escuchando Wake Word 'Vaquita' en micrófono... (Ctrl+C para detener)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nDeteniendo Wake Word listener...")
        wake_listener.stop()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    load_configuration()

    parser = argparse.ArgumentParser(
        prog="vaquita",
        description="Virtual Assistant Query & Utility Interface for Terminal Automation",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Auto-approve modifying commands without interactive confirmation.",
    )
    parser.add_argument(
        "--provider",
        choices=["groq", "nvidia", "llamacpp", "openai_compat"],
        help="LLM provider to use (overrides VAQUITA_PROVIDER env var).",
    )
    parser.add_argument(
        "--router",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable local neural intent routing (default: enabled).",
    )
    parser.add_argument(
        "--memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable persistent user memory (default: enabled).",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run Vaquita as a local background HTTP daemon for desktop/mobile clients.",
    )
    parser.add_argument(
        "--hud",
        "--toggle",
        dest="hud",
        action="store_true",
        help="Toggle or launch the desktop assistant window (ideal for Hyprland/Wayland shortcuts).",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Record a voice command from microphone and process it directly.",
    )
    parser.add_argument(
        "--voice-duration",
        type=float,
        default=4.0,
        help="Duration in seconds for voice command capture (default: 4.0).",
    )
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Continuously listen for the local wake word 'Vaquita' in background.",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=300.0,
        help="Audio energy (RMS) threshold for speech activity detection (default: 300.0).",
    )
    parser.add_argument(
        "--wake-backend",
        choices=["auto", "vosk", "transcriber"],
        default="auto",
        help="Wake word detection backend (default: auto, prefers offline vosk).",
    )
    parser.add_argument(
        "--wake-model",
        type=str,
        default=None,
        help="Optional path to a local Vosk model directory.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the local HTTP daemon (default: 8765).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address for the local HTTP daemon (default: 127.0.0.1).",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt to execute directly. If omitted, starts interactive mode.",
    )

    args, remaining = parser.parse_known_args(argv)

    if args.hud:
        if _trigger_hud("toggle"):
            return 0
        print("Desktop binary not found. Compile it with 'cargo build --manifest-path desktop/src-tauri/Cargo.toml'.", file=sys.stderr)
        return 1

    if args.serve:
        return _run_serve(args)

    if args.listen:
        return _run_listen(args)

    try:
        agent = build_agent(
            provider_name=args.provider,
            auto_approve=args.yes,
            enable_router=args.router,
            enable_memory=args.memory,
        )
    except ProviderAuthError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Initialization error: {exc}", file=sys.stderr)
        return 1

    prompt_tokens = list(args.prompt) + remaining
    prompt_text = " ".join(prompt_tokens).strip()

    if args.voice:
        from vaquita.core.voice import VoiceError
        from vaquita.factory import create_audio_recorder, create_transcriber

        transcriber = create_transcriber()
        if transcriber is None:
            print(
                "Error: Voice commands require GROQ_API_KEY or a local Vosk model.",
                file=sys.stderr,
            )
            return 1

        recorder = create_audio_recorder()
        print(f"[voice] Escuchando micrófono ({args.voice_duration:.1f}s)... Habla ahora.")
        try:
            audio_bytes = recorder.record(duration_seconds=args.voice_duration)
            transcribed = transcriber.transcribe(audio_bytes).strip()
        except VoiceError as exc:
            print(f"Error de captura de voz: {exc}", file=sys.stderr)
            return 1

        if not transcribed:
            print("[voice] No se detectó ningún comando de voz inteligible.")
            return 0

        print(f"[voice] Voz reconocida: \"{transcribed}\"")
        prompt_text = transcribed

    if prompt_text:
        try:
            _execute_and_print(agent, prompt_text)
            return 0
        except ProviderError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    run_interactive(agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
