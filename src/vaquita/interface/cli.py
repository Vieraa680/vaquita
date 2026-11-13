import argparse
from pathlib import Path
import sys
from collections.abc import Sequence

from dotenv import load_dotenv

from vaquita.core.agent import VaquitaAgent
from vaquita.core.providers import (
    ProviderAuthError,
    ProviderError,
    ToolCall,
)
from vaquita.core.router import IntentRoute
from vaquita.factory import create_agent


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
) -> VaquitaAgent:
    confirm_fn = None if auto_approve else _confirm_command
    return create_agent(
        provider_name=provider_name,
        confirm_fn=confirm_fn,
        on_tool_execute=_log_tool_call,
        enable_router=enable_router,
        on_intent=_log_intent,
    )


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
            response = agent.run(user_input)
            print(response)
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
        choices=["groq", "nvidia", "gemini", "openai_compat"],
        help="LLM provider to use (overrides VAQUITA_PROVIDER env var).",
    )
    parser.add_argument(
        "--router",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable local neural intent routing (default: enabled).",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt to execute directly. If omitted, starts interactive mode.",
    )

    args, remaining = parser.parse_known_args(argv)

    try:
        agent = build_agent(
            provider_name=args.provider,
            auto_approve=args.yes,
            enable_router=args.router,
        )
    except ProviderAuthError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Initialization error: {exc}", file=sys.stderr)
        return 1

    prompt_tokens = list(args.prompt) + remaining
    prompt_text = " ".join(prompt_tokens).strip()
    if prompt_text:
        try:
            response = agent.run(prompt_text)
            print(response)
            return 0
        except ProviderError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    run_interactive(agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
