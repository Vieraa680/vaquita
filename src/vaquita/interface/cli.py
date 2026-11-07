import argparse
import sys
from collections.abc import Sequence

from dotenv import load_dotenv

from vaquita.core.agent import VaquitaAgent
from vaquita.core.providers import ProviderAuthError, ProviderError, ToolCall
from vaquita.infrastructure.gemini import GeminiProvider
from vaquita.tools.bash import BashTool


def _log_tool_call(call: ToolCall) -> None:
    cmd = call.arguments.get("command", "")
    if cmd:
        print(f"[{call.name}] {cmd}")
    else:
        print(f"[{call.name}] {call.arguments}")


def build_agent() -> VaquitaAgent:
    provider = GeminiProvider()
    tools = [BashTool()]
    return VaquitaAgent(provider=provider, tools=tools, on_tool_execute=_log_tool_call)


def run_interactive(agent: VaquitaAgent) -> None:
    print("Vaquita - Terminal Assistant (type 'exit' or 'quit' to leave)")
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

        try:
            response = agent.run(user_input)
            print(response)
        except ProviderError as exc:
            print(f"Error: {exc}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="vaquita",
        description="Virtual Assistant Query & Utility Interface for Terminal Automation",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt to execute directly. If omitted, starts interactive mode.",
    )

    args = parser.parse_args(argv)

    try:
        agent = build_agent()
    except ProviderAuthError:
        print(
            "Error: GEMINI_API_KEY is not configured.\n"
            "Please add GEMINI_API_KEY to your .env file or environment variables.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"Initialization error: {exc}", file=sys.stderr)
        return 1

    prompt_text = " ".join(args.prompt).strip()
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
