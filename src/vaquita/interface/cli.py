import argparse
import os
import sys
from collections.abc import Sequence

from dotenv import load_dotenv

from vaquita.core.agent import VaquitaAgent
from vaquita.core.providers import (
    BaseLLMProvider,
    ProviderAuthError,
    ProviderError,
    ToolCall,
)
from vaquita.infrastructure.gemini import GeminiProvider
from vaquita.infrastructure.openai_compat import OpenAICompatibleProvider
from vaquita.tools.bash import BashTool


def _log_tool_call(call: ToolCall) -> None:
    cmd = call.arguments.get("command", "")
    if cmd:
        print(f"[{call.name}] {cmd}")
    else:
        print(f"[{call.name}] {call.arguments}")


def build_agent(provider_name: str | None = None) -> VaquitaAgent:
    selected = (provider_name or os.getenv("VAQUITA_PROVIDER") or "gemini").lower()

    provider: BaseLLMProvider
    if selected == "groq":
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderAuthError(
                "GROQ_API_KEY is not configured.\n"
                "Please add GROQ_API_KEY to your .env file or environment variables."
            )
        model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        provider = OpenAICompatibleProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    elif selected == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderAuthError(
                "NVIDIA_API_KEY is not configured.\n"
                "Please add NVIDIA_API_KEY to your .env file or environment variables."
            )
        model = os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
        base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        provider = OpenAICompatibleProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    elif selected == "openai_compat":
        provider = OpenAICompatibleProvider(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )

    elif selected == "gemini":
        provider = GeminiProvider()

    else:
        raise ValueError(f"Unknown provider '{selected}'. Supported: groq, nvidia, gemini, openai_compat.")

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
        "--provider",
        choices=["groq", "nvidia", "gemini", "openai_compat"],
        help="LLM provider to use (overrides VAQUITA_PROVIDER env var).",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt to execute directly. If omitted, starts interactive mode.",
    )

    args = parser.parse_args(argv)

    try:
        agent = build_agent(provider_name=args.provider)
    except ProviderAuthError as exc:
        print(f"Error: {exc}", file=sys.stderr)
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
