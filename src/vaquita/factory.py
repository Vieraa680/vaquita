from collections.abc import Callable
import os
from pathlib import Path

from vaquita.core.agent import VaquitaAgent, build_system_prompt
from vaquita.core.providers import (
    BaseLLMProvider,
    ProviderAuthError,
    ToolCall,
)
from vaquita.core.router import BaseIntentRouter, IntentRoute
from vaquita.infrastructure.gemini import GeminiProvider
from vaquita.infrastructure.host import detect_host
from vaquita.infrastructure.openai_compat import OpenAICompatibleProvider
from vaquita.memory import (
    BaseMemory,
    BaseProfileStore,
    EpisodicMemoryStore,
    UserProfileStore,
)
from vaquita.neural.router import NeuralIntentRouter
from vaquita.tools.base import BaseTool
from vaquita.tools.bash import BashTool
from vaquita.tools.memory import (
    RecallMemoryTool,
    RememberFactTool,
    SaveEpisodeTool,
)


def create_router(
    model_path: str | None = None,
    dataset_path: str | None = None,
    auto_train: bool = True,
) -> BaseIntentRouter:
    return NeuralIntentRouter.from_path_or_seed(
        model_path=model_path,
        dataset_path=dataset_path,
        auto_train=auto_train,
    )


def create_profile_store(path: Path | str | None = None) -> BaseProfileStore:
    return UserProfileStore(path=path)


def create_episodic_memory(db_path: Path | str | None = None) -> BaseMemory:
    return EpisodicMemoryStore(db_path=db_path)


def create_provider(provider_name: str | None = None) -> BaseLLMProvider:
    selected = (provider_name or os.getenv("VAQUITA_PROVIDER") or "gemini").lower()

    if selected == "groq":
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderAuthError(
                "GROQ_API_KEY is not configured.\n"
                "Please add GROQ_API_KEY to your .env file or environment variables."
            )
        model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        return OpenAICompatibleProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    if selected == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderAuthError(
                "NVIDIA_API_KEY is not configured.\n"
                "Please add NVIDIA_API_KEY to your .env file or environment variables."
            )
        model = os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
        base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        return OpenAICompatibleProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    if selected == "openai_compat":
        return OpenAICompatibleProvider(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )

    if selected == "gemini":
        return GeminiProvider()

    raise ValueError(f"Unknown provider '{selected}'. Supported: groq, nvidia, gemini, openai_compat.")


def create_agent(
    provider_name: str | None = None,
    provider: BaseLLMProvider | None = None,
    tools: list[BaseTool] | None = None,
    auto_approve: bool = False,
    confirm_fn: Callable[[str], bool] | None = None,
    on_tool_execute: Callable[[ToolCall], None] | None = None,
    system_prompt: str | None = None,
    router: BaseIntentRouter | None = None,
    enable_router: bool = False,
    on_intent: Callable[[IntentRoute], None] | None = None,
    profile_store: BaseProfileStore | None = None,
    episodic_memory: BaseMemory | None = None,
    enable_memory: bool = False,
) -> VaquitaAgent:
    resolved_provider = provider or create_provider(provider_name=provider_name)
    resolved_confirm = None if auto_approve else confirm_fn

    resolved_profile = profile_store
    resolved_episodic = episodic_memory
    if enable_memory:
        if resolved_profile is None:
            resolved_profile = create_profile_store()
        if resolved_episodic is None:
            resolved_episodic = create_episodic_memory()

    if tools is not None:
        resolved_tools = list(tools)
    else:
        resolved_tools = [BashTool(confirm_fn=resolved_confirm)]
        if resolved_profile is not None:
            resolved_tools.append(RememberFactTool(resolved_profile))
        if resolved_episodic is not None:
            resolved_tools.extend(
                [
                    SaveEpisodeTool(resolved_episodic),
                    RecallMemoryTool(resolved_episodic),
                ]
            )

    if system_prompt is not None:
        resolved_system_prompt = system_prompt
    else:
        memory_ctx = resolved_profile.format_context() if resolved_profile else None
        resolved_system_prompt = build_system_prompt(
            detect_host().format_context(),
            memory_context=memory_ctx,
        )

    resolved_router = router
    if resolved_router is None and enable_router:
        resolved_router = create_router()

    return VaquitaAgent(
        provider=resolved_provider,
        tools=resolved_tools,
        system_prompt=resolved_system_prompt,
        on_tool_execute=on_tool_execute,
        router=resolved_router,
        on_intent=on_intent,
    )
