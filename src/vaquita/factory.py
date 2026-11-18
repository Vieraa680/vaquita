from collections.abc import Callable
from datetime import date
import os
from pathlib import Path

from vaquita.core.agent import VaquitaAgent, build_system_prompt
from vaquita.core.fast_path import BaseFastPath
from vaquita.core.providers import (
    BaseLLMProvider,
    ProviderAuthError,
    ToolCall,
)
from vaquita.core.router import BaseIntentRouter, IntentRoute
from vaquita.core.voice import (
    BaseAudioRecorder,
    BaseAudioTranscriber,
    BaseWakeWordDetector,
    BaseWakeWordListener,
    WakeWordModelError,
)
from vaquita.infrastructure.fast_path import LocalFastPath
from vaquita.infrastructure.host import detect_host
from vaquita.infrastructure.openai_compat import OpenAICompatibleProvider
from vaquita.infrastructure.voice import (
    GroqAudioTranscriber,
    LocalAudioRecorder,
    VoskAudioTranscriber,
)
from vaquita.infrastructure.vosk_wake import VoskWakeWordDetector
from vaquita.infrastructure.wake_word import LocalWakeWordListener
from vaquita.memory import (
    BaseMemory,
    BaseProfileStore,
    EpisodicMemoryStore,
    UserProfileStore,
)
from vaquita.neural.router import NeuralIntentRouter
from vaquita.tools.base import BaseTool
from vaquita.tools.bash import BashTool
from vaquita.tools.files import ReadFileTool, SearchFilesTool
from vaquita.tools.media import MediaControlTool
from vaquita.tools.memory import (
    RecallMemoryTool,
    RememberFactTool,
    SaveEpisodeTool,
)
from vaquita.tools.reminders import (
    BaseReminderStore,
    CancelReminderTool,
    ListRemindersTool,
    SQLiteReminderStore,
    SetReminderTool,
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


def create_reminder_store(db_path: Path | str | None = None) -> BaseReminderStore:
    return SQLiteReminderStore(db_path=db_path)


def create_audio_recorder(sample_rate: int = 16000) -> BaseAudioRecorder:
    return LocalAudioRecorder(sample_rate=sample_rate)


def create_transcriber(
    api_key: str | None = None,
    backend: str | None = None,
    model_path: str | None = None,
) -> BaseAudioTranscriber | None:
    selected = (backend or os.getenv("VAQUITA_TRANSCRIBER") or "auto").lower()

    if selected == "vosk":
        return VoskAudioTranscriber(model_path=model_path)

    if selected == "groq":
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            return None
        return GroqAudioTranscriber(api_key=key)

    if selected == "auto":
        key = api_key or os.getenv("GROQ_API_KEY")
        if key:
            return GroqAudioTranscriber(api_key=key)
        try:
            return VoskAudioTranscriber(model_path=model_path)
        except Exception:
            return None

    raise ValueError(f"Unknown transcriber backend '{selected}'. Supported: auto, groq, vosk.")


def create_wake_word_listener(
    transcriber: BaseAudioTranscriber | None = None,
    detector: BaseWakeWordDetector | None = None,
    energy_threshold: float = 300.0,
    backend: str = "auto",
    model_path: str | None = None,
) -> BaseWakeWordListener | None:
    if detector is not None:
        return LocalWakeWordListener(
            detector=detector,
            energy_threshold=energy_threshold,
        )

    selected = (backend or os.getenv("VAQUITA_WAKE_BACKEND") or "auto").lower()

    if selected == "vosk":
        vosk_detector = VoskWakeWordDetector(model_path=model_path)
        return LocalWakeWordListener(
            detector=vosk_detector,
            energy_threshold=energy_threshold,
        )

    if selected == "transcriber":
        resolved_transcriber = transcriber or create_transcriber()
        if resolved_transcriber is None:
            return None
        return LocalWakeWordListener(
            transcriber=resolved_transcriber,
            energy_threshold=energy_threshold,
        )

    if selected == "auto":
        try:
            vosk_detector = VoskWakeWordDetector(model_path=model_path)
            return LocalWakeWordListener(
                detector=vosk_detector,
                energy_threshold=energy_threshold,
            )
        except WakeWordModelError:
            resolved_transcriber = transcriber or create_transcriber()
            if resolved_transcriber is None:
                return None
            return LocalWakeWordListener(
                transcriber=resolved_transcriber,
                energy_threshold=energy_threshold,
            )

    raise ValueError(f"Unknown wake word backend '{selected}'. Supported: auto, vosk, transcriber.")


def create_provider(provider_name: str | None = None) -> BaseLLMProvider:
    selected = (provider_name or os.getenv("VAQUITA_PROVIDER") or "groq").lower()

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

    if selected == "llamacpp":
        return OpenAICompatibleProvider(
            api_key=os.getenv("LLAMACPP_API_KEY", ""),
            base_url=os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8080/v1"),
            model=os.getenv("LLAMACPP_MODEL", "local-model"),
            temperature=0.15,
            top_p=0.9,
            min_p=0.05,
            repeat_penalty=1.1,
            require_api_key=False,
        )

    if selected == "openai_compat":
        return OpenAICompatibleProvider(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )

    raise ValueError(f"Unknown provider '{selected}'. Supported: groq, nvidia, llamacpp, openai_compat.")


def create_fast_path() -> BaseFastPath:
    return LocalFastPath()


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
    fast_path: BaseFastPath | None = None,
    enable_fast_path: bool = False,
    reminder_store: BaseReminderStore | None = None,
    enable_reminders: bool = False,
    enable_file_tools: bool = False,
    enable_media_tools: bool = False,
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
        if enable_reminders:
            resolved_reminders = reminder_store or create_reminder_store()
            resolved_tools.extend(
                [
                    SetReminderTool(resolved_reminders),
                    ListRemindersTool(resolved_reminders),
                    CancelReminderTool(resolved_reminders),
                ]
            )
        if enable_file_tools:
            resolved_tools.extend([ReadFileTool(), SearchFilesTool()])
        if enable_media_tools:
            resolved_tools.append(MediaControlTool())

    if system_prompt is not None:
        resolved_system_prompt = system_prompt
    else:
        memory_ctx = resolved_profile.format_context() if resolved_profile else None
        resolved_system_prompt = build_system_prompt(
            detect_host().format_context(),
            memory_context=memory_ctx,
            current_date=date.today().isoformat(),
        )

    resolved_router = router
    if resolved_router is None and enable_router:
        resolved_router = create_router()

    resolved_fast_path = fast_path
    if resolved_fast_path is None and enable_fast_path:
        resolved_fast_path = create_fast_path()

    return VaquitaAgent(
        provider=resolved_provider,
        tools=resolved_tools,
        system_prompt=resolved_system_prompt,
        on_tool_execute=on_tool_execute,
        router=resolved_router,
        on_intent=on_intent,
        fast_path=resolved_fast_path,
        episodic_memory=resolved_episodic,
    )
