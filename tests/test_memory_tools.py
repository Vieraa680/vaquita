from pathlib import Path

from vaquita.memory import EpisodicMemoryStore, UserProfileStore
from vaquita.tools.memory import (
    RecallMemoryTool,
    RememberFactTool,
    SaveEpisodeTool,
)


def test_remember_fact_tool(tmp_path: Path) -> None:
    profile_store = UserProfileStore(path=tmp_path / "profile.json")
    tool = RememberFactTool(profile_store)

    assert tool.definition.name == "remember_fact"

    # Execution
    result = tool.execute(key="favorite_lang", value="python", section="preferences")
    assert "Fact remembered" in result
    assert profile_store.get_fact("favorite_lang") == "python"

    # Validation errors
    assert "Error" in tool.execute(key="", value="test")
    assert "Error" in tool.execute(key="key", value=None)


def test_save_episode_tool(tmp_path: Path) -> None:
    memory_store = EpisodicMemoryStore(db_path=tmp_path / "memory.db")
    tool = SaveEpisodeTool(memory_store)

    assert tool.definition.name == "save_episode"

    result = tool.execute(
        content="Desplegamos el nuevo cluster Kubernetes",
        category="ops",
        tags=["k8s", "cluster"],
    )
    assert "Episode recorded" in result
    assert "Desplegamos" in result

    # Validation error
    assert "Error" in tool.execute(content="")


def test_recall_memory_tool(tmp_path: Path) -> None:
    memory_store = EpisodicMemoryStore(db_path=tmp_path / "memory.db")
    memory_store.store(
        content="Corregimos la fuga de memoria en el backend",
        category="bugfix",
        tags=["performance"],
    )
    tool = RecallMemoryTool(memory_store)

    assert tool.definition.name == "recall_memory"

    result = tool.execute(query="fuga memoria backend")
    assert "Found 1 relevant memories" in result
    assert "bugfix" in result

    empty_result = tool.execute(query="planetas galaxia")
    assert "No matching memories found" in empty_result

    error_result = tool.execute(query="")
    assert "Error" in error_result
