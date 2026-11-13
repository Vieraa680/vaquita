from pathlib import Path

from vaquita.memory import BaseProfileStore, UserProfileStore


def test_user_profile_store_implements_protocol() -> None:
    store = UserProfileStore(path=None)
    assert isinstance(store, BaseProfileStore)


def test_user_profile_store_crud_and_persistence(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    store = UserProfileStore(path=profile_path)

    store.set_fact("name", "Alberto", section="profile")
    store.set_fact("editor", "nvim", section="preferences")
    store.set_fact("vaquita", "All-purpose personal AI assistant", section="projects")
    store.set_fact("wifi_ssid", "HomeNet", section="facts")

    assert store.get_fact("name") == "Alberto"
    assert store.get_fact("editor") == "nvim"
    assert store.get_fact("non_existent", default="none") == "none"

    formatted = store.format_context()
    assert "[User Memory & Profile Context]" in formatted
    assert "- name: Alberto" in formatted
    assert "- editor: nvim" in formatted
    assert "- vaquita: All-purpose personal AI assistant" in formatted

    # Re-open in a fresh instance to ensure atomic disk persistence
    reloaded = UserProfileStore(path=profile_path)
    assert reloaded.get_fact("name") == "Alberto"
    assert reloaded.get_fact("wifi_ssid") == "HomeNet"

    # Remove fact
    removed = reloaded.remove_fact("wifi_ssid", section="facts")
    assert removed is True
    assert reloaded.get_fact("wifi_ssid") is None
