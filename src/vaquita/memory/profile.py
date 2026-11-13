import json
from pathlib import Path
from typing import Any

from vaquita.memory.base import BaseProfileStore

DEFAULT_PROFILE_PATH = Path.home() / ".config" / "vaquita" / "profile.json"


class UserProfileStore(BaseProfileStore):
    """Persistent user profile and facts store with atomic JSON writes."""

    VALID_SECTIONS = ("profile", "preferences", "projects", "facts")

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_PROFILE_PATH
        self._data: dict[str, dict[str, Any]] = {
            section: {} for section in self.VALID_SECTIONS
        }
        self.reload()

    def reload(self) -> None:
        if not self.path.is_file():
            return

        try:
            content = self.path.read_text(encoding="utf-8")
            loaded = json.loads(content)
            if isinstance(loaded, dict):
                for section in self.VALID_SECTIONS:
                    section_data = loaded.get(section)
                    if isinstance(section_data, dict):
                        self._data[section] = {
                            str(k): v for k, v in section_data.items()
                        }
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        temp_path.replace(self.path)

    def get_fact(self, key: str, default: Any = None) -> Any:
        for section in self.VALID_SECTIONS:
            if key in self._data[section]:
                return self._data[section][key]
        return default

    def set_fact(self, key: str, value: Any, section: str = "facts") -> None:
        target_section = section.lower()
        if target_section not in self.VALID_SECTIONS:
            target_section = "facts"

        self._data[target_section][str(key)] = value
        self._save()

    def remove_fact(self, key: str, section: str = "facts") -> bool:
        target_section = section.lower()
        if target_section in self._data and key in self._data[target_section]:
            del self._data[target_section][key]
            self._save()
            return True
        return False

    def all_facts(self) -> dict[str, Any]:
        return {
            section: dict(entries)
            for section, entries in self._data.items()
            if entries
        }

    def format_context(self) -> str:
        blocks: list[str] = []

        if self._data["profile"]:
            lines = [f"  - {k}: {v}" for k, v in sorted(self._data["profile"].items())]
            blocks.append("User Profile:\n" + "\n".join(lines))

        if self._data["preferences"]:
            lines = [f"  - {k}: {v}" for k, v in sorted(self._data["preferences"].items())]
            blocks.append("User Preferences:\n" + "\n".join(lines))

        if self._data["projects"]:
            lines = [f"  - {k}: {v}" for k, v in sorted(self._data["projects"].items())]
            blocks.append("Active Projects:\n" + "\n".join(lines))

        if self._data["facts"]:
            lines = [f"  - {k}: {v}" for k, v in sorted(self._data["facts"].items())]
            blocks.append("Known Facts:\n" + "\n".join(lines))

        if not blocks:
            return ""

        return "[User Memory & Profile Context]\n" + "\n\n".join(blocks)
