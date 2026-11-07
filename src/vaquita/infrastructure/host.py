from dataclasses import dataclass
import getpass
import os
from pathlib import Path
import platform


@dataclass(frozen=True)
class HostInfo:
    os_name: str
    distro_id: str
    kernel: str
    architecture: str
    shell: str
    user: str
    cwd: str

    def format_context(self) -> str:
        lines = [
            "System Environment:",
            f"- OS: {self.os_name} ({self.distro_id})",
            f"- Kernel & Arch: {self.kernel} {self.architecture}",
            f"- Shell: {self.shell}",
            f"- User: {self.user}",
            f"- Working Directory: {self.cwd}",
        ]
        return "\n".join(lines)


def detect_host() -> HostInfo:
    os_name = platform.system()
    distro_id = "linux"

    try:
        os_release = platform.freedesktop_os_release()
        os_name = os_release.get("PRETTY_NAME") or os_release.get("NAME") or os_name
        distro_id = os_release.get("ID", distro_id)
    except OSError:
        pass

    try:
        user = getpass.getuser()
    except Exception:
        user = os.getenv("USER", "unknown")

    return HostInfo(
        os_name=os_name,
        distro_id=distro_id,
        kernel=f"{platform.system()} {platform.release()}",
        architecture=platform.machine(),
        shell=os.getenv("SHELL", "bash"),
        user=user,
        cwd=Path.cwd().as_posix(),
    )
