from unittest.mock import patch

from vaquita.core.agent import DEFAULT_SYSTEM_PROMPT, build_system_prompt
from vaquita.infrastructure.host import HostInfo, detect_host


def test_detect_host_real() -> None:
    info = detect_host()
    assert isinstance(info, HostInfo)
    assert info.os_name
    assert info.distro_id
    assert info.kernel
    assert info.architecture
    assert info.shell
    assert info.user
    assert info.cwd


def test_detect_host_os_release_fallback() -> None:
    with patch("platform.freedesktop_os_release", side_effect=OSError("Not found")):
        info = detect_host()
        assert info.distro_id == "linux"


def test_detect_host_user_fallback() -> None:
    with patch("getpass.getuser", side_effect=RuntimeError("no user")):
        with patch.dict("os.environ", {"USER": "test_fallback_user"}):
            info = detect_host()
            assert info.user == "test_fallback_user"


def test_host_info_format_context() -> None:
    info = HostInfo(
        os_name="Arch Linux",
        distro_id="arch",
        kernel="Linux 6.18.5",
        architecture="x86_64",
        shell="/bin/bash",
        user="testuser",
        cwd="/home/testuser/project",
    )
    formatted = info.format_context()
    assert "System Environment:" in formatted
    assert "- OS: Arch Linux (arch)" in formatted
    assert "- Kernel & Arch: Linux 6.18.5 x86_64" in formatted
    assert "- Shell: /bin/bash" in formatted
    assert "- User: testuser" in formatted
    assert "- Working Directory: /home/testuser/project" in formatted


def test_build_system_prompt_without_context() -> None:
    assert build_system_prompt(None) == DEFAULT_SYSTEM_PROMPT
    assert build_system_prompt("") == DEFAULT_SYSTEM_PROMPT
    assert build_system_prompt("   ") == DEFAULT_SYSTEM_PROMPT


def test_build_system_prompt_with_context() -> None:
    prompt = build_system_prompt("System Environment:\n- OS: Ubuntu")
    assert prompt.startswith(DEFAULT_SYSTEM_PROMPT)
    assert "System Environment:\n- OS: Ubuntu" in prompt
