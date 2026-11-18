from pathlib import Path

from vaquita.tools.files import ReadFileTool, SearchFilesTool


def test_read_file_tool_reads_line_ranges_and_handles_errors(tmp_path: Path) -> None:
    sample = tmp_path / "notes.txt"
    sample.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")

    tool = ReadFileTool()
    out = tool.execute(path=str(sample), start_line=2, end_line=3)
    assert out == "2: beta\n3: gamma"

    missing = tool.execute(path=str(tmp_path / "nope.txt"))
    assert missing.startswith("Error:")

    bad_range = tool.execute(path=str(sample), start_line=4, end_line=2)
    assert bad_range.startswith("Error:")


def test_search_files_tool_finds_matches_and_skips_ignored_dirs(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def hello():\n    return 'vaquita'\n", encoding="utf-8")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("vaquita\n", encoding="utf-8")

    tool = SearchFilesTool()
    found = tool.execute(pattern="vaquita", directory=str(tmp_path))
    assert "app.py:2: return 'vaquita'" in found
    assert ".git" not in found

    none_found = tool.execute(pattern="nonexistent_xyz", directory=str(tmp_path))
    assert none_found == "No matches found."
