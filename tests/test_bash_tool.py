from vaquita.tools.base import BaseTool
from vaquita.tools.bash import BashTool


def test_bash_tool_satisfies_protocol() -> None:
    tool = BashTool()
    assert isinstance(tool, BaseTool)


def test_bash_tool_definition() -> None:
    tool = BashTool()
    defn = tool.definition
    assert defn.name == "bash"
    assert "command" in defn.parameters["properties"]
    assert "command" in defn.parameters["required"]


def test_bash_tool_execute_success() -> None:
    tool = BashTool()
    output = tool.execute(command="echo 'working'")
    assert output == "working"


def test_bash_tool_execute_empty() -> None:
    tool = BashTool()
    output = tool.execute(command="   ")
    assert "Empty command" in output


def test_bash_tool_execute_silent_success() -> None:
    tool = BashTool()
    output = tool.execute(command="true")
    assert "successfully with no output" in output


def test_bash_tool_execute_failure() -> None:
    tool = BashTool()
    output = tool.execute(command="echo 'failure reason' >&2; exit 2")
    assert "exit code 2" in output
    assert "failure reason" in output


def test_bash_tool_blocks_destructive_command() -> None:
    tool = BashTool()
    output = tool.execute(command="rm -rf /")
    assert "Execution blocked" in output


def test_bash_tool_timeout() -> None:
    tool = BashTool(timeout=0.05)
    output = tool.execute(command="sleep 1")
    assert "timed out" in output


def test_bash_tool_line_truncation() -> None:
    tool = BashTool(max_lines=20)
    output = tool.execute(command="python3 -c 'for i in range(1, 101): print(f\"item_{i}\")'")
    assert "item_1" in output
    assert "item_100" in output
    assert "lines omitted to conserve tokens" in output


def test_bash_tool_char_truncation() -> None:
    tool = BashTool(max_chars=200)
    output = tool.execute(command="python3 -c 'print(\"X\" * 1000)'")
    assert "content truncated to conserve tokens" in output
    assert len(output) < 300


def test_bash_tool_stderr_truncation() -> None:
    tool = BashTool(max_lines=20)
    cmd = "python3 -c 'import sys; [sys.stderr.write(f\"err_{i}\\n\") for i in range(1, 101)]; sys.exit(1)'"
    output = tool.execute(command=cmd)
    assert "exit code 1" in output
    assert "lines omitted to conserve tokens" in output
