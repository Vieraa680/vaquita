# Vaquita

Virtual Assistant Query & Utility Interface for Terminal Automation.

A lightweight, modular personal AI assistant for Linux designed for task execution, system automation, and terminal interaction.

---

## Architecture Overview

Vaquita follows a strict layered architecture with clear separation of concerns:

- **Interface Layer (`src/vaquita/interface`)**: CLI entrypoint and interactive REPL loop.
- **Core Layer (`src/vaquita/core`)**: Agent orchestration loop, multi-turn history trimming, tool execution flow, and provider abstractions.
- **Factory (`src/vaquita/factory.py`)**: Composition root for provider instantiation and agent assembly.
- **Tools Layer (`src/vaquita/tools`)**: Type-hinted tools with execution guardrails, token-conserving output truncation, and mutating command confirmations.
- **Infrastructure Layer (`src/vaquita/infrastructure`)**: Shell subprocess execution with process group management, host environment inspection, and LLM provider adapters.

---

## Supported LLM Providers

Vaquita uses the Strategy Pattern to decouple agent logic from specific model vendors:

| Provider | Description | Default Model | Configuration Variables |
| :--- | :--- | :--- | :--- |
| `gemini` | Google Gemini API (Free tier supported) | `gemini-3.1-flash-lite` | `GEMINI_API_KEY`, `GEMINI_MODEL` |
| `groq` | Groq high-speed cloud inference | `qwen/qwen3.8-27b` | `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_BASE_URL` |
| `nvidia` | NVIDIA NIM API | `meta/llama-3.3-70b-instruct` | `NVIDIA_API_KEY`, `NVIDIA_MODEL`, `NVIDIA_BASE_URL` |
| `openai_compat` | Generic OpenAI-compatible endpoints | `gpt-4o-mini` | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` |

---

## Installation

### Prerequisites
- Linux OS
- Python >= 3.11
- `uv` (recommended) or `pip`

### Setup

```bash
git clone https://github.com/albers/vaquita.git
cd vaquita

# Create virtual environment and install dependencies with uv
uv sync --all-extras

# Or with standard pip
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Configuration

Vaquita automatically searches for environment configurations in the following order:
1. Shell environment variables.
2. Current working directory `.env`.
3. User configuration directory: `~/.config/vaquita/.env`.

Copy the example configuration:

```bash
cp .env.example .env
```

Configure your preferred provider API keys:

```dotenv
VAQUITA_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## Usage

### Interactive Mode (REPL)

Start the interactive terminal assistant:

```bash
vaquita
```

Interactive commands:
- `clear` or `reset`: Clears current session history.
- `exit` or `quit`: Exits the session.

### Direct Command Mode (One-Shot)

Execute prompts directly from the command line:

```bash
vaquita inspect disk usage and memory consumption
vaquita find files modified in the last 24 hours
```

Options and flags within commands are automatically passed without requiring quotes:

```bash
vaquita ls -la /var/log
```

### Command Flags

- `-y`, `--yes`: Auto-approves mutating commands (e.g. `rm`, `kill`, `systemctl restart`) without interactive prompt confirmation.
- `--provider`: Selects the provider for the execution (`gemini`, `groq`, `nvidia`, `openai_compat`).

```bash
vaquita --provider groq "check system uptime"
vaquita -y "create a backup directory in /tmp/backup"
```

---

## Safety and Guardrails

1. **Destructive Command Blocking**: Commands matching high-risk destructive patterns (e.g. recursive `rm` targeting root, top-level system directories, or wildcards like `rm -rf *`, `rm -rf /etc`, `mkfs`, fork bombs) are unconditionally blocked by `BashTool`.
2. **Mutating Command Confirmation**: Non-destructive commands that alter system state (package installations, file removals, process terminations, git resets) require explicit interactive confirmation (`[y/N]`) unless `--yes` is specified.
3. **Subprocess Isolation**: Commands run inside isolated process groups (`start_new_session=True`), ensuring child processes and pipelines are terminated when timeouts expire.
4. **Token Conservation**: Command outputs exceeding line or character thresholds are truncated intelligently to preserve context budget.

---

## Testing and Verification

Run the test suite and strict type checking:

```bash
# Run all unit and integration tests
pytest

# Run static type analysis
mypy src tests
```
