# Vaquita

Virtual Assistant Query & Utility Interface for Terminal Automation.

A lightweight, modular personal AI assistant for Linux designed for task execution, system automation, and terminal interaction.

---

## Architecture Overview

Vaquita follows a layered architecture with strict separation of concerns:

- **Interface Layer (`src/vaquita/interface`)**: CLI entrypoint, interactive REPL, and local HTTP/SSE daemon.
- **Core Layer (`src/vaquita/core`)**: Agent orchestration loop, query segmentation, tool selection, history trimming, and provider abstractions.
- **Neural Layer (`src/vaquita/neural`)**: Local pure-Python TF-IDF + MLP intent router with continuous learning.
- **Memory Layer (`src/vaquita/memory`)**: Persistent user profile facts and SQLite episodic memory.
- **Factory (`src/vaquita/factory.py`)**: Composition root for provider instantiation and agent assembly.
- **Tools Layer (`src/vaquita/tools`)**: Type-hinted tools with execution guardrails, output truncation, and mutating command confirmations.
- **Infrastructure Layer (`src/vaquita/infrastructure`)**: Shell subprocess execution, host inspection, deterministic fast-path, voice/wake-word adapters, and OpenAI-compatible LLM client.

---

## Supported LLM Providers

Vaquita uses the Strategy Pattern to decouple agent logic from specific model vendors:

| Provider | Description | Default Model | Configuration Variables |
| :--- | :--- | :--- | :--- |
| `groq` | Groq high-speed cloud inference | `qwen/qwen3.8-27b` | `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_BASE_URL` |
| `llamacpp` | Local `llama.cpp` server (offline, $0 cost) | `local-model` | `LLAMACPP_BASE_URL`, `LLAMACPP_MODEL`, `LLAMACPP_API_KEY` |
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

Vaquita searches for environment configurations in the following order:
1. Shell environment variables.
2. Current working directory `.env`.
3. User configuration directory: `~/.config/vaquita/.env`.

Copy the example configuration:

```bash
cp .env.example .env
```

Configure your preferred provider:

```dotenv
VAQUITA_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
```

For local offline inference with `llama.cpp`:

```dotenv
VAQUITA_PROVIDER=llamacpp
LLAMACPP_BASE_URL=http://127.0.0.1:8080/v1
```

---

## Usage

### Interactive Mode (REPL)

Start the interactive terminal assistant:

```bash
vaquita
```

Interactive commands:
- `/teach <intent> <phrase>`: Adds a training phrase to the local intent router and retrains it.
- `clear` or `reset`: Clears current session history.
- `exit` or `quit`: Exits the session.

### Direct Command Mode (One-Shot)

Execute prompts directly from the command line:

```bash
vaquita inspect disk usage and memory consumption
vaquita find files modified in the last 24 hours
```

Options and flags within commands are passed without requiring quotes:

```bash
vaquita ls -la /var/log
```

### Command Flags

- `-y`, `--yes`: Auto-approves mutating commands (e.g. `rm`, `kill`, `systemctl restart`) without interactive confirmation.
- `--provider`: Selects the provider (`groq`, `llamacpp`, `nvidia`, `openai_compat`).
- `--teach INTENT TEXT`: Teaches the local intent router a new phrase and retrains.
- `--no-router`: Disables the local neural intent router.
- `--no-memory`: Disables persistent profile and episodic memory.
- `--serve`: Starts the local HTTP/SSE daemon (`--host`, `--port`).
- `--no-daemon`: Bypasses a running local daemon and executes standalone.
- `--voice`: Records microphone input for a single voice command (`--voice-duration`).
- `--listen`: Runs continuous offline wake-word detection (`--wake-backend`, `--wake-model`).

```bash
vaquita --provider llamacpp "estado de la ram"
vaquita --teach sys_ram "como viene la memoria"
vaquita -y "create a backup directory in /tmp/backup"
```

---

## Safety and Guardrails

1. **Destructive Command Blocking**: Commands matching high-risk destructive patterns (e.g. recursive `rm` targeting root, top-level system directories, or wildcards like `rm -rf *`, `rm -rf /etc`, `mkfs`, fork bombs) are unconditionally blocked by `BashTool`.
2. **Mutating Command Confirmation**: Non-destructive commands that alter system state (package installations, file removals, process terminations, git resets) require explicit interactive confirmation (`[y/N]`) unless `--yes` is specified.
3. **Subprocess Isolation**: Commands run inside isolated process groups (`start_new_session=True`), ensuring child processes and pipelines are terminated when timeouts expire.
4. **Token Conservation**: Command outputs exceeding line or character thresholds are truncated to preserve context budget.

---

## Testing and Verification

Run the test suite and strict type checking:

```bash
uv run pytest
uv run mypy src
```
