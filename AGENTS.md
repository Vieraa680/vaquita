# AGENTS.md

Instructions, architectural standards, and constraints for AI agents and human contributors working on **Vaquita**.

---

## 1. Project Purpose & Scope
**Vaquita** (**V**irtual **A**ssistant **Q**uery & **U**tility **I**nterface for **T**erminal **A**utomation) is a lightweight, modular personal AI assistant for Linux. Its primary goals are task execution, system automation, and intelligent interaction without unnecessary complexity or bloated abstractions.

---

## 2. Git & Commit Conventions

### Subject Line Constraints
- Format: `<type>(<scope>): <short description>` or `<type>: <short description>`
- Maximum length: **50 characters** strictly.
- Mood: Imperative, present tense (`add`, `fix`, `refactor`, not `added` or `fixes`).
- Case: All lowercase for type and subject. No trailing period.

### Allowed Commit Types
- `feat`: New feature or user-facing behavior.
- `fix`: Bug fix.
- `refactor`: Code reorganization with no behavior change.
- `test`: Adding or updating tests.
- `docs`: Documentation changes only.
- `chore`: Build tasks, package configs, maintenance.

### Commit Body
- If explanation or context is needed, leave **one blank line** after the subject.
- Wrap lines at 72 characters.
- Detail the *what* and *why*, not the *how*.

Example:
```git
feat(brain): add gemini provider strategy

Implement BaseLLMProvider protocol and Gemini 2.5
Flash integration using the official google-genai SDK.
```

---

## 3. Architecture & Design Patterns

To maintain a clean, maintainable, and predictable codebase, all implementations must follow these software engineering principles:

### A. Strategy Pattern for LLM Providers
- The core agent must never couple directly to a specific LLM vendor.
- Define a common interface (`BaseLLMProvider` protocol or abstract class).
- Providers (`GeminiProvider`, `OllamaProvider`, etc.) must be interchangeable plug-ins.

### B. Layered Architecture (Separation of Concerns)
- **Interface Layer**: CLI, terminal, or future API/WebSocket handlers. Handles only I/O.
- **Agent / Core Layer**: Orchestrates thinking, memory, and dispatching. Does not execute OS commands directly.
- **Tool / Action Layer**: Discrete, isolated functions with strict input/output schemas (type-hinted).
- **Infrastructure Layer**: Direct OS interactions, subprocesses, filesystem, and external network APIs.

### C. Dependency Inversion Principle (DIP)
- High-level modules must depend on abstractions (protocols/interfaces), not low-level concrete implementations.
- Inject dependencies (e.g., config, provider, file system handlers) into classes/functions instead of instantiating them globally.

### D. Single Responsibility Principle (SRP)
- Every file and class has one clear responsibility.
- No "god objects" or multi-hundred-line utility dumps. Keep modules small and cohesive.

### E. Explicit Over Implicit (No "Magic")
- Fail fast with meaningful exceptions.
- Do not catch generic exceptions (`except Exception: pass`). Log or bubble up errors cleanly.
- Strict type hinting across all Python signatures (`mypy` compliant).

### F. Clean Code & Zero Comment Noise
- Do **not** write trivial or redundant comments that merely narrate what the code visibly does (e.g., `# import libraries`, `# initialize client`, `# return response`).
- Code must be self-documenting through precise naming, type hints, and clean structure.
- Reserve comments strictly for non-obvious design choices, subtle workarounds, or domain constraints (*why*, never *what*).

---

## 4. Agent Operational Guardrails (Anti-Slop Rules)

1. **No Unsolicited Boilerplate**:
   - Never generate multiple unnecessary configuration files, decorative scripts, or speculative abstractions without explicit user request.
   - Do not add dependencies that have not been discussed and agreed upon.

2. **Step-by-Step Delivery**:
   - Propose the technical approach and file changes before writing them.
   - Deliver one focused, working feature at a time. Validate it before moving forward.

3. **No Fluff or Fake Content**:
   - Avoid decorative emojis, marketing prose, and filler text in code, commits, and documentation.
   - System prompts must be clear, concise, and operational—not flowery or verbose.

4. **Security & Financial Guardrails**:
   - Budget constraint: **$0 mandatory cost**. Always default to free tiers (e.g., Gemini Free Tier without credit card) or local offline inference.
   - Never hardcode credentials. Enforce strict `.env` usage excluded by `.gitignore`.
   - Never execute destructive host actions (`rm -rf`, modifying system configurations, killing arbitrary processes) without explicit confirmation.

---

## 5. Execution & Verification Standards

1. **Subprocess & OS Execution**:
   - All system commands executed by tools must enforce explicit timeouts and capture stdout/stderr cleanly.
   - Always validate process return codes. Never fail silently.

2. **Verification Before Delivery**:
   - Every newly created component or tool must be testable via automated tests (`pytest`) or an explicit verification step before completion.
