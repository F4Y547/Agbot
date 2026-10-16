from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines; never overrides existing env vars."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val


def _str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v.strip()


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    v = v.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return default


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


@dataclass
class Config:
    # --- endpoint ---
    base_url: str = ""                       # AGENT_BASE_URL
    api_key: str = "EMPTY"                   # AGENT_API_KEY
    model: str = "deepseek-v4.1-flash-crack"  # AGENT_MODEL
    # --- generation ---
    reasoning_effort: str = "low"            # AGENT_REASONING_EFFORT
    reasoning_effort_send: str = "auto"      # AGENT_SEND_REASONING_EFFORT: auto|on|off
    max_tokens: int = 32768                  # AGENT_MAX_TOKENS
    temperature: float = 1.0                 # AGENT_TEMPERATURE
    top_p: float = 0.95                      # AGENT_TOP_P
    # --- behavior ---
    max_steps: int = 12                      # AGENT_MAX_STEPS
    approval: bool = True                    # AGENT_APPROVAL: on|off
    show_reasoning: bool = True              # AGENT_SHOW_REASONING: on|off
    context_budget_chars: int = 600_000      # AGENT_CONTEXT_BUDGET_CHARS
    shell: str = ""                          # AGENT_SHELL (auto-detected when empty)
    # --- workspace / sessions ---
    workspace: Path = field(default_factory=Path.cwd)
    session_dir: Path | None = None
    # --- local fallback ---
    fallback_ollama: bool = True             # AGENT_FALLBACK_OLLAMA: on|off
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:8b"

    def send_reasoning_effort_for(self, is_ollama: bool) -> bool:
        mode = (self.reasoning_effort_send or "auto").lower()
        wished = self.reasoning_effort.strip().lower() not in ("", "off", "none", "disabled")
        if mode == "on":
            return wished
        if mode == "off":
            return False
        return wished and not is_ollama

    def resolved_session_dir(self) -> Path:
        d = self.session_dir or (ROOT / "sessions")
        d = Path(d)
        return d if d.is_absolute() else (ROOT / d)

    def resolved_workspace(self) -> Path:
        w = Path(self.workspace)
        return w if w.is_absolute() else (ROOT / w)

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "Config":
        load_dotenv(Path(env_file) if env_file else ROOT / ".env")
        sess = _str("AGENT_SESSION_DIR", "")
        ws = _str("AGENT_WORKSPACE", "")
        return cls(
            base_url=_str("AGENT_BASE_URL"),
            api_key=_str("AGENT_API_KEY", "EMPTY"),
            model=_str("AGENT_MODEL", "deepseek-v4.1-flash-crack"),
            reasoning_effort=_str("AGENT_REASONING_EFFORT", "low"),
            reasoning_effort_send=_str("AGENT_SEND_REASONING_EFFORT", "auto"),
            max_tokens=_int("AGENT_MAX_TOKENS", 32768),
            temperature=_float("AGENT_TEMPERATURE", 1.0),
            top_p=_float("AGENT_TOP_P", 0.95),
            max_steps=_int("AGENT_MAX_STEPS", 12),
            approval=_bool("AGENT_APPROVAL", True),
            show_reasoning=_bool("AGENT_SHOW_REASONING", True),
            context_budget_chars=_int("AGENT_CONTEXT_BUDGET_CHARS", 600_000),
            shell=_str("AGENT_SHELL"),
            workspace=Path(ws) if ws else Path.cwd(),
            session_dir=Path(sess) if sess else None,
            fallback_ollama=_bool("AGENT_FALLBACK_OLLAMA", True),
            ollama_url=_str("AGENT_OLLAMA_URL", "http://127.0.0.1:11434"),
            ollama_model=_str("AGENT_OLLAMA_MODEL", "qwen3:8b"),
        )
