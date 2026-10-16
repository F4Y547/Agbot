from __future__ import annotations

import html
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories. Paths are relative to the workspace unless absolute.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to list (default '.')"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file with line numbers. Returns up to `limit` lines starting at `offset` (1-based).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to read"},
                    "offset": {"type": "integer", "description": "First line number to return (default 1)"},
                    "limit": {"type": "integer", "description": "Max lines to return (default 400, max 2000)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content. Requires user approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to write"},
                    "content": {"type": "string", "description": "Full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "Replace an exact string in a file. old_string must occur exactly once. Requires user approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to edit"},
                    "old_string": {"type": "string", "description": "Exact text to find (must be unique)"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a non-interactive shell command in the workspace and capture output. Requires user approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command line to execute"},
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Kill the command after this many seconds (default 60, max 600)",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a URL over HTTP(S) and return readable text (HTML stripped, truncated).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_chars": {"type": "integer", "description": "Max characters to return (default 20000)"},
                },
                "required": ["url"],
            },
        },
    },
]

GATED_TOOLS = {"run_shell", "write_file", "replace_in_file"}

_MAX_OUTPUT = 24_000


def _truncate(text: str, limit: int = _MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.65)
    tail = limit - head
    return f"{text[:head]}\n...[truncated {len(text) - limit} chars]...\n{text[-tail:]}"


class ToolRunner:
    def __init__(
        self,
        workspace: Path,
        shell: str = "",
        abort_check: Callable[[], bool] | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.shell = shell
        self.abort_check = abort_check

    # ------------------------------------------------------------------ utils
    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (self.workspace / path)

    def is_gated(self, name: str) -> bool:
        return name in GATED_TOOLS

    def describe(self, name: str, args: dict) -> str:
        if name == "run_shell":
            return f"command:\n  {args.get('command', '')}\n\ncwd: {self.workspace}"
        if name == "write_file":
            content = args.get("content", "") or ""
            preview = content[:600] + ("\n…" if len(content) > 600 else "")
            return f"write {args.get('path', '')}  ({len(content)} chars)\n\n{preview}"
        if name == "replace_in_file":
            old = (args.get("old_string") or "")[:400]
            new = (args.get("new_string") or "")[:400]
            return f"edit {args.get('path', '')}\n\n--- old ---\n{old}\n\n--- new ---\n{new}"
        return json.dumps(args, ensure_ascii=False)[:800]

    # ------------------------------------------------------------------- run
    def run(self, name: str, args: dict) -> str:
        try:
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                return f"ERROR: unknown tool '{name}'. Available: list_dir, read_file, write_file, replace_in_file, run_shell, web_fetch"
            return _truncate(handler(args))
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"

    # ---------------------------------------------------------------- tools
    def _tool_list_dir(self, args: dict) -> str:
        path = self._resolve(args.get("path") or ".")
        if not path.is_dir():
            return f"ERROR: '{path}' is not a directory"
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = [f"{path}:"]
        for p in entries[:300]:
            if p.is_dir():
                lines.append(f"  {p.name}/")
            else:
                try:
                    size = p.stat().st_size
                except OSError:
                    size = -1
                lines.append(f"  {p.name}  ({size} bytes)")
        if len(entries) > 300:
            lines.append(f"  … {len(entries) - 300} more entries")
        return "\n".join(lines)

    def _tool_read_file(self, args: dict) -> str:
        path = self._resolve(args.get("path") or "")
        if not path.is_file():
            return f"ERROR: file not found: {path}"
        offset = max(1, int(args.get("offset") or 1))
        limit = min(2000, max(1, int(args.get("limit") or 400)))
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
        chunk = raw[offset - 1: offset - 1 + limit]
        width = len(str(offset + len(chunk)))
        body = "\n".join(f"{offset + i:>{width}}: {ln}" for i, ln in enumerate(chunk))
        head = f"{path}  (lines {offset}-{offset + len(chunk) - 1} of {len(raw)})"
        return f"{head}\n{body}"

    def _tool_write_file(self, args: dict) -> str:
        path = self._resolve(args.get("path") or "")
        content = args.get("content")
        if not isinstance(content, str):
            return "ERROR: 'content' must be a string"
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
        path.write_text(content, encoding="utf-8", newline="")
        verb = "overwrote" if existing is not None else "created"
        return f"OK: {verb} {path} ({len(content)} chars)"

    def _tool_replace_in_file(self, args: dict) -> str:
        path = self._resolve(args.get("path") or "")
        old = args.get("old_string")
        new = args.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str):
            return "ERROR: 'old_string' and 'new_string' must be strings"
        if not path.is_file():
            return f"ERROR: file not found: {path}"
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            return "ERROR: old_string not found in file"
        if count > 1:
            return f"ERROR: old_string occurs {count} times; make it unique (include more context)"
        path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="")
        return f"OK: edited {path}"

    def _tool_run_shell(self, args: dict) -> str:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return "ERROR: empty command"
        timeout = int(args.get("timeout_seconds") or 60)
        timeout = max(1, min(600, timeout))
        if self.shell:
            argv = [self.shell, "-c", command] if os.name != "nt" else [self.shell, "-NoLogo", "-NoProfile", "-Command", command]
        elif os.name == "nt":
            argv = ["powershell", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command]
        else:
            argv = ["/bin/bash", "-lc", command]
        proc = subprocess.Popen(
            argv,
            cwd=str(self.workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        deadline = time.monotonic() + timeout
        out = err = ""
        while True:
            try:
                out, err = proc.communicate(timeout=0.2)  # drains pipes, avoids deadlocks
                break
            except subprocess.TimeoutExpired:
                if self.abort_check is not None and self.abort_check():
                    proc.kill()
                    proc.communicate()
                    return "ERROR: command aborted by user"
                if time.monotonic() > deadline:
                    proc.kill()
                    proc.communicate()
                    return f"ERROR: command timed out after {timeout}s"
        out = out or ""
        err = err or ""
        parts = [f"exit code: {proc.returncode}"]
        if out.strip():
            parts.append(f"stdout:\n{out}")
        if err.strip():
            parts.append(f"stderr:\n{err}")
        if not out.strip() and not err.strip():
            parts.append("(no output)")
        return "\n".join(parts)

    def _tool_web_fetch(self, args: dict) -> str:
        url = args.get("url") or ""
        if not re.match(r"^https?://", url):
            return "ERROR: only http(s) URLs are supported"
        max_chars = min(100_000, max(500, int(args.get("max_chars") or 20_000)))
        import httpx

        r = httpx.get(
            url,
            timeout=30,
            follow_redirects=True,
            headers={"User-Agent": "Agbot/0.1 (+terminal agent)"},
        )
        text = r.text
        ctype = r.headers.get("content-type", "")
        if "html" in ctype.lower() or text.lstrip()[:1] == "<":
            text = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", text)
            text = re.sub(r"(?s)<[^>]+>", " ", text)
            text = html.unescape(text)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()
        head = f"GET {url} -> {r.status_code} ({ctype or 'unknown'}, {len(text)} chars)\n\n"
        return head + text[:max_chars] + ("\n...[truncated]" if len(text) > max_chars else "")
