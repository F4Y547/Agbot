from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import replace

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from .config import Config
from .llm import LlmClient, StreamResult, ToolCall
from .session import Session
from .tools import TOOL_SCHEMAS, ToolRunner

SYSTEM_PROMPT = """\
You are Agbot, a capable terminal agent running on {os_name} ({platform}) in the directory:
{workspace}
Today is {date}.

You accomplish tasks by calling tools. Available tools:
- list_dir(path)                     list files/directories
- read_file(path, offset, limit)     read a text file with line numbers
- write_file(path, content)          create or overwrite a file   [needs user approval]
- replace_in_file(path, old, new)    exact-string edit            [needs user approval]
- run_shell(command, timeout_seconds) run a shell command          [needs user approval]
- web_fetch(url, max_chars)          fetch a URL as readable text

Rules:
1. Inspect before changing: read files / list directories before editing or running things.
2. Work in small, verifiable steps. After changing code, run it or run tests to verify.
3. Never invent file contents or command output — call the tool.
4. Shell commands must be non-interactive (no editors, pagers, or prompts) and should bound their output.
5. When the task is complete, write a short final answer describing exactly what you did.
   Do not call tools in the final answer message.
"""


class AgentQuit(Exception):
    pass


def system_prompt(cfg: Config) -> str:
    return SYSTEM_PROMPT.format(
        os_name=platform.system(),
        platform=platform.platform(terse=True),
        workspace=str(cfg.resolved_workspace()),
        date=time.strftime("%Y-%m-%d"),
    )


def trim_messages(messages: list[dict], budget: int) -> bool:
    """Drop the oldest turns until the char budget fits. Returns True if anything was dropped."""

    def total() -> int:
        n = 0
        for m in messages:
            n += len(str(m.get("content") or ""))
            n += len(str(m.get("tool_calls") or ""))
        return n

    trimmed = False
    while total() > budget and len(messages) > 2:
        dropped = messages.pop(1)
        if dropped.get("tool_calls"):
            while len(messages) > 1 and messages[1].get("role") == "tool":
                messages.pop(1)
        if len(messages) > 1 and messages[1].get("role") == "tool":
            messages.pop(1)
        trimmed = True
    return trimmed


class Agent:
    def __init__(self, cfg: Config, console: Console, client: LlmClient, session: Session) -> None:
        self.cfg = cfg
        self.console = console
        self.client = client
        self.session = session
        self.tools = ToolRunner(cfg.resolved_workspace(), cfg.shell)
        self.approve_all = not cfg.approval
        self._recent_calls: list[str] = []
        self.messages: list[dict] = [{"role": "system", "content": self._system_prompt()}]

    # ------------------------------------------------------------------ setup
    def _system_prompt(self) -> str:
        return system_prompt(self.cfg)

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self._system_prompt()}]
        self._recent_calls.clear()

    def load_history(self, messages: list[dict]) -> None:
        for m in messages:
            if m.get("role") != "system":
                self.messages.append(m)
        self._trim()

    # --------------------------------------------------------------- one turn
    def run(self, user_text: str) -> None:
        user_msg = {"role": "user", "content": user_text}
        self.messages.append(user_msg)
        self.session.log_message(user_msg)

        for step in range(1, self.cfg.max_steps + 1):
            self._trim()
            result = self._stream()

            if not result.ok:
                self.console.print(f"[red]model error:[/red] {result.error}")
                return
            self._print_stats(result)

            assistant: dict = {"role": "assistant", "content": result.content or ""}
            if result.tool_calls:
                assistant["content"] = result.content or None
                assistant["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments or "{}"},
                    }
                    for tc in result.tool_calls
                ]
            self.messages.append(assistant)
            self.session.log_message(assistant)

            if result.tools_unsupported:
                self.console.print(
                    "[yellow]server/model reported that tools are unsupported; "
                    "answering without tools[/yellow]"
                )
                return
            if not result.tool_calls:
                return  # final answer

            for tc in result.tool_calls:
                text = self._execute(tc)
                tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": text}
                self.messages.append(tool_msg)
                self.session.log_message(tool_msg)

        self.console.print(
            f"[yellow]reached max steps ({self.cfg.max_steps}) without a final answer[/yellow]"
        )

    # ------------------------------------------------------------- streaming
    def _stream(self) -> StreamResult:
        console = self.console
        state = {"reasoning": False, "content": False}

        def on_reasoning(text: str) -> None:
            if not self.cfg.show_reasoning:
                return
            if not state["reasoning"]:
                console.print()
                console.print("[dim]🧠 reasoning…[/dim]", end=" ")
                state["reasoning"] = True
            console.print(text, end="", style="dim italic", markup=False, highlight=False)

        def on_content(text: str) -> None:
            if not state["content"]:
                if state["reasoning"]:
                    console.print()
                state["content"] = True
            console.print(text, end="", markup=False, highlight=False)

        result = self.client.stream_chat(
            self.messages, TOOL_SCHEMAS, self.cfg, on_reasoning, on_content
        )
        if state["content"] or state["reasoning"]:
            console.print()
        return result

    def _print_stats(self, result: StreamResult) -> None:
        usage = result.usage or {}
        parts: list[str] = []
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        if pt is not None:
            parts.append(f"in {pt}")
        if ct is not None:
            parts.append(f"out {ct}")
        parts.append(f"{result.elapsed_s:.1f}s")
        if ct and result.elapsed_s:
            parts.append(f"{ct / result.elapsed_s:.0f} tok/s")
        self.console.print(f"[dim]— {', '.join(parts)}[/dim]")

    # ----------------------------------------------------------- tool exec
    def _execute(self, tc: ToolCall) -> str:
        name = tc.name or ""
        raw_args = tc.arguments or ""
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except Exception as e:  # noqa: BLE001
            text = f"Invalid tool arguments ({e}). Raw: {raw_args[:500]}"
            self.console.print(f"[red]✗ {name}: invalid arguments[/red]")
            return text

        signature = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
        self._recent_calls.append(signature)
        self._recent_calls = self._recent_calls[-3:]
        if len(self._recent_calls) == 3 and len(set(self._recent_calls)) == 1:
            self._recent_calls.clear()
            return (
                "ERROR: the same tool call was repeated 3 times. Stop repeating it - "
                "summarize what you have or take a different approach."
            )

        if self.tools.is_gated(name) and not self.approve_all:
            self.console.print(
                Panel(
                    Text(self.tools.describe(name, args)),
                    title=f"approve: {name}",
                    border_style="yellow",
                )
            )
            try:
                answer = Prompt.ask(
                    "[bold yellow]run this?[/bold yellow]",
                    choices=["y", "n", "a", "q"],
                    default="y",
                    console=self.console,
                )
            except (EOFError, KeyboardInterrupt):
                raise AgentQuit()
            if answer == "q":
                raise AgentQuit()
            if answer == "a":
                self.approve_all = True
            if answer == "n":
                self.session.log("tool_denied", tool=name, arguments=args)
                return "User declined this tool call. Ask what to change or take another approach."
            self.session.log("tool_approved", tool=name, arguments=args)

        short = json.dumps(args, ensure_ascii=False)
        if len(short) > 120:
            short = short[:120] + "…"
        self.console.print(f"[bold cyan]⚙ {name}[/bold cyan] [dim]{short}[/dim]")

        t0 = time.monotonic()
        text = self.tools.run(name, args)
        seconds = time.monotonic() - t0
        self.session.log("tool_result", tool=name, seconds=round(seconds, 2), result=text[:4000])

        preview = text if len(text) <= 1400 else text[:1400] + f" …(+{len(text) - 1400} chars)"
        self.console.print(
            Panel(Text(preview), title=f"{name} · {seconds:.1f}s", border_style="dim")
        )
        return text

    # ---------------------------------------------------------- context trim
    def _trim(self) -> None:
        if trim_messages(self.messages, self.cfg.context_budget_chars):
            self.console.print("[dim]· context trimmed to fit budget[/dim]")


def make_selftest_config(cfg: Config) -> Config:
    return replace(cfg, approval=False, max_steps=4, show_reasoning=False, context_budget_chars=max(50_000, cfg.context_budget_chars // 4))
