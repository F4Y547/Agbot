"""Full-screen terminal UI for Agbot, in the spirit of opencode:

- header with model/provider, scrolling message log, status line, input box
- streaming assistant Markdown, collapsible reasoning stream
- one card per tool call with live status and output preview
- modal approval dialog for gated tools (y / n / a / q)
- ctrl+c interrupts generation (and kills running shell commands), ctrl+q quits
"""

from __future__ import annotations

import json
import threading
import time

from rich.markup import escape as markup_escape
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Header, Input, Markdown, Static

from .config import Config
from .llm import LlmClient, StreamResult, ToolCall
from .loop import AgentQuit, system_prompt, trim_messages
from .session import Session
from .tools import TOOL_SCHEMAS, ToolRunner


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _keep_pinned(widget) -> None:
    """Scroll the message log to the bottom after a widget grows (best effort)."""
    try:
        parent = widget.parent
        if parent is not None:
            parent.scroll_end(animate=False)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------- widgets


class UserMessage(Static):
    def __init__(self, text: str) -> None:
        super().__init__("❯ " + text, markup=False, classes="user-msg")


class ReasoningBlock(Collapsible):
    def __init__(self) -> None:
        self._static = Static("", markup=False, classes="reasoning-text")
        self._length = 0
        super().__init__(self._static, title="thinking…", collapsed=False, classes="reasoning")

    def set_text(self, text: str) -> None:
        self._length = len(text)
        self._static.update(text)
        _keep_pinned(self)

    def finish(self) -> None:
        self.title = f"thinking · {self._length} chars (press enter on it to expand)"
        self.collapsed = True


class AssistantMessage(Vertical):
    def __init__(self) -> None:
        super().__init__(classes="assistant")
        self._md = Markdown("")
        self._stats = Static("", markup=False, classes="stats")

    def compose(self) -> ComposeResult:
        yield self._md
        yield self._stats

    def set_markdown(self, text: str) -> None:
        self._md.update(text)
        _keep_pinned(self)

    def set_stats(self, text: str) -> None:
        self._stats.update(text)
        _keep_pinned(self)


class ToolCallBlock(Vertical):
    def __init__(self, name: str, preview: str) -> None:
        super().__init__(classes="tool")
        self._name = name
        self._preview = preview
        self._head = Static(self._head_text("⚙", None))
        self._body = Static("running…", markup=False, classes="tool-body")

    def _head_text(self, glyph: str, seconds: float | None) -> str:
        color = {"✓": "green", "✗": "red"}.get(glyph, "yellow")
        tail = f"  [dim]{markup_escape(self._preview)}[/dim]" if self._preview else ""
        timing = f"  [dim]{seconds:.1f}s[/dim]" if seconds is not None else ""
        return f"[{color}]{glyph}[/{color}] [bold]{self._name}[/bold]{tail}{timing}"

    def compose(self) -> ComposeResult:
        yield self._head
        yield self._body

    def set_result(self, text: str, seconds: float, ok: bool) -> None:
        self._head.update(self._head_text("✓" if ok else "✗", seconds))
        body = text if len(text) <= 4000 else text[:4000] + f"\n… (+{len(text) - 4000} chars — full output in session log)"
        self._body.update(body)
        _keep_pinned(self)


# ------------------------------------------------------------------------ modal


class ApprovalModal(ModalScreen[str]):
    BINDINGS = [
        Binding("y", "answer('y')", "yes"),
        Binding("n", "answer('n')", "no"),
        Binding("a", "answer('a')", "always"),
        Binding("q", "answer('q')", "quit"),
        Binding("escape", "answer('n')", "cancel", show=False),
        Binding("ctrl+c", "answer('q')", "quit", show=False, priority=True),
    ]

    def __init__(self, tool_name: str, detail: str) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._detail = detail

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Static(f"Approve tool: {self._tool_name}", id="modal-title")
            yield Static(self._detail, markup=False, id="modal-detail")
            with Horizontal(id="modal-buttons"):
                yield Button("Yes (y)", id="btn-y", variant="success")
                yield Button("No (n)", id="btn-n", variant="error")
                yield Button("Always (a)", id="btn-a", variant="primary")
                yield Button("Quit (q)", id="btn-q", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id:
            self.dismiss(event.button.id.removeprefix("btn-"))

    def action_answer(self, value: str) -> None:
        self.dismiss(value)


# -------------------------------------------------------------------------- app


class AgbotApp(App):
    CSS = """
    #log { height: 1fr; padding: 0 1; }
    #status { height: 1; color: $text-muted; background: $panel; padding: 0 1; }
    #prompt { height: 3; border: round $accent; margin: 0 1 1 1; }
    #prompt:focus { border: double $accent; }
    .user-msg { background: $boost; padding: 0 1; margin: 1 0 0 0; }
    .assistant { margin: 0 0 1 0; }
    .stats { color: $text-muted; }
    .note { color: $text-muted; }
    .reasoning { margin: 0 0 1 0; }
    .reasoning-text { color: $text-muted; }
    .tool { border-left: thick $warning; background: $surface; padding: 0 1; margin: 0 0 1 0; }
    .tool-body { color: $text-muted; }
    ApprovalModal { align: center middle; }
    #modal-box { width: 90%; max-width: 110; height: auto; max-height: 80%; border: thick $warning; background: $surface; padding: 1 2; }
    #modal-title { text-style: bold; }
    #modal-detail { max-height: 18; overflow-y: auto; color: $text-muted; padding: 1 0; }
    #modal-buttons { height: auto; align-horizontal: right; }
    #modal-buttons Button { margin-left: 1; min-width: 9; }
    """

    TITLE = "Agbot"
    BINDINGS = [
        Binding("ctrl+c", "interrupt_or_quit", "stop / quit", priority=True, show=False),
        Binding("ctrl+q", "quit_now", "quit", priority=True, show=False),
    ]

    def __init__(
        self,
        cfg: Config,
        client: LlmClient,
        is_ollama: bool,
        resume_messages: list[dict] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.client = client
        self.is_ollama = is_ollama
        self.session = Session(cfg.resolved_session_dir(), "tui")
        self.session.log("endpoint", base_url=client.base_url, model=client.model, ui="tui")
        self.tools = ToolRunner(
            cfg.resolved_workspace(), cfg.shell, abort_check=lambda: self._abort.is_set()
        )
        self.messages: list[dict] = [{"role": "system", "content": system_prompt(cfg)}]
        self.approve_all = not cfg.approval
        self._abort = threading.Event()
        self._busy = False
        self._state = "ready"
        self._recent_calls: list[str] = []
        self._resume = list(resume_messages or [])
        self.sub_title = f"{client.model} · {'Ollama' if is_ollama else 'SGLang'}"

    # ------------------------------------------------------------- scaffolding
    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="log")
        yield Static("", id="status")
        yield Input(placeholder="Ask Agbot to do something…   (/help · /exit)", id="prompt")

    def _log(self) -> VerticalScroll:
        return self.query_one("#log", VerticalScroll)

    def _input(self) -> Input:
        return self.query_one("#prompt", Input)

    def _scroll_end(self) -> None:
        self._log().scroll_end(animate=False)

    def _note(self, text: str) -> None:
        self._log().mount(Static(text, markup=False, classes="note"))
        self._scroll_end()

    def _update_status(self) -> None:
        color = "green" if self._state == "ready" else "yellow"
        hints = "enter send · ctrl+c stop" if self._busy else "enter send · /help · ctrl+c quit"
        self.query_one("#status", Static).update(
            f"[{color}]●[/{color}] {markup_escape(self._state)}  ·  "
            f"{markup_escape(self.client.model)}  ·  {hints}"
        )

    def _set_state(self, state: str) -> None:
        self._state = state
        self._update_status()

    def on_mount(self) -> None:
        provider = "Ollama fallback" if self.is_ollama else "SGLang (remote)"
        self._note(f"provider   {provider}")
        self._note(f"endpoint   {self.client.base_url}")
        self._note(f"model      {self.client.model}")
        approvals = "auto (yolo)" if not self.approval_prompting() else "y/n/a/q before shell + file writes"
        self._note(f"approvals  {approvals}")
        for msg in self._resume:
            content = str(msg.get("content") or "")
            if not content:
                continue
            if msg.get("role") == "user":
                self._log().mount(UserMessage(content))
            elif msg.get("role") == "assistant":
                bubble = AssistantMessage()
                bubble.set_markdown(content)
                self._log().mount(bubble)
        if self._resume:
            self._note("— resumed previous session —")
        self._scroll_end()
        self._update_status()
        self._input().focus()
        self.set_interval(0.3, self._pin_scroll)

    def _pin_scroll(self) -> None:
        """Keep the log pinned to the bottom while a turn is streaming."""
        if self._busy:
            self._log().scroll_end(animate=False)

    def approval_prompting(self) -> bool:
        return not self.approve_all

    # ------------------------------------------------------------------ input
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = (event.value or "").strip()
        event.input.value = ""
        if not text:
            return
        if self._busy:
            self._note("still working — press ctrl+c to interrupt first")
            return
        if text.startswith("/"):
            self._slash(text)
            return
        self._start_turn(text)

    def _slash(self, text: str) -> None:
        cmd, _, rest = text.partition(" ")
        cmd = cmd.lower()
        rest = rest.strip()
        if cmd in ("/exit", "/quit"):
            self.exit()
        elif cmd == "/help":
            self._note(
                "commands: /help · /reset · /tools · /effort low|high|xhigh|max · "
                "/think on|off · /model NAME · /exit   (ctrl+c stops generation)"
            )
        elif cmd == "/reset":
            self.messages = [{"role": "system", "content": system_prompt(self.cfg)}]
            self._log().remove_children()
            self._note("history cleared")
        elif cmd == "/tools":
            for tool in TOOL_SCHEMAS:
                fn = tool["function"]
                self._note(f"{fn['name']}  —  {fn['description']}")
        elif cmd == "/effort":
            if rest:
                self.cfg.reasoning_effort = rest
            self._note(f"reasoning_effort = {self.cfg.reasoning_effort}")
        elif cmd == "/think":
            self.cfg.show_reasoning = rest.lower() in ("on", "1", "true", "yes")
            self._note(f"reasoning display = {'on' if self.cfg.show_reasoning else 'off'}")
        elif cmd == "/model":
            if rest:
                if self.is_ollama:
                    self.cfg.ollama_model = rest
                else:
                    self.cfg.model = rest
                self.client.model = rest
                self.sub_title = f"{rest} · {'Ollama' if self.is_ollama else 'SGLang'}"
                self._update_status()
            else:
                self._note(f"model = {self.client.model}")
        else:
            self._note(f"unknown command {cmd} — try /help")

    # ------------------------------------------------------------------- turn
    def _start_turn(self, text: str) -> None:
        self._busy = True
        self._abort.clear()
        self._recent_calls.clear()
        self._log().mount(UserMessage(text))
        self._scroll_end()
        msg = {"role": "user", "content": text}
        self.messages.append(msg)
        self.session.log_message(msg)
        self._set_state("thinking…")
        self._input().disabled = True
        self._turn_worker()

    @work(thread=True, exclusive=True)
    def _turn_worker(self) -> None:
        try:
            self._loop_turn()
        except AgentQuit:
            self.call_from_thread(self._note, "quit requested — ending session")
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self._note, f"unexpected error: {type(e).__name__}: {e}")
        finally:
            self.call_from_thread(self._turn_finished)

    def _turn_finished(self) -> None:
        self._busy = False
        self._abort.clear()
        inp = self._input()
        inp.disabled = False
        inp.focus()
        self._set_state("ready")
        self._log().scroll_end(animate=False)

    def _loop_turn(self) -> None:
        cfg = self.cfg
        for _ in range(cfg.max_steps):
            if self._abort.is_set():
                self.call_from_thread(self._note, "· interrupted")
                return
            if trim_messages(self.messages, cfg.context_budget_chars):
                self.call_from_thread(self._note, "· context trimmed to fit budget")

            result = self._stream_call()
            if result.aborted:
                self.call_from_thread(self._note, "· generation interrupted")
                return
            if not result.ok:
                self.call_from_thread(self._note, f"model error: {result.error}")
                return

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
                self.call_from_thread(
                    self._note, "server reported tools unsupported — answering without tools"
                )
                return
            if not result.tool_calls:
                return

            for tc in result.tool_calls:
                if self._abort.is_set():
                    self.call_from_thread(self._note, "· interrupted")
                    return
                text = self._execute_tool(tc)
                tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": text}
                self.messages.append(tool_msg)
                self.session.log_message(tool_msg)

        self.call_from_thread(
            self._note, f"· reached max steps ({cfg.max_steps}) without a final answer"
        )

    # -------------------------------------------------------------- streaming
    def _stream_call(self) -> StreamResult:
        cfg = self.cfg
        assistant_holder: list[AssistantMessage] = []
        reasoning_holder: list[ReasoningBlock] = []
        buf = {"content": "", "reasoning": "", "t_content": 0.0, "t_reasoning": 0.0}

        def flush_content(final: bool = False) -> None:
            if not buf["content"]:
                return
            if not assistant_holder:
                assistant_holder.append(self.call_from_thread(self._mount_assistant))
            if final or (time.monotonic() - buf["t_content"]) > 0.12:
                self.call_from_thread(assistant_holder[0].set_markdown, buf["content"])
                buf["t_content"] = time.monotonic()

        def flush_reasoning(final: bool = False) -> None:
            if not cfg.show_reasoning or not buf["reasoning"]:
                return
            if not reasoning_holder:
                reasoning_holder.append(self.call_from_thread(self._mount_reasoning))
            if final or (time.monotonic() - buf["t_reasoning"]) > 0.12:
                self.call_from_thread(reasoning_holder[0].set_text, buf["reasoning"])
                buf["t_reasoning"] = time.monotonic()

        def on_content(text: str) -> None:
            buf["content"] += text
            flush_content()

        def on_reasoning(text: str) -> None:
            buf["reasoning"] += text
            flush_reasoning()

        result = self.client.stream_chat(
            self.messages,
            TOOL_SCHEMAS,
            cfg,
            on_reasoning,
            on_content,
            should_abort=lambda: self._abort.is_set(),
        )
        flush_reasoning(final=True)
        flush_content(final=True)
        if reasoning_holder:
            self.call_from_thread(reasoning_holder[0].finish)
        if assistant_holder:
            self.call_from_thread(assistant_holder[0].set_stats, self._stats_line(result))
        return result

    @staticmethod
    def _stats_line(result: StreamResult) -> str:
        usage = result.usage or {}
        parts: list[str] = []
        if usage.get("prompt_tokens") is not None:
            parts.append(f"in {usage['prompt_tokens']}")
        if usage.get("completion_tokens") is not None:
            parts.append(f"out {usage['completion_tokens']}")
        parts.append(f"{result.elapsed_s:.1f}s")
        ct = usage.get("completion_tokens")
        if ct and result.elapsed_s:
            parts.append(f"{ct / result.elapsed_s:.0f} tok/s")
        return "— " + " · ".join(parts)

    # ------------------------------------------------------------------ tools
    def _execute_tool(self, tc: ToolCall) -> str:
        name = tc.name or "?"
        raw = tc.arguments or ""
        try:
            args = json.loads(raw) if raw.strip() else {}
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self._note, f"invalid tool arguments for {name}: {e}")
            return f"Invalid tool arguments ({e}). Raw: {raw[:500]}"

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
            answer = self._ask_approval(name, args)
            if answer == "q":
                raise AgentQuit()
            if answer == "a":
                self.approve_all = True
            if answer == "n":
                self.session.log("tool_denied", tool=name, arguments=args)
                return "User declined this tool call. Ask what to change or take another approach."
            self.session.log("tool_approved", tool=name, arguments=args)

        self.call_from_thread(self._set_state, f"running {name}…")
        preview = _clip(json.dumps(args, ensure_ascii=False), 140)
        block = self.call_from_thread(self._mount_tool, name, preview)
        t0 = time.monotonic()
        text = self.tools.run(name, args)
        seconds = time.monotonic() - t0
        ok = not text.startswith(("ERROR:", "Invalid tool"))
        self.session.log("tool_result", tool=name, seconds=round(seconds, 2), result=text[:4000])
        self.call_from_thread(block.set_result, text, seconds, ok)
        self.call_from_thread(self._set_state, "thinking…")
        return text

    def _ask_approval(self, name: str, args: dict) -> str:
        answer: dict[str, str] = {}
        done = threading.Event()

        def show() -> None:
            def callback(value: str | None) -> None:
                answer["value"] = value or "n"
                done.set()

            self.push_screen(ApprovalModal(name, self.tools.describe(name, args)), callback)

        self.call_from_thread(self._set_state, f"approval needed: {name}")
        self.call_from_thread(show)
        while not done.wait(0.2):
            if self._abort.is_set():
                self.call_from_thread(self._dismiss_modal)
                done.wait(2.0)
                return "q"
        return answer.get("value", "n")

    def _dismiss_modal(self) -> None:
        if isinstance(self.screen, ApprovalModal):
            self.screen.dismiss("q")

    # --------------------------------------------------------------- mounting
    def _mount_assistant(self) -> AssistantMessage:
        widget = AssistantMessage()
        self._log().mount(widget)
        self._scroll_end()
        return widget

    def _mount_reasoning(self) -> ReasoningBlock:
        widget = ReasoningBlock()
        self._log().mount(widget)
        self._scroll_end()
        return widget

    def _mount_tool(self, name: str, preview: str) -> ToolCallBlock:
        widget = ToolCallBlock(name, preview)
        self._log().mount(widget)
        self._scroll_end()
        return widget

    # -------------------------------------------------------------- shortcuts
    def action_interrupt_or_quit(self) -> None:
        if self._busy:
            if not self._abort.is_set():
                self._abort.set()
                self._set_state("interrupting…")
        else:
            self.exit()

    def action_quit_now(self) -> None:
        self.exit()

    def on_unmount(self) -> None:
        try:
            self.session.close()
        except Exception:  # noqa: BLE001
            pass
