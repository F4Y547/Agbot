"""Headless smoke tests for the Agbot TUI (Textual test pilot).

Usage:
    python dev/tui_smoke.py           # fast: boot, /help, /tools, approval modal
    python dev/tui_smoke.py --live    # also run one real model turn (needs Ollama)
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textual.widgets import Input, Static  # noqa: E402

from agbot.config import Config  # noqa: E402
from agbot.llm import LlmClient  # noqa: E402
from agbot.tui import AgbotApp, ApprovalModal  # noqa: E402


def static_text(widget: Static) -> str:
    for attr in ("content", "renderable", "_Static__content"):
        value = getattr(widget, attr, None)
        if value is not None:
            return str(value)
    return ""


async def wait_for(predicate, pilot, timeout: float = 5.0) -> bool:
    import time as _time

    deadline = _time.time() + timeout
    while _time.time() < deadline:
        await pilot.pause(0.05)
        if predicate():
            return True
    return False


def make_app(yolo: bool = False) -> AgbotApp:
    cfg = Config.from_env()
    cfg.approval = not yolo
    cfg.show_reasoning = False
    client = LlmClient(cfg, is_ollama=True)
    return AgbotApp(cfg, client, True, [])


async def test_boot_help_exit() -> None:
    app = make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notes = [static_text(w) for w in app.query(".note")]
        assert any("endpoint" in n for n in notes), f"banner missing: {notes}"
        inp = app.query_one("#prompt", Input)
        inp.value = "/help"
        await pilot.press("enter")
        await pilot.pause()
        notes = [static_text(w) for w in app.query(".note")]
        assert any("/effort" in n for n in notes), f"/help output missing: {notes[-3:]}"
        inp.value = "/tools"
        await pilot.press("enter")
        await pilot.pause()
        notes = [static_text(w) for w in app.query(".note")]
        assert any("list_dir" in n for n in notes), "tools list missing"
        inp.value = "/exit"
        await pilot.press("enter")
    print("PASS  boot + banner + /help + /tools + /exit")


async def test_approval_modal() -> None:
    app = make_app()
    results: dict[str, str] = {}
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(
            ApprovalModal("run_shell", "echo hi"),
            lambda value: results.__setitem__("answer", value or ""),
        )
        assert await wait_for(lambda: isinstance(app.screen, ApprovalModal), pilot), (
            "modal did not open"
        )
        await pilot.press("a")
        assert await wait_for(lambda: results.get("answer") == "a", pilot), (
            f"modal key answer = {results}"
        )
        app.push_screen(
            ApprovalModal("write_file", "path: notes.txt"),
            lambda value: results.__setitem__("answer2", value or ""),
        )
        assert await wait_for(lambda: isinstance(app.screen, ApprovalModal), pilot), (
            "second modal did not open"
        )
        app.screen.query_one("#btn-n").press()
        assert await wait_for(lambda: results.get("answer2") == "n", pilot), (
            f"modal button answer = {results}"
        )
    print("PASS  approval modal (keys + buttons)")


async def test_live_turn() -> None:
    app = make_app(yolo=True)
    ok, err = app.client.probe()
    assert ok, f"Ollama not reachable: {err}"
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        inp = app.query_one("#prompt", Input)
        inp.value = "Reply with exactly: TUI-OK"
        await pilot.press("enter")
        deadline = time.time() + 600
        while time.time() < deadline:
            await pilot.pause(0.5)
            if not app._busy:
                break
        assert not app._busy, "turn did not finish in time"
        assistant = [m for m in app.messages if m.get("role") == "assistant" and m.get("content")]
        assert assistant, f"no assistant message: {app.messages[-2:]}"
        text = str(assistant[-1]["content"])
        assert "tui-ok" in text.lower(), f"unexpected answer: {text!r}"
        assert not inp.disabled, "input not re-enabled after turn"
    print(f"PASS  live turn against Ollama: {text.strip()[:60]!r}")


def main() -> int:
    live = "--live" in sys.argv
    asyncio.run(test_boot_help_exit())
    asyncio.run(test_approval_modal())
    if live:
        asyncio.run(test_live_turn())
    print("TUI SMOKE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
