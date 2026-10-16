"""Run one real TUI exchange (against Ollama) and save an SVG screenshot to docs/tui.svg."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textual.widgets import Input  # noqa: E402

from agbot.config import Config  # noqa: E402
from agbot.llm import LlmClient  # noqa: E402
from agbot.tui import AgbotApp  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "tui.svg"
PROMPT = (
    "Call the list_dir tool on the current directory, then reply with one short "
    "sentence about what you saw, ending with TUI-DEMO."
)


async def main() -> int:
    cfg = Config.from_env()
    cfg.approval = False          # no modal for the screenshot
    cfg.show_reasoning = True     # show the thinking block
    client = LlmClient(cfg, is_ollama=True)
    ok, err = client.probe()
    assert ok, f"Ollama not reachable: {err}"

    app = AgbotApp(cfg, client, True, [])
    async with app.run_test(size=(110, 34)) as pilot:
        await pilot.pause()
        inp = app.query_one("#prompt", Input)
        inp.value = PROMPT
        await pilot.press("enter")
        deadline = time.time() + 900
        while time.time() < deadline:
            await pilot.pause(0.5)
            if not app._busy:
                break
        assert not app._busy, "turn did not finish in time"
        await pilot.pause(0.3)
        svg = app.export_screenshot(title="Agbot TUI")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(svg, encoding="utf-8")
        turns = [m for m in app.messages if m.get("role") == "assistant" and m.get("content")]
        print(f"screenshot: {OUT} ({len(svg)} bytes)")
        if turns:
            print("last answer:", str(turns[-1]["content"]).strip()[:100])
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
