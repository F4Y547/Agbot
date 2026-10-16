from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .config import Config, ROOT
from .llm import LlmClient
from .loop import Agent, AgentQuit, make_selftest_config
from .session import Session, latest_session_file, load_messages


def _fix_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="agbot",
        description="Terminal agent for an OpenAI-compatible LLM endpoint "
        "(remote SGLang running DeepSeek-V4.1-Flash-Uncensored, or local Ollama).",
    )
    p.add_argument("prompt", nargs="?", default=None, help="One-shot prompt; omit for interactive mode")
    p.add_argument("--url", help="Base URL, e.g. http://127.0.0.1:30001/v1 (overrides AGENT_BASE_URL)")
    p.add_argument("--model", help="Model name to request")
    p.add_argument("--effort", help="reasoning_effort: low|high|xhigh|max (overrides AGENT_REASONING_EFFORT)")
    p.add_argument("--no-think", action="store_true", help="Do not display the reasoning stream")
    p.add_argument("--yolo", action="store_true", help="Auto-approve all tool calls")
    p.add_argument("--max-steps", type=int, help="Max tool rounds per user message")
    p.add_argument("--workspace", help="Workspace directory for tools (default: current dir)")
    p.add_argument("--session-dir", help="Where session JSONL logs are written")
    p.add_argument("--env", help="Path to a .env file (default: repo root .env)")
    p.add_argument("--resume", action="store_true", help="Resume the most recent session's history")
    p.add_argument("--selftest", action="store_true", help="Run a scripted tool-call smoke test and exit")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    env_path: str | Path | None = args.env
    if not env_path:
        local = Path.cwd() / ".env"
        if local.is_file() and local.resolve() != (ROOT / ".env").resolve():
            env_path = local
    cfg = Config.from_env(env_file=env_path)
    if args.url:
        cfg.base_url = args.url
    if args.model:
        cfg.model = args.model
        cfg.ollama_model = args.model
    if args.effort:
        cfg.reasoning_effort = args.effort
    if args.no_think:
        cfg.show_reasoning = False
    if args.yolo:
        cfg.approval = False
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.workspace:
        cfg.workspace = Path(args.workspace)
    if args.session_dir:
        cfg.session_dir = Path(args.session_dir)
    return cfg


def resolve_target(cfg: Config, console: Console) -> tuple[LlmClient, bool]:
    """Return (client, is_ollama). Prefers the remote endpoint, falls back to Ollama."""
    if cfg.base_url:
        remote = LlmClient(cfg, is_ollama=False)
        ok, err = remote.probe()
        if ok:
            return remote, False
        console.print(f"[yellow]remote endpoint {cfg.base_url} unreachable ({err})[/yellow]")
        if not cfg.fallback_ollama:
            console.print("[red]AGENT_FALLBACK_OLLAMA is off — cannot continue[/red]")
            raise SystemExit(2)
        console.print("[yellow]falling back to local Ollama[/yellow]")

    local = LlmClient(cfg, is_ollama=True)
    ok, err = local.probe()
    if ok:
        return local, True

    console.print(
        f"[red]No model endpoint available.[/red]\n"
        f"  remote (AGENT_BASE_URL): {cfg.base_url or 'not set'}\n"
        f"  ollama ({cfg.ollama_url}): {err}\n\n"
        "Start Ollama ('ollama serve') or set AGENT_BASE_URL to the GPU box "
        "(see deploy/README.md)."
    )
    raise SystemExit(2)


def banner(cfg: Config, client: LlmClient, is_ollama: bool, console: Console) -> None:
    label = "Ollama (local fallback)" if is_ollama else "SGLang (remote)"
    effort = (
        cfg.reasoning_effort
        if cfg.send_reasoning_effort_for(is_ollama)
        else f"{cfg.reasoning_effort} (not sent)"
    )
    lines = [
        f"provider   {label}",
        f"endpoint   {client.base_url}",
        f"model      {client.model}",
        f"effort     {effort}",
        f"thinking   {'shown' if cfg.show_reasoning else 'hidden'}",
        f"approvals  {'auto (--yolo)' if not cfg.approval else 'ask before shell/write'}",
        f"workspace  {cfg.resolved_workspace()}",
        f"sessions   {cfg.resolved_session_dir()}",
    ]
    console.print(Panel(Text("\n".join(lines)), title="Agbot", border_style="cyan"))


def run_selftest(cfg: Config, client: LlmClient, console: Console) -> int:
    st_cfg = make_selftest_config(cfg)
    session = Session(cfg.resolved_session_dir(), "selftest")
    agent = Agent(st_cfg, console, client, session)
    console.print(
        f"[bold]Selftest[/bold] · {client.base_url} · model={client.model} "
        "(asks the model to call list_dir, then answer)"
    )
    agent.run(
        "Call the list_dir tool on the current directory. Then reply with one short "
        "sentence about what you saw, ending with SELFTEST-OK."
    )
    used_tools = any(m.get("role") == "tool" for m in agent.messages)
    assistant_turns = sum(1 for m in agent.messages if m.get("role") == "assistant")
    ok = used_tools and assistant_turns >= 2
    session.log("selftest", ok=ok, used_tools=used_tools, assistant_turns=assistant_turns)
    session.close()
    if ok:
        console.print("[bold green]SELFTEST PASS[/bold green] — tool loop works against this endpoint.")
        return 0
    console.print(
        "[bold red]SELFTEST FAIL[/bold red] — the model did not complete a tool round-trip. "
        "Check that the model supports tool calling."
    )
    return 1


def repl(cfg: Config, client: LlmClient, console: Console, resume: bool) -> None:
    slug = "session"
    session_dir = cfg.resolved_session_dir()
    prior: list[dict] = []
    if resume:
        latest = latest_session_file(session_dir)
        if latest:
            prior = load_messages(latest)
            slug = "resumed"
            console.print(f"[dim]resuming {latest} ({len(prior)} messages)[/dim]")

    session = Session(session_dir, slug)
    session.log("endpoint", base_url=client.base_url, model=client.model)
    agent = Agent(cfg, console, client, session)
    if prior:
        agent.load_history(prior)

    console.print("[dim]/help for commands · /exit to quit[/dim]")
    try:
        while True:
            try:
                line = console.input("[bold green]you ❯[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line in ("/exit", "/quit"):
                break
            if line == "/help":
                console.print(
                    "  /exit              quit\n"
                    "  /reset             clear conversation history\n"
                    "  /tools             list tools\n"
                    "  /effort <tier>     low|high|xhigh|max\n"
                    "  /think on|off      show/hide reasoning stream\n"
                    "  /model <name>      switch model name\n"
                )
                continue
            if line == "/reset":
                agent.reset()
                console.print("[dim]history cleared[/dim]")
                continue
            if line == "/tools":
                from .tools import TOOL_SCHEMAS

                for t in TOOL_SCHEMAS:
                    fn = t["function"]
                    console.print(f"  [cyan]{fn['name']}[/cyan]  {fn['description']}")
                continue
            if line.startswith("/effort "):
                cfg.reasoning_effort = line.split(None, 1)[1].strip()
                console.print(f"[dim]reasoning_effort = {cfg.reasoning_effort}[/dim]")
                continue
            if line.startswith("/think "):
                value = line.split(None, 1)[1].strip().lower()
                cfg.show_reasoning = value in ("on", "1", "true", "yes")
                console.print(f"[dim]reasoning display = {'on' if cfg.show_reasoning else 'off'}[/dim]")
                continue
            if line.startswith("/model "):
                name = line.split(None, 1)[1].strip()
                if client.is_ollama:
                    cfg.ollama_model = name
                else:
                    cfg.model = name
                client.model = name
                console.print(f"[dim]model = {client.model}[/dim]")
                continue
            try:
                agent.run(line)
            except AgentQuit:
                console.print("[dim](tool declined — quitting)[/dim]")
                break
    finally:
        session.close()
        console.print(f"[dim]session log: {session.path}[/dim]")


def main(argv: list[str] | None = None) -> int:
    _fix_console_encoding()
    args = parse_args(argv)
    cfg = build_config(args)
    console = Console()

    client, is_ollama = resolve_target(cfg, console)
    if args.selftest:
        return run_selftest(cfg, client, console)

    banner(cfg, client, is_ollama, console)
    if args.prompt:
        session = Session(cfg.resolved_session_dir(), args.prompt[:40] or "oneshot")
        session.log("endpoint", base_url=client.base_url, model=client.model)
        agent = Agent(cfg, console, client, session)
        try:
            agent.run(args.prompt)
        except AgentQuit:
            console.print("[dim](declined)[/dim]")
        finally:
            session.close()
            console.print(f"[dim]session log: {session.path}[/dim]")
        return 0

    repl(cfg, client, console, resume=args.resume)
    return 0
