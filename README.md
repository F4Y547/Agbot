# Agbot — agent environment for DeepSeek-V4.1-Flash-UNCENSORED-FP8

A terminal agent with tools (shell, files, web) that runs **on your PC** and talks
to **`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8`** served on a rented
4×H200 GPU box — with a local Ollama fallback so the whole loop works before
(or without) the big endpoint.

```
┌────────────────────── your PC ──────────────────────┐        ┌──────── GPU box (4×H200) ────────┐
│  python -m agbot                                     │  SSH   │  docker compose                  │
│    ├─ agent loop, tools, approvals, session logs     │◄──────►│   ├─ effort-proxy   :30001       │
│    └─ OpenAI-compatible client                       │ tunnel │   └─ SGLang DSV4.1   :30000       │
│                                                      │ 30001  │      (DeepSeek-V4.1-Flash-Crack) │
│  fallback when tunnel/pod is down:                   │        └──────────────────────────────────┘
│    Ollama qwen3:8b  (local, small)                   │
└──────────────────────────────────────────────────────┘
```

## Why this shape

The model is 763 B params (~510 GB of weights, FP8/FP4) and needs an NVLink
domain of 4–8 H200s. No consumer machine can run it — so the environment splits
into a **deploy kit** for the rented box and a **local agent** that reaches it
over an SSH tunnel, degrading to a small local Ollama model when the box is off.

## Quick start (works right now, no GPU needed)

```powershell
powershell -ExecutionPolicy Bypass -File dev\setup.ps1   # venv + deps + qwen3:8b
powershell -ExecutionPolicy Bypass -File dev\smoke.ps1   # verifies the tool loop
.\.venv\Scripts\python.exe -m agbot                      # interactive agent
```

One-shot example:

```powershell
.\.venv\Scripts\python.exe -m agbot --yolo "Read requirements.txt and summarize which packages this project needs"
```

## When the GPU box is up

1. Deploy the model — full guide in [`deploy/README.md`](deploy/README.md)
   (4×H200 pod, `docker compose up -d`, ~28 min cold start).
2. Open the tunnel from your PC:
   ```powershell
   ssh -N -L 30001:127.0.0.1:30001 user@<gpu-box-ip>
   ```
3. Point the agent at it (repo `.env`):
   ```ini
   AGENT_BASE_URL=http://127.0.0.1:30001/v1
   AGENT_MODEL=deepseek-v4.1-flash-crack
   ```
4. `python -m agbot` — the banner shows `provider SGLang (remote)`. Kill the
   tunnel and it falls back to Ollama automatically.

## Agent features

- **Reasoning-stream aware**: renders `delta.reasoning_content` (dimmed 🧠 stream)
  separately from the answer; also strips inline think tags for models whose
  server doesn't split channels (Ollama). `--no-think` hides it.
- **Tools**: `list_dir`, `read_file`, `write_file`, `replace_in_file`,
  `run_shell` (PowerShell/bash), `web_fetch`.
- **Approvals**: shell/write/edit require `y/n/a(lways)/q` before running;
  `--yolo` for auto-approve; declines are fed back to the model.
- **Loop guards**: max tool steps, repeated-call detection, output truncation,
  context trimming to a char budget, hard timeouts on shell.
- **Sessions**: every message/tool result logged to `sessions/*.jsonl`;
  `--resume` continues the latest one.
- **Effort control**: `AGENT_REASONING_EFFORT=low|high|xhigh|max` or `/effort`
  in the REPL. `low` keeps agent loops snappy on the remote model.

## CLI reference

```
python -m agbot [prompt] [options]

  --url URL            endpoint override (e.g. http://127.0.0.1:30001/v1)
  --model NAME         model name to request
  --effort TIER        reasoning_effort: low|high|xhigh|max
  --no-think           hide the reasoning stream
  --yolo               auto-approve all tools
  --max-steps N        max tool rounds per message (default 12)
  --workspace PATH     directory tools operate in (default: cwd)
  --resume             continue the most recent session
  --selftest           scripted tool-call smoke test, exit 0/1
```

REPL commands: `/help /reset /tools /effort <tier> /think on|off /model <name> /exit`

## Configuration (`/.env`, see `.env.example`)

| var | default | notes |
|---|---|---|
| `AGENT_BASE_URL` | *(empty)* | remote SGLang/proxy URL; empty → Ollama |
| `AGENT_MODEL` | `deepseek-v4.1-flash-crack` | served model name |
| `AGENT_REASONING_EFFORT` | `low` | remote only; `max` = best quality, slowest |
| `AGENT_FALLBACK_OLLAMA` | `1` | auto-fallback when remote unreachable |
| `AGENT_OLLAMA_MODEL` | `qwen3:8b` | small local stand-in (tool-capable) |
| `AGENT_APPROVAL` | `on` | `off` == `--yolo` |
| `AGENT_MAX_TOKENS` | `32768` | budget for answer + reasoning; `effort=max` can emit thousands of thinking tokens |

## Layout

```
agbot/            agent package (config, llm client, tools, loop, cli)
deploy/           GPU-box deployment: docker-compose, effort proxy, guide, systemd unit
dev/              setup + smoke scripts
sessions/         JSONL session logs (created on first run)
```

## Troubleshooting

- `--selftest` fails → the endpoint's model doesn't support tool calling, or the
  server rejected `tools`. Try a bigger Ollama model (`AGENT_OLLAMA_MODEL=qwen3:14b`).
- Remote reachable but "stuck" during thinking → that's the reasoning stream;
  it can run thousands of tokens at `effort=max`. Use `low` for tool loops, or
  `/think off` to just wait for the answer.
- Raw `<think>` tags visible in output → server didn't split channels; the
  client strips inline tags automatically, but prefer the effort-proxy
  (see deploy/README.md) for a proper split.

**Content note:** the model is a weight-level *abliterated* checkpoint — it will
answer prompts the base model refuses. Use it responsibly, and keep the
endpoint private (tunnel/firewall); nothing about this scaffolding changes what
the model itself will say.
