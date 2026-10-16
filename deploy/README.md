# Deploying the model on a GPU box

This directory deploys **`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8`**
(763 B params, ~510 GB of weights, FP8/FP4 MoE) behind an OpenAI-compatible API
using the SGLang preview image the model authors validated.

> **This will not run on your PC.** Minimum: a **4×H200 NVLink pod** (NV18 mesh),
> ~1 TB system RAM, ~1 TB fast disk. 8×H200 also works (see tuning). Sub-TP4
> (1–2 GPU) does not work on the model shape at all.

---

## 1. Rent the hardware

Known-good: **JarvisLabs** (`dev-dsv41` image, validated by the model authors),
**RunPod** / **Lambda** / **Together-style** 4× or 8× H200 SXM pods.

Rough cost: an H200-class pod runs about **$14–20/hr for 4 GPUs** (spot/reserved
cheaper). Cold start is ~28 min, so do a full bring-up session, tune, then stop
the pod — don't pay for idle minutes.

Box requirements:

| item | requirement |
|---|---|
| GPUs | 4× H200 SXM in one NVLink domain (8× optional) |
| system RAM | ≥ 256 GB (≥ 1 TB if using the Engram host table, default) |
| disk | ~1 TB free (weights are ~510 GB, warm cache helps restarts) |
| OS | Ubuntu 22.04/24.04 with recent NVIDIA driver, Docker + nvidia-container-toolkit |
| ports | only your SSH port open; SGLang/proxy stay behind localhost + tunnel |

## 2. Bring-up

```bash
# on the GPU box
git clone <this repo> /opt/agbot && cd /opt/agbot/deploy
cp .env.example .env
nano .env                      # paste your HF_TOKEN, keep defaults

docker compose up -d
docker compose logs -f sglang  # watch the weight load (~28 min cold)
```

When `docker compose ps` shows **both healthy** (~30–45 min cold, ~10 min warm
with the cached weights), the server is up. Quick check:

```bash
curl -s http://127.0.0.1:30001/v1/models | head -c 400
curl -s http://127.0.0.1:30001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash-crack",
       "messages":[{"role":"user","content":"Say OK and one sentence about yourself."}],
       "reasoning_effort":"low","max_tokens":512}' | head -c 1200
```

## 3. Connect your PC's agent (SSH tunnel)

The SGLang and proxy ports are bound to localhost on the box. On **your PC**:

```powershell
ssh -N -L 30001:127.0.0.1:30001 user@<gpu-box-ip>
```

Then set in the repo's `.env` on your PC:

```ini
AGENT_BASE_URL=http://127.0.0.1:30001/v1
AGENT_MODEL=deepseek-v4.1-flash-crack
```

and run `python -m agbot` — it auto-detects the tunnel, uses it, and falls back
to Ollama when the tunnel/pod is down.

**Why the proxy on 30001 instead of SGLang on 30000?** The `deepseek-v41`
reasoning parser only splits `reasoning_content` / `content` when the request
carries an explicit `reasoning_effort`. The proxy injects it for any client that
forgets (e.g. a plain web UI or `curl`), so raw think tags never leak into
`content`. The Agbot client always sends an effort itself, so it can also talk
to 30000 directly.

## 4. Tuning knobs (`deploy/.env`)

| var | meaning |
|---|---|
| `CONTEXT_LENGTH` | 262144 default. KV = 890 B/token; 1M context drops concurrency to 12. Storm-proof agent use: keep 262144 (≈80 concurrent). |
| `MAX_RUNNING_REQUESTS` | lower to 12 if you run 1M ctx and see OOM under load |
| `ENGRAM_HOST_TABLE` | `1` moves 203 GB of Engram tables to host RAM (needs ~1 TB RAM, frees ~46 GiB/GPU). Set `0` on smaller-RAM boxes → weight load becomes 122 GB/GPU, reduce context accordingly. |
| `PROXY_EFFORT` | effort injected for clients that omit it: `max` (best) or `low` (fast, good for tool loops) |
| `SERVER_REASONING_EFFORT` | server-side default (`max` on the base model) |
| `TP_SIZE`/`EP_SIZE` | 4/4 on 4×H200. On 8×H200 use `TP_SIZE=8` and **delete** the `--ep-size` line in `docker-compose.yml`. |

### Optional: DSpark speculative decoding (+~12% decode)

Add to the `command:` block in `docker-compose.yml` (and mount the SPS table,
see the commented volume):

```yaml
      --speculative-algorithm DSPARK
      --speculative-dspark-sps-table-path /dspark/dspark_sps.json
```

Generate the cost table on the box:
`python -m sglang.benchmark.dspark_sps_profiler` (requires
`SGLANG_RAGGED_VERIFY_MODE=cap-accept`, already set). Single-stream decode:
101 → 113 tok/s (T=0). Skip this for the first bring-up.

### Optional: from-source SGLang (instead of the image)

The `dsv4.1` branch of `sgl-project/sglang` (PR #38798):

```bash
git clone --depth 1 --branch dsv4.1 https://github.com/sgl-project/sglang.git
python3 -m venv sglang-venv && sglang-venv/bin/pip install -U pip setuptools wheel
export PATH=/root/.cargo/bin:$PATH          # Rust toolchain needed for the build
cd sglang/python && ../sglang-venv/bin/pip install -e .
export PATH=$(dirname $(which ninja)):$PATH # ninja MUST be on PATH or JIT build dies at ~28 min in
```

### vLLM

Not ready for this model at time of writing (no `DeepseekV41` entry in
`registry.py`; umbrella PR #56214 pending). Use SGLang.

## 5. Troubleshooting (all of these bit the model authors)

| symptom | fix |
|---|---|
| `Mxfp4FlashinferCutlassMoEMethod requires ... multiples of 128` | you forgot `--ep-size 4` at TP4 |
| `FileNotFoundError: 'ninja'` then EXIT 137 mid load | put ninja on PATH inside the container / source env |
| raw `<think>` tags leaking into `content` | client didn't send `reasoning_effort`; use the proxy or send it explicitly |
| tool calls not parsed | parser must be `deepseekv41` (spaced DSML tags); the V4 parser won't do |
| `torchcodec` / `libavutil.so.56` errors | `apt-get install -y ffmpeg` (video only; text/image unaffected) |
| OOM under both long-context concurrency | lower `--max-running-requests` to 12, `--chunked-prefill-size` 4096, or lower `CONTEXT_LENGTH` |
| container dead after a rare OOM | compose `restart: unless-stopped` + the systemd unit below |

## 6. Ops

```bash
# auto-start on boot (optional)
sudo cp systemd/agbot-stack.service /etc/systemd/system/
sudo systemctl enable --now agbot-stack

# status / logs
docker compose ps && docker compose logs --tail 100 effort-proxy

# teardown (frees GPU billing; keeps the 510 GB weight cache volume)
docker compose down

# nuke the weight cache too (do this before deleting a pod)
docker compose down -v
```

**Cost discipline:** stop the pod when done. The images and weights re-pull in
~10–20 min warm, and the agent falls back to Ollama meanwhile.

**Content note:** this checkpoint is deliberately uncensored — it answers
prompts the base model refuses, across all measured harm categories. It is a
research/personal-use model; you are responsible for what you generate and for
keeping the endpoint private (the SSH tunnel default matters).
