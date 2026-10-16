#!/usr/bin/env python3
"""reasoning_effort injector for SGLang's deepseek-v41 reasoning parser.

Why this exists
---------------
SGLang's `--reasoning-parser deepseek-v41` only splits the model output into
`delta.reasoning_content` / `delta.content` on the code path that receives an
explicit `reasoning_effort` in the request. Requests that omit it get raw
think tags leaked inside `delta.content`.

This tiny reverse proxy sits in front of SGLang and adds `reasoning_effort`
to /v1/chat/completions requests that don't carry one (and optionally
overrides any that do). Everything else is passed through untouched,
streaming included.

Run:
  pip install aiohttp
  python effort_proxy.py --listen 0.0.0.0:30001 --upstream http://127.0.0.1:30000 --effort max
"""
from __future__ import annotations

import argparse
import json

from aiohttp import ClientSession, web

HOP_BY_HOP = {
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "upgrade",
    "content-length",
    "content-encoding",
    "host",
}


async def proxy(request: web.Request) -> web.StreamResponse:
    app = request.app
    body = await request.read()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}

    if request.path.endswith("/v1/chat/completions") and body:
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            if app["effort"] and (app["override"] or not data.get("reasoning_effort")):
                data["reasoning_effort"] = app["effort"]
                body = json.dumps(data).encode("utf-8")
                headers["content-type"] = "application/json"

    url = app["upstream"] + request.path_qs
    session: ClientSession = app["session"]
    try:
        upstream_resp = await session.request(request.method, url, data=body, headers=headers)
    except Exception as e:  # noqa: BLE001
        raise web.HTTPBadGateway(text=f"upstream error: {e}")

    resp = web.StreamResponse(status=upstream_resp.status)
    for key, value in upstream_resp.headers.items():
        if key.lower() not in HOP_BY_HOP:
            resp.headers[key] = value
    await resp.prepare(request)
    try:
        async for chunk in upstream_resp.content.iter_chunked(8192):
            await resp.write(chunk)
    except (ConnectionResetError, ConnectionAbortedError):
        pass
    finally:
        upstream_resp.release()
    try:
        await resp.write_eof()
    except ConnectionResetError:
        pass
    return resp


async def main() -> None:
    parser = argparse.ArgumentParser(description="Inject reasoning_effort in front of SGLang")
    parser.add_argument("--listen", default="0.0.0.0:30001")
    parser.add_argument("--upstream", default="http://127.0.0.1:30000")
    parser.add_argument("--effort", default="max", help="value injected when the client omits one")
    parser.add_argument(
        "--override",
        action="store_true",
        help="overwrite a client-provided reasoning_effort instead of only filling gaps",
    )
    args = parser.parse_args()

    host, _, port = args.listen.rpartition(":")

    app = web.Application(client_max_size=512 * 1024 * 1024)
    app["upstream"] = args.upstream.rstrip("/")
    app["effort"] = args.effort
    app["override"] = args.override
    app["session"] = ClientSession()

    async def _cleanup(_app: web.Application) -> None:
        await _app["session"].close()

    app.on_cleanup.append(_cleanup)
    app.router.add_route("*", "/{tail:.*}", proxy)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host or "0.0.0.0", int(port))
    await site.start()
    print(
        f"effort-proxy listening on {args.listen} -> {app['upstream']} "
        f"(inject effort={args.effort!r}, override={args.override})",
        flush=True,
    )
    import asyncio

    await asyncio.Event().wait()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
