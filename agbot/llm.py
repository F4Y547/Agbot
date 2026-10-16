from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import BadRequestError

from .config import Config


class ThinkTagSplitter:
    """Route inline think-tag blocks (angle-bracket "think" markers) to the
    reasoning channel.

    Needed for models served without native reasoning separation
    (e.g. Ollama's OpenAI-compat endpoint) whose content may contain
    raw think tags. Handles tags split across streaming chunks.
    """

    _OPEN = "<think"
    _CLOSE = "</think"

    def __init__(self) -> None:
        self.in_think = False
        self.buf = ""

    def _partial_tail(self) -> int:
        low = self.buf.lower()
        best = 0
        for tag in (self._OPEN, self._CLOSE):
            for n in range(1, len(tag)):
                if low.endswith(tag[:n]) and n > best:
                    best = n
        return best

    def feed(self, text: str) -> tuple[str, str]:
        """Returns (reasoning_text, content_text)."""
        self.buf += text
        reasoning: list[str] = []
        content: list[str] = []
        while self.buf:
            low = self.buf.lower()
            if not self.in_think:
                ci = low.find(self._OPEN)
                cj = low.find(self._CLOSE)
                if cj != -1 and (ci == -1 or cj < ci):
                    # closing tag while we are "outside" -> what came before was reasoning
                    reasoning.append(self.buf[:cj])
                    end = self.buf.find(">", cj)
                    if end == -1:
                        break
                    self.buf = self.buf[end + 1:]
                    continue
                if ci != -1:
                    content.append(self.buf[:ci])
                    end = self.buf.find(">", ci)
                    if end == -1:
                        break
                    self.buf = self.buf[end + 1:]
                    self.in_think = True
                    continue
                hold = self._partial_tail()
                if hold:
                    content.append(self.buf[:-hold])
                    self.buf = self.buf[-hold:]
                else:
                    content.append(self.buf)
                    self.buf = ""
                break
            else:
                cj = low.find(self._CLOSE)
                if cj != -1:
                    reasoning.append(self.buf[:cj])
                    end = self.buf.find(">", cj)
                    if end == -1:
                        break
                    self.buf = self.buf[end + 1:]
                    self.in_think = False
                    continue
                hold = self._partial_tail()
                if hold:
                    reasoning.append(self.buf[:-hold])
                    self.buf = self.buf[-hold:]
                else:
                    reasoning.append(self.buf)
                    self.buf = ""
                break
        return "".join(reasoning), "".join(content)

    def flush(self) -> tuple[str, str]:
        rest, self.buf = self.buf, ""
        if not rest:
            return "", ""
        return (rest, "") if self.in_think else ("", rest)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class StreamResult:
    ok: bool = True
    error: str | None = None
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    elapsed_s: float = 0.0
    first_token_s: float | None = None
    tools_unsupported: bool = False
    aborted: bool = False


ReasoningCb = Callable[[str], None] | None


class LlmClient:
    """Thin wrapper over the OpenAI-compatible API used by SGLang and Ollama."""

    def __init__(self, cfg: Config, is_ollama: bool) -> None:
        from openai import OpenAI

        self.is_ollama = is_ollama
        self.cfg = cfg
        self.model = cfg.ollama_model if is_ollama else cfg.model
        if is_ollama:
            base = cfg.ollama_url.rstrip("/") + "/v1"
        else:
            base = cfg.base_url.rstrip("/")
        self.base_url = base
        self.client = OpenAI(
            base_url=base,
            api_key=cfg.api_key or "EMPTY",
            timeout=900.0,
            max_retries=1,
        )

    # ------------------------------------------------------------------ probe
    def probe(self, timeout: float = 4.0) -> tuple[bool, str]:
        try:
            import httpx

            r = httpx.get(
                self.base_url + "/models",
                timeout=timeout,
                headers={"Authorization": f"Bearer {self.cfg.api_key or 'EMPTY'}"},
            )
            if r.status_code == 200:
                return True, ""
            return False, f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            return False, e.__class__.__name__ + (f": {e}" if str(e) else "")

    def list_models(self) -> list[str]:
        try:
            return [m.id for m in self.client.models.list().data]
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------- chat call
    def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        cfg: Config,
        on_reasoning: ReasoningCb = None,
        on_content: ReasoningCb = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> StreamResult:
        base_kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
        )
        has_tools = bool(tools)
        if has_tools:
            base_kwargs["tools"] = tools
            base_kwargs["tool_choice"] = "auto"
        if cfg.send_reasoning_effort_for(self.is_ollama):
            base_kwargs["extra_body"] = {"reasoning_effort": cfg.reasoning_effort}

        include_usage = True
        while True:
            kwargs = dict(base_kwargs)
            if include_usage:
                kwargs["stream_options"] = {"include_usage": True}
            try:
                stream = self.client.chat.completions.create(**kwargs)
            except BadRequestError as e:
                msg = str(e).lower()
                if include_usage and "stream_options" in msg:
                    include_usage = False
                    continue
                if has_tools and "tool" in msg:
                    base_kwargs.pop("tools", None)
                    base_kwargs.pop("tool_choice", None)
                    res = self.stream_chat(
                        messages, None, cfg, on_reasoning, on_content, should_abort
                    )
                    res.tools_unsupported = True
                    return res
                return StreamResult(ok=False, error=f"{type(e).__name__}: {e}")
            except Exception as e:  # noqa: BLE001
                return StreamResult(ok=False, error=f"{type(e).__name__}: {e}")
            try:
                return self._consume(stream, on_reasoning, on_content, should_abort)
            except Exception as e:  # noqa: BLE001
                return StreamResult(
                    ok=False, error=f"stream interrupted: {type(e).__name__}: {e}"
                )

    # ------------------------------------------------------------ chunk loop
    def _consume(
        self,
        stream,
        on_reasoning: ReasoningCb,
        on_content: ReasoningCb,
        should_abort: Callable[[], bool] | None = None,
    ) -> StreamResult:
        def _cb(cb: ReasoningCb, text: str) -> None:
            if cb is None or not text:
                return
            try:
                cb(text)
            except Exception:  # noqa: BLE001
                pass

        splitter = ThinkTagSplitter()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        slots: dict[int, ToolCall] = {}
        usage: dict[str, Any] | None = None
        t0 = time.monotonic()
        t_first: float | None = None
        aborted = False

        try:
            for chunk in stream:
                if should_abort is not None and should_abort():
                    aborted = True
                    break
                u = getattr(chunk, "usage", None)
                if u is not None:
                    usage = u if isinstance(u, dict) else u.model_dump()
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue

                rc = getattr(delta, "reasoning_content", None)
                if not rc:
                    extra = getattr(delta, "model_extra", None) or {}
                    rc = extra.get("reasoning_content") or extra.get("reasoning") or ""

                for tc in getattr(delta, "tool_calls", None) or []:
                    idx = getattr(tc, "index", 0) or 0
                    slot = slots.get(idx)
                    if slot is None:
                        slot = ToolCall(id="", name="", arguments="")
                        slots[idx] = slot
                    if getattr(tc, "id", None):
                        slot.id = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        fname = getattr(fn, "name", None)
                        if fname:
                            if slot.name and fname.startswith(slot.name):
                                slot.name = fname          # full name repeated / extension
                            else:
                                slot.name += fname         # name streamed in pieces
                        if getattr(fn, "arguments", None):
                            slot.arguments += fn.arguments

                if rc:
                    if t_first is None:
                        t_first = time.monotonic() - t0
                    reasoning_parts.append(rc)
                    _cb(on_reasoning, rc)

                c = getattr(delta, "content", None)
                if c:
                    if t_first is None:
                        t_first = time.monotonic() - t0
                    r_out, c_out = splitter.feed(c)
                    if r_out:
                        reasoning_parts.append(r_out)
                        _cb(on_reasoning, r_out)
                    if c_out:
                        content_parts.append(c_out)
                        _cb(on_content, c_out)
        finally:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

        r_tail, c_tail = splitter.flush()
        if r_tail:
            reasoning_parts.append(r_tail)
            _cb(on_reasoning, r_tail)
        if c_tail:
            content_parts.append(c_tail)
            _cb(on_content, c_tail)

        tcs = [slots[k] for k in sorted(slots)]
        for i, tc in enumerate(tcs, 1):
            if not tc.id:
                tc.id = f"call_{i}"

        return StreamResult(
            ok=True,
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            tool_calls=tcs,
            usage=usage,
            elapsed_s=time.monotonic() - t0,
            first_token_s=t_first,
            aborted=aborted,
        )
