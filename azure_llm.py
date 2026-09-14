"""Minimal client for Azure AI Foundry's unified model-inference REST API.

One Foundry project (one endpoint + one key) can host several model
deployments from different families -- GPT, Grok, DeepSeek, Llama, etc. This
hits the OpenAI-chat-completions-compatible route that works identically
across all of them, differentiated only by `model` = the deployment name
shown in the Foundry portal's "Models + endpoints" tab. That is also why the
three deployments in this project share one AZURE_FOUNDRY_ENDPOINT /
AZURE_FOUNDRY_API_KEY pair in .env instead of having one each.

Route derivation: the endpoint handed out by the Foundry portal is
project-scoped (".../api/projects/<project>"), but the unified inference
route lives at the resource root, one level up -- see
https://learn.microsoft.com/en-us/azure/ai-foundry/model-inference/how-to/inference
"""
from __future__ import annotations

import logging
import queue
import threading
import time

import requests

log = logging.getLogger("azure_llm")

API_VERSION = "2024-05-01-preview"

# requests' own `timeout=` only guards the gap between individual bytes/
# chunks arriving, not the call's total duration -- confirmed live: a Grok-4.6
# call sat past 5 real minutes with `timeout=120` set and never raised,
# apparently because the connection kept receiving *something* often enough
# to keep resetting that per-chunk clock without ever completing. A hard
# wall-clock cap, enforced from the calling thread via a daemon worker (see
# _request_with_hard_timeout), bounds the actual wait regardless of what's
# happening at the socket level.
#
# Originally implemented with a module-level ThreadPoolExecutor -- wrong,
# and not just "leaks a worker": CPython's ThreadPoolExecutor registers an
# atexit hook that unconditionally .join()s (no timeout) every worker thread
# the pool ever created, and its workers are plain non-daemon threads. If a
# hard-timeout ever actually fires because a request is genuinely and
# permanently stuck -- which is exactly the scenario this constant exists
# for -- that abandoned worker thread would still be joined at process exit,
# hanging the whole script forever at shutdown even though every real
# decision had already finished and every output file was already written.
# A plain daemon thread per call has no such hook and can never block exit.
_HARD_CALL_TIMEOUT_SECONDS = 180.0

# Same problem, different vendor: TradingAgents' Yahoo Finance data tools
# (tradingagents/dataflows/y_finance.py, stockstats_utils.py) set no
# request timeout that reliably bounds a stalled connection -- yf_retry
# only catches YFRateLimitError (an actual 429 response), not a connection
# that simply never replies. Confirmed live: a single TradingAgentsGraph
# .propagate() call went completely silent for 13+ minutes (zero CPU,
# lsof still showing the TCP connection to Yahoo ESTABLISHED) with no
# exception ever raised to catch. propagate() runs a full multi-agent
# debate (several sequential LLM calls), so this needs a much larger
# budget than a single LLM call -- 900s is generous headroom above what a
# real, working call has taken in this project's own runs, not a tight
# bound expected to fire often.
_PROPAGATE_HARD_TIMEOUT_SECONDS = 900.0


def _request_with_hard_timeout(fn, *args, hard_timeout: float, **kwargs):
    """Runs fn(*args, **kwargs) on a fresh daemon thread with a hard
    wall-clock cap. Raises TimeoutError if it doesn't finish in time; the
    worker thread is then abandoned (daemon=True -- see module docstring
    above for why this must NOT be a ThreadPoolExecutor worker)."""
    result: queue.Queue = queue.Queue(maxsize=1)

    def _run():
        try:
            result.put(("ok", fn(*args, **kwargs)))
        except BaseException as exc:      # forwarded to the caller, not swallowed
            result.put(("error", exc))

    threading.Thread(target=_run, daemon=True).start()
    try:
        status, payload = result.get(timeout=hard_timeout)
    except queue.Empty:
        raise TimeoutError(
            f"did not respond within {hard_timeout:.0f}s (hard wall-clock cap) -- "
            f"the connection was open but stalled, requests' own timeout= did not fire"
        )
    if status == "error":
        raise payload
    return payload


class TransientAzureError(Exception):
    """A failure worth retrying as-is: HTTP 429/5xx, or a 200 whose `choices`
    came back empty. Confirmed live against grok-4.6 under concurrent load: a
    real production run hit 121 empty-`choices` 200 responses out of ~300
    decisions (~32%), each one previously an unhandled IndexError that
    phase3.run_episode's per-day guard silently downgraded to a no-trade day
    -- correct in that it didn't crash the run, but wrong in that ~a third of
    that model's "decisions" were actually a swallowed bug, not HOLD. An
    empty-choices 200 with no content_filter_results populated looks like a
    transient backend hiccup rather than a real moderation block (a genuine
    content-filter block still returns one choice, with finish_reason=
    "content_filter"), so it goes in the same retryable bucket as 429/5xx
    rather than being treated as a permanent, request-is-bad failure."""

# Deployment names as chosen in the Foundry portal -- NOT secrets, so they
# live in code rather than .env. Keyed by the same string used as the
# coordination_cost.MODEL_PRICES lookup key; only DeepSeek's casing actually
# differs from its price-table key.
AZURE_DEPLOYMENT_NAMES: dict[str, str] = {
    "gpt-5.6-sol": "gpt-5.6-sol",
    "grok-4.6": "grok-4.6",
    "deepseek-v4-pro": "DeepSeek-V4-Pro",
    "kimi-k2.6": "Kimi-K2.6",
}

# Same idea, separate table -- these deployments route through
# phase1_5_agent.Phase15Bridge's provider="anthropic_foundry" branch
# (anthropic.AnthropicFoundry, native Messages API), not AzureFoundryClient
# above, so they're kept in their own namespace rather than mixed into
# AZURE_DEPLOYMENT_NAMES even though both live in the same Foundry project.
ANTHROPIC_FOUNDRY_DEPLOYMENT_NAMES: dict[str, str] = {
    "claude-opus-5-azure": "claude-opus-5",
}

# Per-model max_tokens override, applied on top of Phase15Bridge's 1024
# default. Confirmed live: Kimi-K2.6 emits a separate `reasoning_content`
# block before its final `content` (a real chain-of-thought, despite "K2.6"
# sounding like the non-reasoning tier) and counts BOTH toward
# completion_tokens/max_tokens. A real trading-decision prompt (2 tickers)
# consumed the ENTIRE budget on reasoning alone at both 1024 and 2048,
# returning empty content -> every decision failed JSON parsing. Needed
# 4096 to leave any room for the actual answer, and that was only a
# 2-ticker test; the real pipeline runs a 20-ticker universe, so this is set
# with real margin above the smallest value observed to work, not the bare
# minimum. GPT-5.6-Sol and Grok-4.6 are also reasoning-tuned but have not
# shown this failure mode at the 1024 default in any run this session, so
# this is a targeted override, not a blanket bump that would waste tokens
# (and money) on the three models that don't need it.
AZURE_MAX_TOKENS_OVERRIDE: dict[str, int] = {
    "kimi-k2.6": 8192,
}


class AzureFoundryClient:
    def __init__(self, project_endpoint: str, api_key: str):
        resource_root = project_endpoint.split("/api/projects/")[0]
        self.url = f"{resource_root}/models/chat/completions?api-version={API_VERSION}"
        self.api_key = api_key

    def chat_completions_create(self, deployment: str, messages: list[dict],
                                 max_tokens: int, temperature: float = 0.0,
                                 max_retries: int = 5) -> dict:
        """One decision's call, with retry on TRANSIENT failure.

        A full episode sweep is a single long-running per-model process
        making ~thousands of these sequentially (see run_concurrent.py's
        docstring on why this can't parallelize within one model), and
        _attempt raising propagates all the way up through run_episode and
        aborts every remaining episode in that process -- confirmed by
        reading coordination_cost.CoordinationMeter.__call__, which re-raises
        after recording. One transient stall (already observed live against
        this exact endpoint -- see _HARD_CALL_TIMEOUT_SECONDS) should not be
        allowed to cost the rest of an unattended multi-hour run. Retries
        TimeoutError (our own hard-cap, see _request_with_hard_timeout),
        requests.exceptions.Timeout (covers BOTH ConnectTimeout and
        ReadTimeout -- ReadTimeout specifically is NOT a subclass of
        ConnectionError, so catching only ConnectionError here would miss
        the single most likely failure mode: requests' own 120s read timeout
        firing normally on an ordinary slow response, which is the common
        case this whole retry path exists for, not the rare hard-cap case),
        and connection-level failures. A real 4xx/5xx from a fully-formed
        response (already retried once inside _attempt for the
        param-mismatch case) means something is actually wrong with the
        request or account and would just fail identically on every retry.
        """
        delay = 5.0
        for attempt in range(max_retries + 1):
            try:
                return self._attempt(deployment, messages, max_tokens, temperature)
            except (TimeoutError, requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout, TransientAzureError) as exc:
                if attempt == max_retries:
                    raise
                log.warning("%s call failed (%s), retry %d/%d in %.0fs",
                           deployment, exc, attempt + 1, max_retries, delay)
                time.sleep(delay)
                delay = min(delay * 3, 120.0)   # cap growth -- observed outages can span
                                                 # several consecutive days of decisions, so
                                                 # don't give up too fast, but don't let one
                                                 # bad call block a batch run for 10+ minutes
        raise AssertionError("unreachable")

    def _attempt(self, deployment: str, messages: list[dict],
                 max_tokens: int, temperature: float) -> dict:
        """One HTTP round trip, OpenAI chat-completions request/response
        shape, self-correcting for the one class of error that is NOT
        transient: three different model families sit behind this one route
        and each rejects a different subset of standard chat-completions
        params, e.g. confirmed live against this project's own deployments:
          - GPT-5.6-Sol (reasoning-tuned): rejects a non-default `temperature`
            ("Only the default (1) value is supported") outright.
          - Newer reasoning models generally expect `max_completion_tokens`
            instead of `max_tokens`.
        Rather than hardcode a per-deployment quirk table that a future model
        swap invalidates, drop/rename whatever param the error names and
        retry -- self-corrects for quirks not listed above too.
        """
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        body = {"model": deployment, "messages": messages, "temperature": temperature,
                "max_completion_tokens": max_tokens}

        for _ in range(4):        # generous bound; each retry removes exactly one param
            resp = _request_with_hard_timeout(
                requests.post, self.url, headers=headers, json=body, timeout=120,
                hard_timeout=_HARD_CALL_TIMEOUT_SECONDS,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                raise TransientAzureError(
                    f"{deployment}: HTTP {resp.status_code} -- {resp.text[:500]}"
                )
            if resp.status_code != 400:
                resp.raise_for_status()   # any other non-2xx (401/403/404/...) -- not retryable
                data = resp.json()
                if not data.get("choices"):
                    raise TransientAzureError(
                        f"{deployment}: HTTP 200 with empty/missing choices -- "
                        f"{str(data)[:500]}"
                    )
                return data

            try:
                err = resp.json()["error"]
            except Exception:
                resp.raise_for_status()  # unparseable 400 body -- surface it as-is

            param = err.get("param")
            msg = err.get("message", "")
            if param == "temperature" and "temperature" in body:
                log.info("%s rejected temperature (%s); retrying without it", deployment, msg)
                body.pop("temperature")
            elif (param == "max_completion_tokens" or "max_completion_tokens" in msg) \
                    and "max_completion_tokens" in body:
                body["max_tokens"] = body.pop("max_completion_tokens")
            elif (param == "max_tokens" or "max_tokens" in msg) and "max_tokens" in body:
                body["max_completion_tokens"] = body.pop("max_tokens")
            else:
                resp.raise_for_status()  # a 400 this function doesn't know how to fix

        resp.raise_for_status()
        return resp.json()
