"""Minimal OpenAI-compatible LLM client (DeepSeek by default).

Environment:
  DEEPSEEK_API_KEY    required for live calls
  DEEPSEEK_BASE_URL   default https://api.deepseek.com
  DEEPSEEK_MODEL      default deepseek-v4-pro
  OPENAI_MODEL        default gpt-5 (OPENAI_* path)
  OFOX_MODEL / LLM_MODEL  model id for the relay path
  LLM_TEMPERATURE     default 1.0
  LLM_MAX_TOKENS      default 65536

Any OpenAI-compatible endpoint works too: set OPENAI_API_KEY / OPENAI_BASE_URL
and pass model=... explicitly.
"""
import json
import os
import pathlib
import time

DEFAULT_SYSTEM = "Answer the user's request.\n"


def _load_dotenv():
    """repro/.env -> os.environ (KEY=VALUE lines, no export needed)."""
    env_file = pathlib.Path(__file__).resolve().parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))


_load_dotenv()


class LLMError(RuntimeError):
    pass


# ofox.io list price, USD per 1M tokens (context <= 272k)
PRICES = {
    "ofox": {"input": 10.0, "cached_read": 1.0, "cache_write": 12.5, "output": 50.0},
}


def estimate_cost(usage, provider="ofox"):
    """USD cost of one call from the usage dict."""
    p = PRICES.get(provider)
    if not p or not usage:
        return {"usd": 0.0}
    prompt = usage.get("prompt_tokens", 0) or 0
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens", 0) or 0
    if not cached:
        cached = usage.get("prompt_cache_hit_tokens", 0) or 0
    cache_write = details.get("cache_write_tokens", 0) or 0
    uncached = max(prompt - cached - cache_write, 0)
    out = usage.get("completion_tokens", 0) or 0
    usd = (uncached * p["input"] + cached * p["cached_read"]
           + cache_write * p["cache_write"] + out * p["output"]) / 1e6
    return {"usd": round(usd, 6), "input_uncached": uncached, "cached_read": cached,
            "cache_write": cache_write, "output": out,
            "tier": "<=272k" if prompt <= 272_000 else ">272k"}


def default_config(model=None, temperature=None, max_tokens=None):
    if os.environ.get("OFOX_API_KEY"):
        cfg = {
            "provider": "ofox",
            "base_url": os.environ.get("OFOX_BASE_URL", "https://api.ofox.io/v1"),
            "api_key": os.environ["OFOX_API_KEY"],
            "model": model or os.environ.get("LLM_MODEL", "openai/gpt-6-astra"),
        }
    elif os.environ.get("DEEPSEEK_API_KEY"):
        cfg = {
            "provider": "deepseek",
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "api_key": os.environ["DEEPSEEK_API_KEY"],
            "model": model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        }
    elif os.environ.get("OPENAI_API_KEY"):
        cfg = {
            "provider": "openai",
            "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "api_key": os.environ["OPENAI_API_KEY"],
            "model": model or os.environ.get("OPENAI_MODEL", "gpt-5"),
        }
    else:
        cfg = {"provider": None, "base_url": None, "api_key": None,
               "model": model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")}
    cfg["temperature"] = float(temperature if temperature is not None
                               else os.environ.get("LLM_TEMPERATURE", 1.0))
    cfg["max_tokens"] = int(max_tokens if max_tokens is not None
                            else os.environ.get("LLM_MAX_TOKENS", 65536))
    default_effort = "xhigh" if cfg.get("provider") == "ofox" else "max"
    cfg["reasoning_effort"] = os.environ.get("LLM_REASONING_EFFORT", default_effort)
    return cfg


def available():
    return bool(os.environ.get("OFOX_API_KEY")
                or os.environ.get("DEEPSEEK_API_KEY")
                or os.environ.get("OPENAI_API_KEY"))


def complete(prompt, config=None, system=DEFAULT_SYSTEM, retries=4,
             timeout=7200, log=print):
    """One chat completion. Returns a JSON-serialisable record.

    Set LLM_STREAM=1 to consume server-sent events incrementally, which keeps
    very long reasoning calls from being buffered by an intermediary.
    """
    from openai import OpenAI

    cfg = config or default_config()
    if not cfg.get("api_key"):
        raise LLMError(
            "no API key: export OFOX_API_KEY=... (or DEEPSEEK_API_KEY / OPENAI_API_KEY)")

    stream = os.environ.get("LLM_STREAM", "0") == "1"
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"],
                    timeout=timeout, max_retries=0)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_err = None
    t0 = time.time()
    for attempt in range(1, retries + 1):
        try:
            if stream:
                chunks = client.chat.completions.create(
                    model=cfg["model"], messages=messages,
                    temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
                    reasoning_effort=cfg.get("reasoning_effort") or None,
                    stream=True, stream_options={"include_usage": True})
                text, reasoning, usage, finish = "", "", None, None
                for ch in chunks:
                    if getattr(ch, "usage", None):
                        usage = ch.usage.model_dump() if hasattr(ch.usage, "model_dump") else dict(ch.usage)
                    if not ch.choices:
                        continue
                    choice = ch.choices[0]
                    if getattr(choice, "finish_reason", None):
                        finish = choice.finish_reason
                    d = choice.delta
                    if getattr(d, "content", None):
                        text += d.content
                    if getattr(d, "reasoning_content", None):
                        reasoning += d.reasoning_content
                resp = None
            else:
                resp = client.chat.completions.create(
                    model=cfg["model"], messages=messages,
                    temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
                    reasoning_effort=cfg.get("reasoning_effort") or None,
                )
            break
        except Exception as e:  # noqa: BLE001 - provider errors vary
            last_err = e
            msg = str(e)
            log(f"  [llm] attempt {attempt} failed: {msg[:300]}")
            if "reasoning_effort" in msg and "xhigh" in msg:
                cfg["reasoning_effort"] = "high"
            time.sleep(min(60, 2 ** attempt) if attempt < retries else 0)
    else:
        raise LLMError(f"LLM call failed after {retries} attempts: {last_err}")

    if not stream:
        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        finish = choice.finish_reason
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)
    usage = usage or {}
    record = {
        "provider": cfg["provider"],
        "model": cfg["model"],
        "base_url": cfg["base_url"],
        "temperature": cfg["temperature"],
        "max_tokens": cfg["max_tokens"],
        "reasoning_effort": cfg.get("reasoning_effort"),
        "finish_reason": finish,
        "truncated": finish == "length",
        "text": text,
        "reasoning": reasoning,
        "usage": usage,
        "cost": estimate_cost(usage, cfg["provider"]),
        "wall_s": round(time.time() - t0, 1),
        "created": int(time.time()),
    }
    return record


def save_record(record, run_dir, prompt):
    import pathlib

    d = pathlib.Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "prompt.txt").write_text(prompt)
    (d / "reply.md").write_text(record["text"])
    if record.get("reasoning"):
        (d / "reasoning.md").write_text(record["reasoning"])
    meta = {k: v for k, v in record.items() if k not in ("text", "reasoning")}
    (d / "usage.json").write_text(json.dumps(meta, indent=1))
    return d
