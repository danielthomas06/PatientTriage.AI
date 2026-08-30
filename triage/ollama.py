"""Local backend -- a model on this machine, via Ollama.

Not a downgrade. For a hospital it is the stronger deployment story: no patient
narrative leaves the building, no third-party data processing agreement, and no
dependency on a network that fails exactly when the department is busiest. It
turns the degradation ladder's middle rung from a claim into something you can
demonstrate by switching the wifi off.

    tier 1  hosted model      richest extraction
    tier 2  LOCAL MODEL       this file -- no network, no key
    tier 3  keyword_seed      no model at all
    tier 4  structured form   the engine alone

Nothing downstream changes between tiers. The engine, the discriminator
vocabulary and the evidence check are identical, so a weaker model produces more
paraphrases and more unsupported negatives -- and those are rejected, which
means MORE UNKNOWNS AND LOWER CONFIDENCE, never confidently-wrong triage. A
cheaper model costs throughput. It cannot cost safety.

No new dependencies: urllib from the standard library.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .core import Protocol, Source

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "180"))

THINK = os.environ.get("OLLAMA_THINK", "").lower() in ("1", "true", "yes")
"""Let a reasoning model think before answering. Off by default -- see `chat`."""

KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
"""How long the server holds the weights in VRAM after a call."""

RICH_PROMPT = os.environ.get("OLLAMA_RICH_PROMPT", "").lower() in ("1", "true", "yes")
"""Send the per-check clinical notes as well as the check names.

Off by default because on a 7B the notes are ~65% of the prompt and buy nothing
-- a small model grades from the label either way, and on CPU that padding is
seconds per call. A larger model on a GPU has neither problem and does use them,
so set OLLAMA_RICH_PROMPT=1 when you have the headroom."""


class Unreachable(RuntimeError):
    """Server down, model not pulled, or the response was unusable."""


def available(model: str = MODEL, host: str = HOST) -> tuple[bool, str]:
    """Cheap enough to call on boot, and it distinguishes the failure modes --
    'server down' and 'model not pulled' need different fixes."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as r:
            tags = json.loads(r.read())
    except Exception as exc:
        reason = getattr(exc, "reason", exc)
        return False, f"Ollama not reachable at {host} ({reason}). Is it running?"

    names = [m.get("name", "") for m in tags.get("models", [])]
    if not names:
        return False, f"Ollama is up but no models are pulled. Try: ollama pull {model}"
    if model not in names and not any(n.split(":")[0] == model.split(":")[0] for n in names):
        return False, (
            f"model {model!r} not pulled. Available: {', '.join(names)}. "
            f"Try: ollama pull {model}"
        )
    return True, f"{model} ready at {host}"


def simplify(schema: dict) -> dict:
    """Flatten `anyOf` unions to string enums.

    Ollama compiles JSON Schema into a llama.cpp grammar, and unions are where
    that compilation misbehaves. Booleans come back as the strings "true"/"false"
    and `coerce` converts them back before anything downstream sees them.
    """
    if not isinstance(schema, dict):
        return schema
    if "anyOf" in schema:
        enums: list = []
        for branch in schema["anyOf"]:
            if branch.get("type") == "boolean":
                enums += ["true", "false"]
            elif branch.get("enum"):
                enums += branch["enum"]
        return {"type": "string", "enum": sorted(set(enums))} if enums else {"type": "string"}
    return {
        k: (
            simplify(v)
            if isinstance(v, dict)
            else [simplify(i) for i in v] if isinstance(v, list) else v
        )
        for k, v in schema.items()
    }


def coerce(value):
    if isinstance(value, str):
        low = value.strip().lower()
        if low == "true":
            return True
        if low == "false":
            return False
    return value


def chat(system: str, user: str, schema: dict, *, model: str = MODEL, host: str = HOST) -> dict:
    """One grammar-constrained call. Raises Unreachable; never returns junk."""
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system + _REINFORCE},
                {"role": "user", "content": user},
            ],
            "format": simplify(schema),
            "stream": False,
            # Reasoning models (Qwen3 and friends) emit a long thinking block by
            # default. For structured extraction that is pure cost: the answer is
            # a schema-constrained object, not an argument, and the guard checks
            # the quote afterwards either way. Measured on a 27B: 1.8s -> 0.8s on
            # a trivial prompt, and far more on a real one. Ignored by models
            # that do not think.
            "think": THINK,
            # Keep the weights resident between calls. A triage encounter is a
            # burst of short calls with gaps; without this the model can unload
            # and every reload is seconds.
            "keep_alive": KEEP_ALIVE,
            # Temperature 0 because reproducibility is a property this system
            # claims, and it has to hold on every tier, not just the hosted one.
            "options": {"temperature": 0, "num_ctx": 8192},
        }
    ).encode()

    req = urllib.request.Request(
        f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            payload = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        raise Unreachable(f"Ollama HTTP {exc.code}: {exc.read().decode(errors='replace')[:200]}") from exc
    except urllib.error.URLError as exc:
        raise Unreachable(f"Ollama unreachable: {exc.reason}") from exc
    except TimeoutError as exc:
        raise Unreachable(
            f"Ollama timed out after {TIMEOUT:.0f}s. The first call loads the model "
            f"into memory and is slow; try again or use a smaller model."
        ) from exc

    text = ((payload.get("message") or {}).get("content") or "").strip()
    if not text:
        raise Unreachable("empty response")
    if text.startswith("```"):                      # small models still fence sometimes
        text = text.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise Unreachable(f"not valid JSON: {exc}; got {text[:160]!r}") from exc


def chat_vision(system: str, user: str, image_b64: str, schema: dict, *,
                 model: str, host: str = HOST) -> dict:
    """Same contract as `chat`, plus one image on the user turn.

    A separate function rather than an optional parameter on `chat`: vision
    models are a distinct, usually-different-sized pull from the text model
    (see triage/vision.py's own OLLAMA_VISION_MODEL), and mixing an image
    parameter into the text-only call site would make it look like every
    local model can take one, which most cannot.
    """
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system + _REINFORCE},
                {"role": "user", "content": user, "images": [image_b64]},
            ],
            "format": simplify(schema),
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "options": {"temperature": 0},
        }
    ).encode()

    req = urllib.request.Request(
        f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            payload = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        raise Unreachable(f"Ollama HTTP {exc.code}: {exc.read().decode(errors='replace')[:200]}") from exc
    except urllib.error.URLError as exc:
        raise Unreachable(f"Ollama unreachable: {exc.reason}") from exc
    except TimeoutError as exc:
        raise Unreachable(f"Ollama timed out after {TIMEOUT:.0f}s analysing the photo.") from exc

    text = ((payload.get("message") or {}).get("content") or "").strip()
    if not text:
        raise Unreachable("empty response")
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise Unreachable(f"not valid JSON: {exc}; got {text[:160]!r}") from exc


# Short and blunt. Small models follow a terse reminder better than a long one,
# and the hard constraints are enforced by the evidence check regardless.
_REINFORCE = (
    "\n\nReturn ONLY JSON matching the schema. No preamble, no markdown fences. "
    "Copy quotes exactly from the source text -- if you cannot copy it word for "
    "word, leave the item out entirely."
)


def vocabulary(protocol: Protocol, *, notes: bool = False) -> str:
    """The check list, trimmed.

    Clinical notes help a large model grade a borderline finding and are dead
    weight for a 7B doing grammar-constrained decoding -- measured at 65% of the
    prompt, which on CPU inference is seconds per call. Off by default; set
    OLLAMA_RICH_PROMPT=1 on hardware that can afford them.
    """
    notes = notes or RICH_PROMPT
    lines = []
    for did, d in sorted(protocol.discriminators.items()):
        if d.source is Source.SENSITIVE:
            continue        # never inferred from narrative, never asked by a kiosk
        lines.append(f"{did}  {d.text}")
        if notes:
            lines.append(f"    {d.question}")
    return "\n".join(lines)
