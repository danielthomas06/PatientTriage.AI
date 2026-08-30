"""A photo in, a proposed caption out -- never a recorded finding.

Same tiered posture as identity_extract.py: hosted Claude (vision-capable)
first, a local vision model second. Unlike every other tier in this project,
there is NO keyword fallback here -- there is no regex that looks at a
picture. Both tiers unreachable means the feature is genuinely unavailable
right now, and says so plainly, rather than inventing a caption from nothing.

The model's job is narrow, matching the one rule that has held everywhere
else a model gets near this system's findings: it may propose, a human must
confirm. See serve.py's /api/photo/confirm -- the ONLY path from a caption to
a check the engine will ever read, and it requires a nurse's click. Nothing
in this module writes to a BeliefState, an Encounter, or a Ledger.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass

MODEL = os.environ.get("TRIAGE_VISION_MODEL", "claude-opus-5")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")

# Only checks with a genuine visual signature in the CTAS pack (see
# triage/protocols/ctas.py) -- deliberately short. A photo's caption is useful
# on its own every time; a mapped check is a bonus for the few complaints
# where the vocabulary already has a visual one, not something to force.
VISUAL_CHECKS = (
    "non_blanching_rash", "widespread_hives",
    "bleed_life_or_limb", "bleed_moderate_minor",
)

try:
    from pydantic import BaseModel, Field

    class PhotoFields(BaseModel):
        caption: str = Field(
            description="One or two plain sentences describing what the photo shows, for a "
                        "nurse to quickly read and check against the patient -- location, "
                        "appearance, colour, size if judgable, any visible bleeding or "
                        "swelling. Not a diagnosis.")
        candidate_checks: list[str] = Field(
            default_factory=list,
            description=f"Zero or more of exactly these ids: {', '.join(VISUAL_CHECKS)}. "
                        f"Only include one if the photo unambiguously shows it -- leave "
                        f"empty rather than force a fit.")

    _HAVE_PYDANTIC = True
except ImportError:  # pragma: no cover - offline install
    _HAVE_PYDANTIC = False


@dataclass(frozen=True, slots=True)
class PhotoAnalysis:
    caption: str = ""
    candidate_checks: tuple[str, ...] = ()
    tier: str = "none"
    available: bool = True
    detail: str = ""


class Unavailable(Exception):
    """Neither vision tier reachable. Caller reports the feature as down."""


_SYSTEM = (
    "You are looking at a photo a patient took of a visible symptom (a wound, "
    "rash, swelling, bruising, or similar) while checking in at an emergency "
    "department. Describe plainly what the photo shows -- location, "
    "appearance, colour, approximate size if judgable, any visible bleeding "
    "or swelling -- in language a nurse can quickly read and verify against "
    "the patient in front of them. This is a description for a human to "
    "confirm, never a diagnosis, and never a triage category.\n\n"
    f"Only propose one of these check ids if the photo unambiguously shows "
    f"it: {', '.join(VISUAL_CHECKS)}. Leave candidate_checks empty otherwise."
)


def _client():
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise Unavailable("anthropic SDK not installed") from exc
    try:
        return anthropic.Anthropic()
    except Exception as exc:  # pragma: no cover - credential resolution
        raise Unavailable(str(exc)) from exc


def _clean_checks(raw) -> tuple[str, ...]:
    return tuple(c for c in (raw or []) if c in VISUAL_CHECKS)


# --------------------------------------------------------------------------
# tier 1 -- hosted
# --------------------------------------------------------------------------

def _analyze_hosted(image_b64: str, mime: str) -> PhotoAnalysis:
    client = _client()
    try:
        response = client.messages.parse(
            model=MODEL, max_tokens=512,
            system=_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": image_b64}},
                    {"type": "text", "text": "Describe what this photo shows."},
                ],
            }],
            output_format=PhotoFields,
        )
    except Exception as exc:  # network, rate limit, refusal, missing credentials
        raise Unavailable(str(exc)) from exc

    f: PhotoFields = response.parsed_output
    return PhotoAnalysis(caption=f.caption.strip(), candidate_checks=_clean_checks(f.candidate_checks),
                          tier="hosted")


# --------------------------------------------------------------------------
# tier 2 -- a model on this machine
# --------------------------------------------------------------------------

def _analyze_local(image_b64: str) -> PhotoAnalysis:
    from . import ollama

    ok, why = ollama.available(model=OLLAMA_VISION_MODEL)
    if not ok:
        raise Unavailable(why)

    schema = {
        "type": "object",
        "properties": {
            "caption": {"type": "string"},
            "candidate_checks": {"type": "array", "items": {"type": "string", "enum": list(VISUAL_CHECKS)}},
        },
        "required": ["caption", "candidate_checks"],
    }
    try:
        raw = ollama.chat_vision(_SYSTEM, "Describe what this photo shows.", image_b64, schema,
                                  model=OLLAMA_VISION_MODEL)
    except ollama.Unreachable as exc:
        raise Unavailable(str(exc)) from exc

    return PhotoAnalysis(caption=(raw.get("caption") or "").strip(),
                          candidate_checks=_clean_checks(raw.get("candidate_checks")), tier="local")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def analyze_photo(image_bytes: bytes, mime: str, *, use_model: bool = True) -> PhotoAnalysis:
    """Photo in, a proposed caption (and maybe a candidate check) out.

    No tier 3: there is no keyword extraction that can look at a picture.
    Both tiers unreachable is reported as `available=False` with a plain
    reason, not a caption invented from nothing.
    """
    if not use_model or not _HAVE_PYDANTIC:
        return _fall_back(image_bytes, mime, "model extraction turned off")

    image_b64 = base64.b64encode(image_bytes).decode()
    try:
        return _analyze_hosted(image_b64, mime)
    except Unavailable as exc:
        return _fall_back(image_bytes, mime, str(exc), image_b64=image_b64)


def _fall_back(image_bytes: bytes, mime: str, why: str, *, image_b64: str | None = None) -> PhotoAnalysis:
    image_b64 = image_b64 or base64.b64encode(image_bytes).decode()
    try:
        return _analyze_local(image_b64)
    except Unavailable as exc:
        return PhotoAnalysis(
            available=False,
            detail=f"no vision model reachable -- hosted: {why}; local: {exc}",
        )
