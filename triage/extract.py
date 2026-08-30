"""The model layer.

Three bounded calls, each with a fixed contract:

    seed(transcript)      narrative      -> risk checks + branch weights
    render(check, ctx)    check id       -> a question for this patient
    parse(check, reply)   free text      -> true / false / unclear

The solver decides *what* to ask. This module only phrases it and reads the
answer back. That separation is what keeps question choice reproducible: only
the wording varies between runs, and varying wording is harmless.

Two guards make the output safe to trust:

  * The schema has no "unknown" case. A check the model did not hear is simply
    absent from the response, so it cannot emit a fabricated negative -- the
    single most dangerous output this system could produce.
  * Every finding carries a verbatim quote, and `_verify` checks that the quote
    actually occurs in the transcript. Findings whose evidence cannot be found
    are dropped, not trusted. Hallucinated evidence fails closed.
"""

import os
import re
from dataclasses import dataclass, field

from .belief import BeliefState, Evidence
from .core import Answer, Protocol, Source

MODEL = os.environ.get("TRIAGE_MODEL", "claude-opus-5")
"""Extraction sits on the critical path, so this is the obvious place to trade
capability for latency -- but that is a clinical call, not ours. Override with
TRIAGE_MODEL to try a smaller model and measure the under-triage cost."""


BRANCH_FLOOR = 0.06
"""Weight every branch carries before the model says anything.

Just above the 0.05 plausibility threshold, so nothing is silently excluded. See
`_accept` for why the model is allowed to rank branches but not to drop them."""


class Unavailable(Exception):
    """No model reachable. Callers fall back to `keyword_seed`."""


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------

try:
    from pydantic import BaseModel, Field

    class Finding(BaseModel):
        check_id: str = Field(description="Exact id from the check list.")
        present: bool = Field(
            description=(
                "true if the patient said this IS the case; false only if they "
                "explicitly denied it. If it was not mentioned, omit the check "
                "entirely -- do not guess."
            )
        )
        quote: str = Field(
            description="Verbatim words from the transcript. Copy exactly; do not paraphrase."
        )

    class BranchGuess(BaseModel):
        branch_id: str = Field(description="Exact id from the branch list.")
        weight: float = Field(description="0 to 1. Include any branch worth keeping open.")

    class Seed(BaseModel):
        findings: list[Finding]
        branches: list[BranchGuess]

    class ParsedAnswer(BaseModel):
        outcome: str = Field(description='One of "yes", "no", "unclear".')
        quote: str = Field(description="Verbatim words the outcome rests on. Empty if unclear.")

    class NextQuestion(BaseModel):
        check_id: str = Field(description="Exact id, copied from the shortlist. Never a check "
                                           "outside it.")
        question: str = Field(description="Plain, non-leading question for the patient.")
        why: str = Field(description="One short clinical sentence, for the nurse, not the "
                                      "patient: why this, why now.")

    _HAVE_PYDANTIC = True
except ImportError:  # pragma: no cover - offline install
    _HAVE_PYDANTIC = False


# --------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Extraction:
    belief: BeliefState
    branch_weights: dict[str, float]
    rejected: list[str] = field(default_factory=list)
    """Findings dropped because their quote was not in the transcript, or
    because they named a check or branch that does not exist. Non-empty is a
    signal worth monitoring -- it is the model failing closed."""

    degraded: bool = False
    """True when this came from the offline keyword fallback rather than a model."""

    fallbacks: list[str] = field(default_factory=list)
    """Why the ladder dropped a rung. Routine, not a fault -- kept apart from
    `rejected` because a missing API key is not the guard refusing to believe a
    finding, and showing them together makes normal operation look broken."""

    tier: str = "hosted"
    """Which rung of the degradation ladder produced this: hosted | local |
    keyword. Worth surfacing rather than hiding -- a nurse should be able to see
    that the tool is running degraded, and the evaluation should be able to
    measure what each rung costs."""


# --------------------------------------------------------------------------
# prompt construction -- stable, therefore cacheable
# --------------------------------------------------------------------------

def _vocabulary(protocol: Protocol) -> str:
    checks = "\n".join(
        f"  {d.id}: {d.text}"
        for d in sorted(protocol.discriminators.values(), key=lambda d: d.id)
    )
    branches = "\n".join(
        f"  {b.id}: {b.name}" for b in sorted(protocol.branches.values(), key=lambda b: b.id)
    )
    return f"CHECKS\n{checks}\n\nBRANCHES\n{branches}"


_SEED_SYSTEM = """\
You convert what an emergency patient said into structured risk checks.

You are not assessing the patient and you are not deciding their priority. A \
deterministic engine does that from the checks you fill in. Your only job is \
to record what was actually said.

Rules, in order of importance:

1. Only report a check if the patient's own words support it. If a check was \
not mentioned, leave it out. An omitted check is read as "unknown", which is \
correct and safe. A check you report as false is read as "the patient denied \
this", which is not safe unless they did.
2. Every finding needs a verbatim quote copied exactly from the transcript. Do \
not paraphrase, tidy, or complete a fragment. If you cannot quote it, do not \
report it.
3. Report `present: false` only for an explicit denial ("no chest pain", \
"never had that before"). Never infer a denial from silence.
4. Keep branches generous. Include any branch that is plausibly in play, even \
at low weight -- a branch left out cannot fire a danger signal later. Being \
wrong about the branch is cheap; excluding the right one is not.
5. Some checks are not yours to fill. Oxygen saturation, pulse and temperature \
come from a device. Level of consciousness and airway come from a clinician \
looking at the patient. A patient cannot state their own saturation, and someone \
speaking in sentences is self-evidently not unresponsive. Leave those out \
entirely -- a nurse or a monitor will answer them.

Reference vocabulary follows. Use these ids exactly.

"""

_RENDER_SYSTEM = """\
You turn one clinical check into one question for one patient.

Write a single question, plain spoken, at a reading age of about eleven. No \
preamble, no explanation, no clinical jargon, no reassurance, no indication of \
how serious the answer might be. Return only the question.

Match the patient's language. If they are in pain or distressed, keep it \
shorter.\
"""

_CHOOSE_SYSTEM = """\
You are continuing a triage interview, deciding what to ask the patient next.

You do not decide how urgent this patient is. A separate, fixed process does \
that from whatever checks get confirmed -- your only job here is choosing \
what to ask next, out of a shortlist you are given, and phrasing it plainly.

Rules, in order of importance:

1. Choose exactly one check id, copied exactly from the shortlist below. \
Never invent one and never choose one that is not on the list.
2. Prefer staying with the complaint the conversation is already about. Only \
choose something from a different domain when it is a genuine safety rule-out \
that has not been covered yet -- and say so plainly in your reason.
3. Do not repeat the substance of something already asked, even reworded.
4. Write the question in plain, spoken language, at a reading age of about \
eleven. Never name or hint at the specific answer it is testing for.
5. Give one short, plain sentence for why this is worth asking now -- written \
for the nurse reading it, not the patient.
"""

_PARSE_SYSTEM = """\
You read a patient's reply and decide whether it answers a clinical check.

Decide from what they actually said. If the reply directly answers the check, \
that IS a clear answer -- give it, and quote the words. "About a six" answers a \
nought-to-ten question. "A bit, into my jaw" answers a question about where pain \
spreads.

Answer "unclear" when the reply genuinely does not settle the question: they did \
not understand, changed the subject, said they did not know, or hedged without \
content. Do not answer unclear merely because the reply was informal or partial.

Answer "no" only for an explicit denial. Silence and "I don't know" are unclear, \
not no.

Quote the patient's own words that the outcome rests on.\
"""

# Tuning note. This previously opened with "Answer 'unclear' freely." Right in
# spirit, wrong in practice: a 7B reads it as "always say unclear" and refused to
# map "about a six" onto moderate pain, or "a bit, into my jaw" onto cardiac
# pain -- the one finding the whole worked example turns on. A/B over six cases
# went 5/6 to 6/6, with no regression on the four that should stay unclear. The
# bias against guessing still belongs here; it just has to be scoped to genuine
# ambiguity rather than stated as a blanket preference.


def _client():
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise Unavailable("anthropic SDK not installed") from exc
    try:
        return anthropic.Anthropic()
    except Exception as exc:  # pragma: no cover - credential resolution
        raise Unavailable(str(exc)) from exc


# --------------------------------------------------------------------------
# evidence verification
# --------------------------------------------------------------------------

def _normalise(text: str) -> str:
    # Apostrophes are deleted rather than spaced, so a model that writes "Its"
    # for the transcript's "It's" still matches. Every other mark becomes a
    # space, which keeps word boundaries honest.
    text = re.sub(r"['’ʼ]", "", text.lower())
    return re.sub(r"[^a-z0-9 ]+", " ", text)


# Only with one of these present in the quoted words is a negative admissible.
# Otherwise "not mentioned" has been silently converted into "no".
_DENIAL = {
    "no", "not", "none", "never", "denies", "denied", "without", "nothing",
    "isnt", "wasnt", "doesnt", "didnt", "havent", "hasnt", "dont", "cant",
    "negative", "nope", "neither", "nor",
}


def _denies(quote: str) -> bool:
    """Does the quoted text actually contain a denial?

    Coarse on purpose, and worth knowing where it is coarse: it checks that a
    denial word appears, not that the denial scopes over the right thing. "It's
    not going away" would license a negative about radiation. It catches the
    egregious case -- a negative asserted from plainly positive evidence -- and
    not a mis-attributed one.
    """
    return bool(set(_normalise(quote).split()) & _DENIAL)


_PAIN_LANGUAGE = re.compile(
    r"\b(\d{1,2})\s*(?:/|out of)\s*10\b|"
    r"\b(worst|unbearable|agonis\w*|excruciat\w*|severe|terrible|"
    r"bad|nasty|really hurts?|quite sore|mild|slight|bit sore|niggl\w*)\b",
    re.IGNORECASE,
)

PAIN_CHECKS = frozenset({"severe_pain", "moderate_pain", "mild_pain"})


def _rates_pain(quote: str) -> bool:
    """Does the quote actually say how bad the pain is, not just that it exists?

    Coarse in the same way `_denies` is coarse: it checks that intensity
    language is present, not that it lines up with the specific band the model
    picked -- "a mild ache" would license `moderate_pain` too. It catches the
    case that actually happened: a model quoting "I've been having stomach
    pain" -- true, verbatim, and silent on severity -- as evidence for
    mild_pain=true.
    """
    return bool(_PAIN_LANGUAGE.search(quote))


def _verify(quote: str, transcript: str) -> bool:
    """Does this quote actually occur in what the patient said?

    Deliberately lenient about punctuation and spacing -- transcripts are
    messy and the model may normalise an apostrophe. Not lenient about words:
    a quote whose words are not in the transcript is fabricated evidence, and
    the finding is dropped.
    """
    q = " ".join(_normalise(quote).split())
    t = " ".join(_normalise(transcript).split())
    return bool(q) and q in t


# --------------------------------------------------------------------------
# shared validation
# --------------------------------------------------------------------------

def _accept(findings, branch_guesses, transcript: str, protocol: Protocol,
            tier: str) -> Extraction:
    """Turn raw model output into a belief state, dropping anything unsupported.

    Every tier goes through here. That is the point: a weaker model produces more
    paraphrases and more unsupported negatives, and those are rejected rather than
    believed -- so a cheaper model costs throughput, never safety.
    """
    belief = BeliefState(protocol)
    rejected: list[str] = []

    for f in findings:
        check_id, present, quote = f["check_id"], f["present"], f["quote"]
        if check_id not in protocol.discriminators:
            rejected.append(f"unknown check {check_id!r}")
            continue

        # A patient cannot state their own oxygen saturation, and someone typing
        # coherently is self-evidently not unresponsive. Measurements come from a
        # device and observations come from a clinician looking at the patient --
        # neither is the model's to fill from a sentence, however plausible its
        # guess. RECORD stays allowed: "I take warfarin" is a legitimate patient
        # report, it just carries less weight than the actual record.
        source = protocol.discriminators[check_id].source
        if source in (Source.MEASURE, Source.OBSERVE):
            rejected.append(
                f"{check_id}: {source.value} check cannot be filled from narrative "
                f"-- needs a device or a clinician"
            )
            continue
        if not _verify(quote, transcript):
            rejected.append(f"{check_id}: quote not in transcript ({quote!r})")
            continue
        if check_id in PAIN_CHECKS and present and not _rates_pain(quote):
            rejected.append(
                f"{check_id}: quote says pain is present, not how severe -- "
                f"'{quote}' rates nothing"
            )
            continue
        if not present and not _denies(quote):
            rejected.append(
                f"{check_id}: negative asserted without a denial in the evidence ({quote!r})"
            )
            continue
        belief = belief.record(
            check_id, Answer.TRUE if present else Answer.FALSE, Evidence(quote, "speech")
        )

    # Every branch starts just above the plausibility threshold and the model can
    # only raise it. This is deliberate and it is a safety property, not a
    # convenience: a branch the model omits is a branch that can never fire a
    # danger signal, and small models omit plenty. Ranking is useful; excluding
    # is not the model's decision to make.
    #
    # The cost is over-triage, which is the direction we choose to be wrong in.
    weights: dict[str, float] = {b: BRANCH_FLOOR for b in protocol.branches}
    named = 0
    for b in branch_guesses:
        if b["branch_id"] not in protocol.branches:
            rejected.append(f"unknown branch {b['branch_id']!r}")
            continue
        weights[b["branch_id"]] = max(BRANCH_FLOOR, min(1.0, float(b["weight"])))
        named += 1

    if not named:
        rejected.append("no usable branches returned; every branch left at the floor")

    # Backstop, not a replacement: a paraphrase the model reads as one branch
    # ("pain at the back of my head" -> Unwell adult, observed against a local
    # model) can still surface the branch a plain regex would have caught.
    # Only ever raises a weight -- never excludes what the model named.
    for pattern, branch_id in _BRANCH_HINTS:
        if branch_id in weights and re.search(pattern, transcript, re.IGNORECASE):
            weights[branch_id] = max(weights[branch_id], 0.5)

    return Extraction(belief=belief, branch_weights=weights, rejected=rejected, tier=tier)


# --------------------------------------------------------------------------
# tier 2 -- a model on this machine
# --------------------------------------------------------------------------

def _seed_local(transcript: str, protocol: Protocol) -> Extraction:
    """Same guards, same vocabulary, no network. Raises Unavailable to fall through."""
    from . import ollama

    ok, why = ollama.available()
    if not ok:
        raise Unavailable(why)

    schema = {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "check_id": {"type": "string",
                                     "enum": sorted(protocol.discriminators)},
                        "present": {"type": "string", "enum": ["true", "false"]},
                        "quote": {"type": "string"},
                    },
                    "required": ["check_id", "present", "quote"],
                },
            },
            "branches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "branch_id": {"type": "string", "enum": sorted(protocol.branches)},
                        "weight": {"type": "number"},
                    },
                    "required": ["branch_id", "weight"],
                },
            },
        },
        "required": ["findings", "branches"],
    }

    try:
        raw = ollama.chat(
            _SEED_SYSTEM,
            f"{ollama.vocabulary(protocol)}\n\nTranscript:\n{transcript}",
            schema,
        )
    except ollama.Unreachable as exc:
        raise Unavailable(str(exc)) from exc

    findings = [
        {"check_id": f.get("check_id"),
         "present": ollama.coerce(f.get("present")) is True,
         "quote": f.get("quote", "")}
        for f in raw.get("findings") or []
    ]
    branches = [
        {"branch_id": b.get("branch_id"), "weight": b.get("weight", 0)}
        for b in raw.get("branches") or []
    ]
    return _accept(findings, branches, transcript, protocol, tier="local")


# --------------------------------------------------------------------------
# call 1 -- seed
# --------------------------------------------------------------------------

def _plain(reason: str) -> str:
    """Turn an SDK exception into something a nurse could act on."""
    low = reason.lower()
    if "could not resolve authentication" in low or "api_key" in low:
        return "no API key set -- using the local model"
    if "rate" in low and "limit" in low:
        return "hosted model rate limited -- using the local model"
    if "credit" in low or "billing" in low:
        return "hosted account has no credit -- using the local model"
    if "connection" in low or "timed out" in low:
        return "hosted model unreachable -- using the local model"
    return reason


def _fall_back(transcript: str, protocol: Protocol, why: str) -> Extraction:
    """Next rung down. Local model if it is there, keywords if it is not."""
    try:
        result = _seed_local(transcript, protocol)
        return Extraction(
            belief=result.belief,
            branch_weights=result.branch_weights,
            rejected=result.rejected,
            fallbacks=[_plain(why)],
            tier="local",
        )
    except Unavailable as exc:
        result = keyword_seed(transcript, protocol)
        return Extraction(
            belief=result.belief,
            branch_weights=result.branch_weights,
            rejected=result.rejected,
            fallbacks=[_plain(why), f"local model unavailable: {exc}"],
            degraded=True,
            tier="keyword",
        )


def seed(transcript: str, protocol: Protocol) -> Extraction:
    """Narrative in, belief state out.

    Walks the degradation ladder rather than failing:

        hosted model  ->  local model  ->  keyword extraction

    Each rung is tried only if the one above is unreachable, and every rung is
    validated by the same guards in `_accept`. Nothing downstream knows or cares
    which one answered, except that `Extraction.tier` says so -- a nurse should
    be able to see the tool is running degraded.
    """
    if not _HAVE_PYDANTIC:
        return _fall_back(transcript, protocol, "pydantic not installed")
    try:
        client = _client()
    except Unavailable as exc:
        return _fall_back(transcript, protocol, f"hosted unavailable: {exc}")

    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": _SEED_SYSTEM + _vocabulary(protocol),
                    # Identical for every patient, so it is a free cache read
                    # after the first arrival of the shift.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": f"<transcript>\n{transcript}\n</transcript>"}],
            output_format=Seed,
        )
    except Exception as exc:  # network, rate limit, refusal, missing credentials
        # Note the SDK raises a bare TypeError when no credential resolves, not
        # an AuthenticationError -- so the message matters more than the type.
        detail = str(exc).strip().splitlines()[0][:160] if str(exc).strip() else type(exc).__name__
        return _fall_back(
            transcript, protocol, f"hosted failed ({type(exc).__name__}): {detail}"
        )

    # One guard for every tier. The hosted model gets no more trust than the
    # local one: same quote verification, same refusal of unsupported negatives.
    parsed: Seed = response.parsed_output
    return _accept(
        [{"check_id": f.check_id, "present": f.present, "quote": f.quote}
         for f in parsed.findings],
        [{"branch_id": b.branch_id, "weight": b.weight} for b in parsed.branches],
        transcript, protocol, tier="hosted",
    )


# --------------------------------------------------------------------------
# tier 2 helpers for render and parse
# --------------------------------------------------------------------------

def _vet_question(text: str, d) -> str:
    """A generated question that names the symptom it tests for plants the answer.
    The canned wording was reviewed; a generated one was not, so on any doubt we
    ship the reviewed phrasing."""
    if not text:
        return d.question
    return d.question if _leads(text, d) else text


def _describe_age(age: float) -> str:
    """Human-readable, not a bare decimal.

    "Patient age: 0.5" is the kind of ambiguous number a model can reason
    straight past -- and did: a 6-month-old was asked about vaginal bleeding
    with the stated reason "a critical safety rule-out for pregnancy
    complications". Age itself already excludes that check from ever
    reaching the model now (see excludes_obstetric_gynae in
    protocols/ctas.py) -- this is the second, independent half of the fix:
    give the model a description an eleven-year-old would also read
    correctly, for every OTHER age-sensitive judgement nothing explicitly
    guards against.
    """
    if age < 2:
        return f"{round(age * 12)} months old (an infant)"
    if age < 16:
        return f"{age:g} years old (a child)"
    return f"{age:g} years old"


def _render_local(d, *, language="English", age=None, distressed=False) -> str:
    from . import ollama

    context = [f"Language: {language}"]
    if age is not None:
        context.append(f"Patient age: {_describe_age(age)}")
    if distressed:
        context.append("The patient is in pain or distressed; keep it very short.")
    if d.leading:
        context.append("Do not use these words: " + ", ".join(d.leading))
    try:
        raw = ollama.chat(
            _RENDER_SYSTEM,
            "Clinical check: " + d.text + "\n" + "\n".join(context)
            + "\n\nWrite the question.",
            {"type": "object", "properties": {"question": {"type": "string"}},
             "required": ["question"]},
        )
    except Exception:
        return d.question
    return _vet_question((raw.get("question") or "").strip(), d)


def _parse_local(d, question: str, reply: str) -> tuple[str, str]:
    from . import ollama

    try:
        raw = ollama.chat(
            _PARSE_SYSTEM,
            # A small model needs the check spelled out, not just named. "Moderate
            # pain (4-6)" plus a reply of "about a six" is a mapping it will not
            # make from the label alone -- so state the decision it has to make.
            "Decide whether the patient's reply means this is TRUE for them.\n\n"
            f"Check: {d.text}\n"
            f"Meaning: {d.question}\n\n"
            f"Question asked: {question}\n"
            f'Patient replied: "{reply}"\n\n'
            'Answer "yes" if the reply means the check is true, "no" if they '
            'denied it, "unclear" only if the reply does not settle it.',
            {"type": "object",
             "properties": {"outcome": {"type": "string", "enum": ["yes", "no", "unclear"]},
                            "quote": {"type": "string"}},
             "required": ["outcome", "quote"]},
        )
    except Exception:
        return "unclear", ""
    return (raw.get("outcome") or "unclear"), (raw.get("quote") or "")


# --------------------------------------------------------------------------
# call 2 -- render
# --------------------------------------------------------------------------

def _leads(question: str, discriminator) -> str | None:
    """Does this question name the thing it is testing for?"""
    low = question.lower()
    for term in discriminator.leading:
        if term in low:
            return term
    return None


def render(
    check_id: str,
    protocol: Protocol,
    *,
    language: str = "English",
    age: int | None = None,
    distressed: bool = False,
) -> str:
    """Phrase one check as a question. Falls back to the canned wording."""
    d = protocol.discriminators[check_id]
    if d.source in (Source.MEASURE, Source.RECORD, Source.OBSERVE):
        return d.question  # nobody is being asked; this is a label for staff

    try:
        client = _client()
        context = [f"Language: {language}"]
        if age is not None:
            context.append(f"Patient age: {_describe_age(age)}")
        if distressed:
            context.append("The patient is in pain or distressed; keep it very short.")

        response = client.messages.create(
            model=MODEL,
            max_tokens=200,
            system=_RENDER_SYSTEM,
            output_config={"effort": "low"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Clinical check: {d.text}\n"
                        f"{chr(10).join(context)}\n\n"
                        "Write the question."
                    ),
                }
            ],
        )
        text = next((b.text for b in response.content if b.type == "text"), "").strip()
        return _vet_question(text, d)
    except Exception:
        return _render_local(d, language=language, age=age, distressed=distressed)


# --------------------------------------------------------------------------
# call 2b -- choose the next question, in context
# --------------------------------------------------------------------------
#
# `render` phrases ONE pre-picked check with no memory of the conversation.
# This picks WHICH of several already-ranked candidates to ask, informed by
# the transcript and what's already been asked -- the reasoning `rank()`'s
# pure expected-risk-reduction math cannot do, because it has no notion of
# "we're already mid-way through this complaint" or "this is a different
# system, but worth ruling out here."
#
# It cannot pick anything outside the shortlist it's given. The shortlist is
# `rank()`'s own top candidates -- already relevance- and cost-adjusted -- so
# this is choosing among vetted options, not reasoning freely over the whole
# vocabulary. And it still never touches urgency: the chosen check becomes a
# question, nothing else, exactly like every other check in the belief state.

def _conversation_block(transcript: list[str], asked_log: dict[str, dict]) -> str:
    lines = ["PATIENT SAID (in their own words):"]
    lines += [f'  "{t}"' for t in transcript] or ["  (nothing yet)"]
    if asked_log:
        lines.append("\nALREADY ASKED THIS ENCOUNTER:")
        for entry in asked_log.values():
            lines.append(f"  Q: {entry['question']}")
            lines.append(f"  A: {entry['answer']}")
    return "\n".join(lines)


def _shortlist_block(shortlist: list[str], protocol: Protocol) -> str:
    lines = ["SHORTLIST -- choose exactly one id from here:"]
    for did in shortlist:
        d = protocol.discriminators[did]
        branches = ", ".join(
            protocol.branches[b].name for b in protocol.branches
            if did in protocol.branches[b].discriminator_ids
        )
        lines.append(f"  {did}: {d.text} (relevant to: {branches})")
    return "\n".join(lines)


def _accept_choice(check_id, question, why, shortlist, protocol):
    """Same discipline as every other model output here: verify or discard.

    A check id outside the offered shortlist is not a creative answer, it's
    an untrusted one -- discard the whole response rather than honour it.
    """
    if check_id not in shortlist:
        return None
    d = protocol.discriminators[check_id]
    return check_id, _vet_question((question or "").strip(), d), (why or "").strip()


def choose_next(
    shortlist: list[str],
    transcript: list[str],
    asked_log: dict[str, dict],
    protocol: Protocol,
    *, language: str = "English", age: int | None = None,
) -> tuple[str, str, str] | None:
    """Pick which shortlisted check to ask next, phrase it, and say why.

    Returns (check_id, question, why), or None if nothing usable came back --
    the caller falls back to the shortlist's own #1 entry (already
    VOI-ranked) with its canned wording, exactly as if this were never
    called.
    """
    if not shortlist:
        return None

    user_msg = (
        _conversation_block(transcript, asked_log) + "\n\n"
        + _shortlist_block(shortlist, protocol)
        + f"\n\nLanguage: {language}"
        + (f"\nPatient age: {_describe_age(age)}" if age is not None else "")
    )

    if _HAVE_PYDANTIC:
        try:
            client = _client()
            response = client.messages.parse(
                model=MODEL, max_tokens=300, system=_CHOOSE_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                output_format=NextQuestion,
            )
            out: NextQuestion = response.parsed_output
            result = _accept_choice(out.check_id, out.question, out.why, shortlist, protocol)
            if result:
                return result
        except Exception:
            pass

    try:
        from . import ollama
        raw = ollama.chat(
            _CHOOSE_SYSTEM, user_msg,
            {"type": "object", "properties": {
                "check_id": {"type": "string"},
                "question": {"type": "string"},
                "why": {"type": "string"},
            }, "required": ["check_id", "question", "why"]},
        )
        return _accept_choice(raw.get("check_id", ""), raw.get("question", ""),
                               raw.get("why", ""), shortlist, protocol)
    except Exception:
        return None


# --------------------------------------------------------------------------
# call 3 -- parse
# --------------------------------------------------------------------------

def parse(check_id: str, question: str, reply: str, protocol: Protocol) -> tuple[Answer, Evidence | None]:
    """Read a reply back onto a check. Unclear answers stay UNKNOWN."""
    d = protocol.discriminators[check_id]
    if not _HAVE_PYDANTIC:
        return keyword_parse(reply)

    try:
        client = _client()
        response = client.messages.parse(
            model=MODEL,
            max_tokens=512,
            system=_PARSE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Clinical check: {d.text}\n"
                        f"Question asked: {question}\n"
                        f"Patient replied: {reply}"
                    ),
                }
            ],
            output_format=ParsedAnswer,
        )
        out: ParsedAnswer = response.parsed_output
        outcome_raw, quote_raw = out.outcome, out.quote
    except Exception:
        outcome_raw, quote_raw = _parse_local(d, question, reply)

    outcome = (outcome_raw or 'unclear').strip().lower()
    if outcome not in ("yes", "no"):
        return keyword_parse(reply)
    if not _verify(quote_raw or '', reply):
        return Answer.UNKNOWN, None  # unverifiable evidence is no evidence
    if outcome == "no" and not _denies(reply):
        return Answer.UNKNOWN, None  # silence and hedging are unclear, not no

    return (
        Answer.TRUE if outcome == "yes" else Answer.FALSE,
        Evidence(quote_raw, "speech"),
    )


# --------------------------------------------------------------------------
# tier 3 -- no model at all
# --------------------------------------------------------------------------

_KEYWORDS: tuple[tuple[str, str], ...] = (
    (r"\b(jaw|arm|neck)\b", "cardiac_pain"),
    (r"\bcan'?t breathe|short of breath|breathless\b", "cannot_complete_sentence"),
    (r"\bvomit|throwing up|been sick\b", "persistent_vomiting"),
    (r"\bblood\b", "minor_haemorrhage"),
    (r"\bfever|temperature|burning up\b", "hot"),
    (r"\bpassed out|blacked out|fainted\b", "history_of_unconsciousness"),
    (r"\bnumb|weak(ness)?|slurr\b", "new_neuro_deficit"),
    (r"\b(warfarin|apixaban|blood thinner)\b", "anticoagulated"),
    (r"\bworst .{0,12}(headache|pain)|thunderclap\b", "thunderclap_headache"),
    (r"\bstiff neck|neck is stiff\b", "neck_stiffness"),
    (r"\bdeformed|bent the wrong way|sticking out\b", "gross_deformity"),
    (r"\bwheez\b", "wheeze"),
    (r"\brigid|board.?like|tender all over\b", "rigid_abdomen"),

    # CTAS-pack ids. Both packs can be active (see PROTOCOL in serve.py vs
    # the illustrative default elsewhere), and the `check_id in
    # protocol.discriminators` guard in `_accept`/`keyword_seed` means an
    # entry for a pack that is not running is simply never used -- these
    # coexist with the illustrative entries above rather than replacing them.
    (r"\b(jaw|arm|neck)\b", "pain_radiating"),
    (r"\bthroat.{0,40}(swell|clos|tight)|tongue.{0,40}swell\b", "throat_or_tongue_swelling"),
    (r"\bchemical.{0,40}eye|eye.{0,40}chemical|splash.{0,20}eye\b", "chemical_eye_exposure"),
    (r"\bcan'?t see|vision (went|is) (dark|blurry|gone)|lost (my )?(vision|sight)\b",
     "sudden_vision_loss"),
    (r"\boverdose|took (too many|lots of|a load of)|swallowed (pills|tablets)\b",
     "high_risk_ingestion"),
)

# Which branch a phrase points at. Without this the fallback leaves every branch
# at the same weight, the plausible set is everything, and the screen reports a
# category "on" an arbitrary list of branches -- which is how a severe headache
# came back reading Abdominal pain. The tier is degraded either way; it should
# not also be misleading.
_BRANCH_HINTS: tuple[tuple[str, str], ...] = (
    (r"\bhead ?ache|migraine|head hurts\b"
     r"|\b(pain|hurts?|aches?|aching|throbbing)\b.{0,25}\bhead\b"
     r"|\bhead\b.{0,25}\b(pain|hurts?|aches?|aching|throbbing)\b", "headache"),
    (r"\bchest\b", "chest_pain"),
    (r"\b(stomach|tummy|abdom|belly|guts)\b", "abdominal_pain"),
    # Excludes "back of my head/neck/hand" -- "back" alone would otherwise
    # also fire this hint for a headache described as pain at the back of
    # the head, which is the paraphrase that motivated widening the pattern
    # above in the first place.
    (r"\bback\b(?!\s+of\b)", "back_pain"),
    (r"\b(ankle|wrist|knee|shoulder|leg|arm|elbow|hip|foot|hand)\b", "limb_problems"),
    (r"\bcan'?t breathe|short of breath|breathless|wheez\b", "breathlessness"),
    (r"\bvomit|throwing up|been sick|nausea|sick\b", "vomiting"),
    (r"\bpassed out|blacked out|fainted|collapsed\b"
     r"|\b(can'?t|not able to|unable to) (stand|sit|stay (up|standing))\b"
     r"|\bfeel(s|ing)? like (i'?m |i |you'?re |you )?(falling|going to fall)\b"
     r"|\b(unsteady|losing (my |your )?balance)\b", "collapse"),
    (r"\bheart (is )?(racing|pounding)|palpitation\b", "palpitations"),
    (r"\bunwell|off legs|not right|generally\b", "unwell_adult"),

    # CTAS-pack branch ids not shared with the illustrative pack's names
    # above (chest_pain, abdominal_pain, headache, vomiting, collapse and
    # palpitations already match by string and need no separate entry).
    (r"\bcan'?t breathe|short of breath|breathless|wheez\b", "shortness_of_breath"),
    (r"\b(ankle|wrist|knee|shoulder|leg|arm|elbow|hip|foot|hand)\b", "extremity_injury"),
    (r"\bunwell|off legs|not right|generally\b", "general_unwell"),
    (r"\bdizzy|dizziness|vertigo|room.{0,10}spinning\b"
     r"|\b(can'?t|not able to|unable to) (stand|sit|stay (up|standing))\b"
     r"|\bfeel(s|ing)? like (i'?m |i |you'?re |you )?(falling|going to fall)\b"
     r"|\b(unsteady|losing (my |your )?balance)\b", "vertigo"),
    (r"\ballerg|hives|anaphyla\b", "allergic_reaction"),
    (r"\brash\b", "rash"),
    (r"\beye\b", "eye_problems"),
    (r"\bdrunk|overdose|intoxicat|withdrawal\b", "substance_misuse"),
    (r"\bvaginal bleed|bleeding.{0,40}pregnan|pregnan.{0,40}bleed\b", "vaginal_bleeding"),
)

# Words a patient uses instead of a number. Not a substitute for asking, but
# better than recording nothing when someone says the worst headache of their
# life and the fallback is all we have.
_INTENSITY: tuple[tuple[str, str], ...] = (
    (r"\b(worst|unbearable|agonis|excruciat|severe|terrible)\b", "severe_pain"),
    (r"\b(bad|nasty|really hurts|quite sore)\b", "moderate_pain"),
    (r"\b(mild|slight|bit sore|niggl)\b", "mild_pain"),
)

_PAIN = re.compile(r"\b(\d{1,2})\s*(?:/|out of)\s*10\b", re.IGNORECASE)


_AFFIRM = frozenset(
    "yes yeah yep yup aye correct right true definitely absolutely "
    "sure certainly mhm uh-huh".split()
)


def keyword_parse(reply: str) -> tuple[Answer, Evidence | None]:
    """Read a reply with no model at all -- the bottom rung of `parse`.

    `parse` had no tier 3. With no key and no local model it fell through to
    UNKNOWN, so on the keyword tier every answer a patient gave was silently
    discarded AFTER paying for a failed round trip. A direct yes or no needs no
    model to read, and the guards that matter still apply: a bare denial is
    required for a negative, and anything hedged stays unknown rather than
    becoming a no.
    """
    words = set(_normalise(reply).split())
    if words & _DENIAL:
        # Both at once ("no, but yes it does spread") is genuinely ambiguous.
        if words & _AFFIRM:
            return Answer.UNKNOWN, None
        return Answer.FALSE, Evidence(reply.strip(), "speech")
    if words & _AFFIRM:
        return Answer.TRUE, Evidence(reply.strip(), "speech")
    return Answer.UNKNOWN, None       # hedging is not an answer


def keyword_seed(transcript: str, protocol: Protocol) -> Extraction:
    """Degraded extraction with no model in the loop.

    Deliberately crude, and honest about it: it fills only what a regex can
    defend, marks itself degraded, and leaves everything else unknown for the
    nurse. It exists so that "the network is down" degrades the assistance
    rather than stopping the queue.
    """
    belief = BeliefState(protocol)

    for pattern, check_id in _KEYWORDS:
        if check_id not in protocol.discriminators:
            continue
        m = re.search(pattern, transcript, re.IGNORECASE)
        if m:
            belief = belief.record(
                check_id, Answer.TRUE, Evidence(m.group(0), "keyword fallback")
            )

    m = _PAIN.search(transcript)
    if not m:
        for pattern, check_id in _INTENSITY:
            hit = re.search(pattern, transcript, re.IGNORECASE)
            if hit and check_id in protocol.discriminators:
                belief = belief.record(
                    check_id, Answer.TRUE, Evidence(hit.group(0), "keyword fallback"))
                break
    if m:
        score = int(m.group(1))
        check = (
            "severe_pain" if score >= 7 else "moderate_pain" if score >= 4 else "mild_pain"
        )
        if check in protocol.discriminators:
            belief = belief.record(check, Answer.TRUE, Evidence(m.group(0), "keyword fallback"))

    # Start every branch at the floor -- nothing is ever excluded -- then raise
    # whichever the phrasing points at. Wider is safer, but a flat set is also
    # uninformative, and the screen has to say something honest about it.
    weights = {b: BRANCH_FLOOR for b in protocol.branches}
    for pattern, branch_id in _BRANCH_HINTS:
        if branch_id in protocol.branches and re.search(pattern, transcript, re.IGNORECASE):
            weights[branch_id] = 0.6
    return Extraction(
        belief=belief,
        branch_weights=weights,
        fallbacks=["no model reachable -- keyword extraction only"],
        degraded=True,
        # Was defaulting to "hosted", so a keyword extraction reported itself as
        # the top tier. A nurse has to be able to see the tool is degraded.
        tier="keyword",
    )
