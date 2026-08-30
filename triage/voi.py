"""Choosing what to find out next.

Not "which question is important" but "which question would most change what
we do" -- and, because ranking someone too low is far more costly than too
high, weighted so that questions capable of revealing danger win over
questions that merely reduce uncertainty.

A question with a perfectly predictable answer is worthless no matter how
serious its subject. A question whose every answer leaves the category
unchanged is worthless no matter how uncertain it is. Both fall out of the
maths rather than being special-cased.
"""

from dataclasses import dataclass
from functools import cache

from .belief import BeliefState
from .core import COST, Answer, Category, Protocol
from .engine import acuity_distribution, decide


UNDER_TRIAGE_PENALTY = 10.0
OVER_TRIAGE_PENALTY = 2.5


@cache
def loss(assigned: Category, actual: Category) -> float:
    """Cost of assigning `assigned` when the truth is `actual`.

    Category values increase as urgency decreases, so a positive difference
    means we ranked someone below where they belonged.

    Both directions are superlinear, and over-triage is deliberately not free.
    That is clinically true -- calling everyone Immediate in a full department
    displaces someone who needed the bay -- and it is also what makes the
    selector work at all. Information has value only when it can change the
    decision, so a loss lopsided enough to pin the answer at Immediate no
    matter what would drive every question's value to exactly zero. Asymmetric,
    not absolute.
    """
    bands = assigned - actual
    if bands == 0:
        return 0.0
    if bands > 0:
        return UNDER_TRIAGE_PENALTY ** bands
    return OVER_TRIAGE_PENALTY ** abs(bands)


def bayes_risk(dist: dict[Category, float]) -> float:
    """Expected loss under the best category we could assign right now."""
    return min(
        sum(p * loss(candidate, actual) for actual, p in dist.items())
        for candidate in Category
    )


@dataclass(frozen=True, slots=True)
class Candidate:
    discriminator_id: str
    value: float        # risk reduction per unit cost -- the ranking key
    raw_gain: float     # risk reduction before cost is applied
    cost: float
    can_escalate: bool  # True if some answer would raise the category

    def describe(self, protocol: Protocol) -> str:
        d = protocol.discriminators[self.discriminator_id]
        tag = "" if self.can_escalate else "   [cannot change the band]"
        return f"{self.value:6.3f}  {d.question}{tag}"


def value_of(
    belief: BeliefState, plausible: frozenset[str], discriminator_id: str
) -> tuple[float, bool]:
    """Expected reduction in risk from answering this check.

    Returns (gain, can_escalate). Both hypothetical answers are priced and
    weighted by how likely they are -- that is the "expected" part, and it is
    what makes a question whose answer we can already guess score near zero.
    """
    before = acuity_distribution(belief, plausible)
    risk_before = bayes_risk(before)

    p = belief.p_true(discriminator_id)

    if_true = acuity_distribution(belief.hypothetical(discriminator_id, Answer.TRUE), plausible)
    if_false = acuity_distribution(belief.hypothetical(discriminator_id, Answer.FALSE), plausible)

    expected_after = p * bayes_risk(if_true) + (1.0 - p) * bayes_risk(if_false)

    # "Could a yes actually move the band?" is asked against the deterministic
    # decision, not the distribution -- otherwise the long tail of low-prior
    # life-threat checks makes every question look like it could escalate.
    now = decide(belief, plausible).category
    if_yes = decide(belief.hypothetical(discriminator_id, Answer.TRUE), plausible).category
    return risk_before - expected_after, if_yes < now


def rank(
    belief: BeliefState,
    plausible: frozenset[str],
    *,
    branch_weights: dict[str, float] | None = None,
    limit: int | None = None,
) -> list[Candidate]:
    """Every unanswered check, best first.

    Cost carries more than time: a record lookup is nearly free and should be
    resolved before anyone is asked anything, while a question that is painful
    to ask in a public waiting room has to earn its place.

    `branch_weights` is optional and affects ORDER only. `plausible_set()`
    keeps every branch above its floor in play -- deliberately, so a weak
    extraction never wrongly rules one out -- but membership there is binary,
    and a branch sitting at the bare floor competed for the nurse's next
    question on exactly equal footing with the one the patient is actually
    describing. That produced real cases: a headache narrative surfacing a
    question that only matters for abdominal pain, because nothing in the
    ranking discounted it for being unrelated. Passing weights here scales a
    check DOWN toward branches the extraction thinks are less likely, without
    excluding anything -- the same "wider is safer" property, just no longer
    blind to *how much* wider. `raw_gain` is untouched by this, on purpose:
    `should_stop` reads it to decide whether anything can still change the
    category, and that call has to stay exact regardless of which branch a
    check happens to sit on.
    """
    top_weight = max(branch_weights.values(), default=1.0) if branch_weights else 1.0
    carriers = _carriers(belief.protocol)

    out: list[Candidate] = []
    for did in belief.unobserved():
        if did not in _reachable(belief.protocol, plausible):
            continue
        gain, escalates = value_of(belief, plausible, did)
        c = COST[belief.protocol.discriminators[did].source]

        relevance = 1.0
        if branch_weights and top_weight > 0:
            carrier_weight = max(
                (branch_weights.get(b, 0.0) for b in carriers[did] if b in plausible),
                default=top_weight,
            )
            # Cubed, not linear. A live case exposed why linear wasn't enough:
            # a dizzy patient, extraction only confidently identified vertigo
            # (0.5 against a 0.06 floor -- a real 8x margin), and a chest-pain
            # -only check still nearly kept pace, because a linear 0.12
            # barely dents a check whose raw value is already large. The
            # branch that's actually plausible should not have to out-argue
            # an unrelated one on value alone -- it should just win, unless
            # nothing plausible has anything left to ask. Cubing (0.12 -> ~
            # 0.0017) does that while leaving `test_nothing_is_ever_fully_
            # excluded` intact: a floor branch's only check still clears
            # zero, so it is still askable once every plausible branch runs
            # dry, just no longer competitive while one hasn't.
            relevance = (carrier_weight / top_weight) ** 3

        out.append(
            Candidate(
                discriminator_id=did,
                value=(gain / c) * relevance,
                raw_gain=gain,
                cost=c,
                can_escalate=escalates,
            )
        )
    out.sort(key=lambda c: (-c.value, c.cost, c.discriminator_id))
    return out[:limit] if limit else out


def _reachable(protocol: Protocol, plausible: frozenset[str]) -> set[str]:
    """Checks that appear on at least one plausible branch."""
    return {
        did
        for b in plausible
        for did in protocol.branches[b].discriminator_ids
    }


def _carriers(protocol: Protocol) -> dict[str, tuple[str, ...]]:
    """Every branch each check appears on, protocol-wide.

    Not cached: `Protocol` holds plain dicts (branches, discriminators), which
    are unhashable, so `functools.cache` would raise on the first call rather
    than memoise it. Left uncached rather than worked around -- this is a
    single pass over a few dozen discriminators, called a handful of times per
    HTTP request, nowhere near hot enough to justify the extra machinery.

    Unfiltered by `plausible` on purpose. A check on every branch (the
    life-threat and general discriminators) naturally picks up the current
    top weight wherever it lands, which is why those keep outranking a
    branch-specific question exactly as before -- nothing here singles them
    out, the general shape of the protocol already does.
    """
    out: dict[str, list[str]] = {}
    for b in protocol.branches.values():
        for did in b.discriminator_ids:
            out.setdefault(did, []).append(b.id)
    return {did: tuple(bs) for did, bs in out.items()}


def should_stop(
    belief: BeliefState,
    plausible: frozenset[str],
    asked: int,
    max_questions: int = 8,
    floor: float = 1e-3,
) -> str | None:
    """Reason to stop, or None to keep going.

    The first rule matters clinically: once someone is Immediate or Very
    urgent, you alert and stop. A system that keeps interviewing a dying
    patient is worse than no system.
    """
    current = decide(belief, plausible)
    if current.category <= Category.ORANGE:
        return f"escalated to {current.category.label} -- alert and stop"
    if asked >= max_questions:
        return "question budget exhausted"
    best = rank(belief, plausible, limit=1)
    if not best or best[0].raw_gain < floor:
        return "nothing further can change the category"
    return None
