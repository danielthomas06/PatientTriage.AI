"""Deterministic scoring.

The whole engine rests on one observation about how these protocols compose.

Within a branch, the first matching discriminator wins -- and because the list
is ordered most-urgent-first, that is the *most urgent* matching discriminator
on that branch. Across plausible branches we take the most urgent result. So
the two rules collapse:

    acuity = most urgent category among all discriminators that are positive
             and appear on at least one plausible branch

Ordering within a branch stops mattering once you take a max across branches,
which turns a fifty-way tree walk into a single pass over a vocabulary. That is
why evaluating every branch at once costs no more than evaluating one, and why
this runs offline in microseconds.
"""

from dataclasses import dataclass
from math import prod

from .belief import BeliefState
from .core import LEAST_URGENT, Answer, Category, Confidence, Protocol
from .news2 import Score


def effective_categories(
    protocol: Protocol, plausible: frozenset[str]
) -> dict[str, Category]:
    """Most urgent category each discriminator carries across plausible branches.

    A discriminator absent from every plausible branch does not appear, which
    is how narrowing the plausible set narrows what can fire.

    Iterates `sorted(plausible)` rather than the frozenset directly. The
    result dict's CONTENT does not depend on visitation order -- every
    entry is picked by an explicit min-comparison, not "last write wins" --
    but its INSERTION order does, and that order propagates downstream into
    `acuity_distribution`'s per-level products. Floating-point
    multiplication is not exactly order-independent, so two genuinely-tied
    candidates in `voi.rank` could come out a few ULPs apart depending on
    which order Python's per-process hash randomisation happened to iterate
    a frozenset in -- close enough to be invisible, and exactly enough to
    flip which one sorts first. Caught as a real, reproducible test flake:
    same encounter, same input, different question order, only fixed by
    pinning PYTHONHASHSEED -- which confirmed the cause rather than papering
    over the symptom. Reproducibility is a stated property of this system
    (see ollama.py's temperature=0), and it has to hold here too.
    """
    out: dict[str, Category] = {}
    for branch_id in sorted(plausible):
        for did, cat in protocol.branches[branch_id].rules:
            current = out.get(did)
            if current is None or cat < current:  # lower value = more urgent
                out[did] = cat
    return out


def plausible_set(
    branch_weights: dict[str, float], tau: float = 0.05
) -> frozenset[str]:
    """Branches still worth evaluating.

    Deliberately loose. Because the final category is a max over this set, a
    generous threshold over-triages rather than under-triages, and the system
    never needs a well-calibrated posterior over branches -- only a set that
    contains the right one.
    """
    return frozenset(b for b, w in branch_weights.items() if w > tau)


# Below this, a category's probability is treated as noise rather than a real
# possibility. Deliberately low: the cost of naming a worst case that does not
# materialise is a second look, and the cost of missing one is the whole problem.
NOISE = 0.02


def _confidence(
    belief: BeliefState, plausible: frozenset[str], assigned: Category
) -> Confidence:
    dist = acuity_distribution(belief, plausible)
    worst = min(
        (c for c, p in dist.items() if p > NOISE and c < assigned), default=assigned
    )
    unresolved = sum(
        1 for d in effective_categories(belief.protocol, plausible)
        if not belief.is_observed(d)
    )
    gap = int(assigned) - int(worst)
    band = "HIGH" if gap == 0 else ("MODERATE" if gap == 1 else "LOW")
    return Confidence(
        band=band,
        assigned=assigned,
        worst_case=worst,
        p_assigned=dist.get(assigned, 0.0),
        unresolved=unresolved,
    )


@dataclass(frozen=True, slots=True)
class Decision:
    category: Category
    fired: str | None           # discriminator id that set the category
    fired_on: tuple[str, ...]   # branches carrying it at that urgency
    from_vitals: bool           # True if the vital-sign score drove it
    confidence: Confidence      # required -- see Confidence for why

    def explain(self, protocol: Protocol) -> str:
        if self.from_vitals:
            return f"{self.category.label} -- driven by the vital-sign score"
        if self.fired is None:
            return f"{self.category.label} -- no discriminator positive"
        text = protocol.discriminators[self.fired].text
        names = ", ".join(protocol.branches[b].name for b in self.fired_on)
        return f"{self.category.label} -- '{text}' (on: {names})"


def _vitals_category(score: Score | None) -> Category | None:
    """Policy mapping from vital-sign score to a floor on urgency.

    A local policy choice, not part of the published score. Kept explicit and
    tunable rather than buried, so a department can set its own tolerance.
    """
    if score is None:
        return None
    match score.band:
        case "high":
            return Category.ORANGE
        case "medium":
            return Category.ORANGE
        case "low-medium":
            return Category.YELLOW
        case _:
            return None


def decide(
    belief: BeliefState,
    plausible: frozenset[str],
    vitals: Score | None = None,
    *,
    floor: Category | None = None,
) -> Decision:
    """The priority. Uses only confirmed positives -- never a probability.

    Two ways a vital-sign reading alone can set the category, and they are
    deliberately separate parameters rather than one overloaded shape.
    `vitals` is a NEWS2 `Score`, valid only for adults, converted to a
    category floor by `_vitals_category`'s policy table. `floor` is an
    already-resolved `Category` from any other age-appropriate score --
    paediatric vitals, scored against the CTAS appendix rather than NEWS2,
    have no `Score` object to build (see `cohort.paediatric_vital_category`).
    Both, if given, apply the same way: as a floor under the belief-driven
    category, never a ceiling above it, and the more urgent of the two wins
    if somehow both are supplied.
    """
    cats = effective_categories(belief.protocol, plausible)

    best: Category = LEAST_URGENT
    fired: str | None = None
    for did, cat in cats.items():
        if belief.answers.get(did) is Answer.TRUE and cat < best:
            best, fired = cat, did

    vitals_cat = _vitals_category(vitals)
    floor_cat = min((c for c in (vitals_cat, floor) if c is not None), default=None)
    if floor_cat is not None and floor_cat < best:
        return Decision(
            category=floor_cat, fired=None, fired_on=(), from_vitals=True,
            confidence=_confidence(belief, plausible, floor_cat),
        )

    on = ()
    if fired is not None:
        on = tuple(
            b for b in sorted(plausible)
            if belief.protocol.branches[b].category_for(fired) == best
        )
    return Decision(
        category=best, fired=fired, fired_on=on, from_vitals=False,
        confidence=_confidence(belief, plausible, best),
    )


def acuity_distribution(
    belief: BeliefState, plausible: frozenset[str]
) -> dict[Category, float]:
    """Probability of each category, given what is still unknown.

    Uses the collapse described in the module docstring: group discriminators
    by the urgency they carry, then walk the levels from most to least urgent.
    P(acuity = L) is P(nothing more urgent fires) x P(at least one at L fires).

    Discriminators are treated as independent. That is a modelling assumption,
    stated here rather than hidden -- and the max-over-plausible rule keeps its
    consequences on the over-triage side.
    """
    cats = effective_categories(belief.protocol, plausible)

    by_level: dict[Category, list[str]] = {}
    for did, cat in cats.items():
        by_level.setdefault(cat, []).append(did)

    dist: dict[Category, float] = {}
    nothing_worse = 1.0
    for level in sorted(by_level):  # most urgent first
        all_negative = prod(1.0 - belief.p_true(d) for d in by_level[level])
        dist[level] = nothing_worse * (1.0 - all_negative)
        nothing_worse *= all_negative

    dist[LEAST_URGENT] = dist.get(LEAST_URGENT, 0.0) + nothing_worse
    return {k: v for k, v in dist.items() if v > 1e-12}


def most_likely(dist: dict[Category, float]) -> Category:
    return max(dist.items(), key=lambda kv: kv[1])[0]
