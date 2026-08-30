"""Synthetic arrivals with ground truth.

Every patient carries a `true_category` -- what a perfect triage would assign
-- alongside the findings a triage process can actually observe. That gap is
the only reason the simulator can measure under-triage at all: real data has
no such label, so a synthetic cohort is the honest way to get one.

Two mechanics carry the whole experiment:

  atypical      the presenting complaint points at a branch that does NOT
                carry the patient's dangerous check. This is the inferior MI
                described as stomach pain.

  deterioration the true category worsens some minutes after arrival, and new
                checks become positive. Nobody is told; it has to be noticed.

Both rates are parameters, not facts. See `Cohort` and the sensitivity sweep
in run_sim.py -- the headline result moves with them, so they are reported
alongside every number rather than buried.
"""

import random
from dataclasses import dataclass, field

from triage import PROTOCOL, Answer, Category


# Roughly a mixed urban department's case mix. Adjustable; reported in output.
DEFAULT_MIX: dict[Category, float] = {
    Category.RED: 0.01,
    Category.ORANGE: 0.10,
    Category.YELLOW: 0.31,
    Category.GREEN: 0.39,
    Category.BLUE: 0.19,
}


@dataclass(frozen=True, slots=True)
class Cohort:
    """Every assumption behind the synthetic population, in one place."""

    n: int = 2000
    arrivals_per_hour: float = 15.0
    """Set above clinician capacity on purpose.

    The brief says "overwhelmed". A department at 70% utilisation has no queue,
    so priority ordering does nothing, nobody waits long enough to deteriorate
    before being seen, and the simulation quietly measures none of the things
    it claims to. Sustained overload is the scenario under test, not an
    accident of parameter choice."""

    mix: dict[Category, float] = field(default_factory=lambda: dict(DEFAULT_MIX))

    p_atypical: float = 0.12
    """Fraction whose presenting complaint points at the wrong branch.

    The single most load-bearing assumption in the whole simulation, and the
    one with the least public evidence behind it. Sweep it; never quote a
    result without it."""

    p_deteriorate: float = 0.08
    """Fraction who get worse while waiting."""

    deterioration_delay: tuple[float, float] = (20.0, 180.0)
    """Minutes after arrival, uniform."""

    seed: int = 20260817


@dataclass
class Patient:
    id: int
    arrival: float
    true_category: Category
    obvious_branch: str
    findings: dict[str, Answer]

    atypical: bool = False
    deteriorates_at: float | None = None
    deterioration_findings: dict[str, Answer] = field(default_factory=dict)
    deteriorated_to: Category | None = None

    def truth_at(self, when: float) -> Category:
        """Ground truth as of a point in time."""
        if self.deteriorates_at is not None and when >= self.deteriorates_at:
            return self.deteriorated_to or self.true_category
        return self.true_category

    def findings_at(self, when: float) -> dict[str, Answer]:
        """What is observable as of a point in time."""
        if self.deteriorates_at is not None and when >= self.deteriorates_at:
            return {**self.findings, **self.deterioration_findings}
        return dict(self.findings)


# --------------------------------------------------------------------------
# mapping categories to checks that produce them
# --------------------------------------------------------------------------

def _checks_by_category() -> dict[Category, list[str]]:
    """Which checks can produce each category, across the whole protocol."""
    out: dict[Category, list[str]] = {}
    for branch in PROTOCOL.branches.values():
        for did, cat in branch.rules:
            out.setdefault(cat, [])
            if did not in out[cat]:
                out[cat].append(did)
    return out


_BY_CATEGORY = _checks_by_category()


def _branches_carrying(check_id: str, category: Category) -> list[str]:
    return [
        b.id
        for b in PROTOCOL.branches.values()
        if b.category_for(check_id) == category
    ]


def _branches_not_carrying(check_id: str) -> list[str]:
    return [
        b.id
        for b in PROTOCOL.branches.values()
        if b.category_for(check_id) is None
    ]


def generate(cohort: Cohort) -> list[Patient]:
    """A reproducible patient stream. Same cohort, same patients, always."""
    rng = random.Random(cohort.seed)
    categories = list(cohort.mix)
    weights = [cohort.mix[c] for c in categories]

    patients: list[Patient] = []
    clock = 0.0
    gap = 60.0 / cohort.arrivals_per_hour

    for i in range(cohort.n):
        clock += rng.expovariate(1.0 / gap)
        true_cat = rng.choices(categories, weights=weights, k=1)[0]

        findings: dict[str, Answer] = {}
        atypical = False
        obvious: str

        candidates = _BY_CATEGORY.get(true_cat, [])
        if not candidates:
            # No check produces this category (BLUE) -- nothing positive.
            obvious = rng.choice(list(PROTOCOL.branches))
        else:
            driver = rng.choice(candidates)
            findings[driver] = Answer.TRUE

            carrying = _branches_carrying(driver, true_cat)
            not_carrying = _branches_not_carrying(driver)

            if rng.random() < cohort.p_atypical and not_carrying:
                # The presentation points somewhere the danger check doesn't live.
                obvious = rng.choice(not_carrying)
                atypical = True
            else:
                obvious = rng.choice(carrying) if carrying else rng.choice(list(PROTOCOL.branches))

        # A little incidental noise so branch weighting has something to do.
        for extra in rng.sample(sorted(PROTOCOL.discriminators), k=rng.randint(0, 2)):
            findings.setdefault(extra, Answer.TRUE if rng.random() < 0.3 else Answer.FALSE)

        p = Patient(
            id=i,
            arrival=clock,
            true_category=true_cat,
            obvious_branch=obvious,
            findings=findings,
            atypical=atypical,
        )

        if true_cat > Category.ORANGE and rng.random() < cohort.p_deteriorate:
            lo, hi = cohort.deterioration_delay
            p.deteriorates_at = clock + rng.uniform(lo, hi)
            worse = Category(max(Category.RED, true_cat - 1))
            p.deteriorated_to = worse
            new_candidates = _BY_CATEGORY.get(worse, [])
            if new_candidates:
                p.deterioration_findings = {rng.choice(new_candidates): Answer.TRUE}

        patients.append(p)

    return patients
