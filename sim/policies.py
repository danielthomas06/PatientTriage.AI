"""The two arms.

Both call the real engine — `decide`, `plausible_set`, the tripwire. The
simulator is therefore a test of the shipped code, not a parallel mock of it.
If the engine regresses, these numbers move.

The comparison is only worth anything if it is fair, so the assistant arm is
given its own failure mode: `p_branch_missed`, the chance its plausible set
wrongly excludes the branch carrying the dangerous check. Set that to 1.0 and
the assistant collapses to the baseline, which is the sanity check.
"""

from dataclasses import dataclass
from random import Random

from triage import PROTOCOL, Answer, BeliefState, Category, Evidence, decide
from triage.monitor import REASSESS_MINUTES


@dataclass(frozen=True, slots=True)
class Outcome:
    """What a policy concluded, and what it cost."""

    category: Category
    triage_minutes: float


class Baseline:
    """Today: commit to one branch in seconds, then never look again."""

    name = "baseline"

    def __init__(self, triage_minutes: float = 8.0):
        self.triage_minutes = triage_minutes

    def triage(self, patient, now: float, rng: Random) -> Outcome:
        belief = _belief(patient.findings_at(now))
        d = decide(belief, frozenset({patient.obvious_branch}))
        return Outcome(d.category, self.triage_minutes * rng.uniform(0.7, 1.4))

    # No recheck_interval and no recheck_every. The baseline never looks
    # again, which is the whole point of the arm.

    def recheck(self, patient, assigned: Category, now: float) -> Category | None:
        return None  # the whole point: nobody looks again


class Assistant:
    """All branches evaluated at once, plus a tripwire, plus re-scoring."""

    name = "assistant"

    def __init__(
        self,
        triage_minutes: float = 3.0,
        recheck_every: float = 15.0,
        p_branch_missed: float = 0.05,
        parallel: bool = True,
        per_category: bool = True,
    ):
        self.triage_minutes = triage_minutes
        self.recheck_every = recheck_every
        """Legacy flat interval, kept only so an ablation can turn per-category
        intervals off and measure what they are worth."""
        self.p_branch_missed = p_branch_missed
        self.parallel = parallel
        self.per_category = per_category
        """Ablation switch, distinct from p_branch_missed.

        `parallel=False` commits to the single obvious branch — exactly what
        the baseline does — so turning it off alongside re-checking should
        land the assistant *on* the baseline, not below it. `p_branch_missed`
        is the different question of how often a model-produced plausible set
        is wrong; at 1.0 it strips the danger branches including the obvious
        one, which leaves the assistant worse off than a nurse who at least
        picked something. That is a sabotage, not an ablation, and conflating
        the two made the first version of this table nonsense."""

    def _plausible(self, patient, rng: Random) -> frozenset[str]:
        """Every branch, minus the ones we sometimes wrongly rule out.

        The design claim is that all branches stay open. The honest caveat is
        that the plausible set comes from a model, and the model can be wrong.

        When it is wrong it drops *every* branch carrying the danger check, not
        one of them. An earlier version removed a single branch and the sanity
        check caught it: dangerous checks sit on several branches at once, so
        dropping one changed almost nothing and the assistant still won at
        p_branch_missed = 1.0. That made the harness look like it was measuring
        parallel evaluation when it was really measuring check redundancy.
        """
        if not self.parallel:
            return frozenset({patient.obvious_branch})
        everything = set(PROTOCOL.branches)
        if rng.random() < self.p_branch_missed:
            everything -= _danger_branches(patient)
        return frozenset(everything) or frozenset({patient.obvious_branch})

    def triage(self, patient, now: float, rng: Random) -> Outcome:
        findings = patient.findings_at(now)
        belief = _belief(findings)
        d = decide(belief, self._plausible(patient, rng))
        return Outcome(d.category, self.triage_minutes * rng.uniform(0.7, 1.4))

    # A flat interval for everyone is the wrong shape, and the brief says so:
    # re-assess "if wait time exceeds safe thresholds FOR THEIR SEVERITY LEVEL".
    # CTAS publishes one per category and they differ eightfold end to end, so a
    # single number either checks the sickest too rarely or the rest pointlessly
    # often. RED is continuous observation rather than an interval; the simulator
    # cannot model continuous, so it uses the shortest interval it can.
    CONTINUOUS_PROXY = 5.0

    def recheck_interval(self, assigned: Category) -> float:
        if not self.per_category:
            return self.recheck_every
        return REASSESS_MINUTES[assigned] or self.CONTINUOUS_PROXY

    def recheck(self, patient, assigned: Category, now: float) -> Category | None:
        """Re-score on the timer. Escalate only — never lower a priority."""
        belief = _belief(patient.findings_at(now))
        scope = (
            frozenset(PROTOCOL.branches)
            if self.parallel
            else frozenset({patient.obvious_branch})
        )
        d = decide(belief, scope)
        return d.category if d.category < assigned else None


def _belief(findings: dict[str, Answer]) -> BeliefState:
    b = BeliefState(PROTOCOL)
    for check_id, answer in findings.items():
        if check_id in PROTOCOL.discriminators:
            b = b.record(check_id, answer, Evidence("simulated", "sim"))
    return b


def _danger_branches(patient) -> set[str]:
    """Branches that carry one of this patient's positive checks at its worst."""
    positives = [k for k, v in patient.findings.items() if v is Answer.TRUE]
    out: set[str] = set()
    for branch in PROTOCOL.branches.values():
        for check_id in positives:
            cat = branch.category_for(check_id)
            if cat is not None and cat <= Category.ORANGE:
                out.add(branch.id)
    return out
