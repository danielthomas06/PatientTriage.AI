"""The waiting queue, and what happens to people in it.

The Round 2 brief makes this mandatory, and names both triggers:

    "The system must monitor patients already in the waiting queue and trigger
     re-assessment if wait time exceeds safe thresholds for their severity level
     or if vitals are re-recorded as worsening."

Note "for their severity level". A single re-check interval for everyone is the
wrong shape: CTAS publishes one per category, and they differ by an order of
magnitude between the top and the bottom of the scale.

WHY THIS MODULE MATTERS MORE THAN THE INITIAL SCORE.

The serious-incident reports are rarely about the triage decision. They are
about a patient triaged CORRECTLY as low acuity who then deteriorated in a
corridor at hour three while nobody looked at them again. Initial triage is a
snapshot; this is the part that keeps watching.

THREE TRIGGERS, and the third is ours rather than the protocol's:

    TIMER      the published re-check interval for this category has elapsed
    WORSENED   re-scoring on new observations returns a more urgent category
    STARVED    waited far past their own target -- our policy, not CTAS's

STARVATION DOES NOT CHANGE THE CATEGORY, and that is deliberate. An earlier
version raised it a band, which looked right on one patient and fell apart on a
full board: every Orange passes its 3x multiple inside an hour, so on a busy
shift the whole queue turns Immediate, and a board where everyone is Immediate
says nothing at all.

Acuity is how sick someone is. Waiting time is how badly the department is
failing them. Mixing them makes both unreadable. So a starved patient is flagged
and sorted to the top of their own band -- impossible to miss, and the acuity
signal still means something at the end of a bad shift.

ESCALATE ONLY. The board may raise a priority without asking. It may never lower
one: a de-escalation is a clinical judgement and needs a named human, which is
what `audit.record_override` is for. So every downgrade in the record is
attributable by construction, and this module cannot produce one.

IMMUTABILITY. The initial category is kept alongside the current one and never
overwritten, per the CTAS instruction quoted in `protocols/ctas.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from .belief import BeliefState
from .core import Category, Protocol
from .engine import decide

# Published CTAS re-assessment intervals, in minutes. RED is continuous nursing
# care rather than an interval, which is why it is zero rather than a number.
REASSESS_MINUTES: dict[Category, int] = {
    Category.RED: 0,
    Category.ORANGE: 15,
    Category.YELLOW: 30,
    Category.GREEN: 60,
    Category.BLUE: 120,
}

# Target time to first clinician assessment, used for the starvation guard.
TARGET_MINUTES: dict[Category, int] = {
    Category.RED: 0,
    Category.ORANGE: 15,
    Category.YELLOW: 30,
    Category.GREEN: 60,
    Category.BLUE: 120,
}

# A patient who has waited this many times their own target is escalated whether
# or not anything about them has changed. Purely our policy -- no protocol
# publishes a starvation rule -- so it is a named constant a department can tune
# rather than a number buried in a condition.
STARVATION_MULTIPLE = 3.0

# Physical beds, not queue slots. Fixed for the demo rather than configurable --
# a real deployment reads this from a bed-management feed. The distinction this
# buys: the board never turns anyone away. A patient with no free bed still
# joins the queue with `bed=None`, which is the honest state of a real
# department at capacity -- prioritised, waiting, and not yet in a bay.
BED_COUNT = 10


class Trigger(StrEnum):
    TIMER = "re-check overdue"
    WORSENED = "worsened on re-scoring"
    STARVED = "waited far past target"


@dataclass(frozen=True, slots=True)
class Escalation:
    ref: str
    at: float
    trigger: Trigger
    was: Category
    now: Category
    detail: str

    def __str__(self) -> str:
        return (f"{self.ref}: {self.was.label} -> {self.now.label} "
                f"({self.trigger}) -- {self.detail}")


@dataclass(frozen=True, slots=True)
class Waiting:
    ref: str
    arrived: float
    initial: Category          # never changes -- the record of what was decided
    current: Category
    belief: BeliefState
    plausible: frozenset[str]
    last_checked: float
    escalations: tuple[Escalation, ...] = ()
    flagged_starved: bool = False
    """Waited far past target. A flag, not a category change -- see the module
    docstring. Fires once per patient rather than on every tick."""

    bed: int | None = None
    """1..BED_COUNT if one was free at admission, else None -- waiting for a
    bed rather than refused. Never reassigned by re-scoring or a tick; only
    `seen()` frees it, by removing the patient from the board entirely."""

    def waited(self, now: float) -> float:
        return now - self.arrived

    def since_check(self, now: float) -> float:
        return now - self.last_checked

    def due_in(self, now: float) -> float:
        """Minutes until the next re-check. Negative means overdue.

        RED is continuous observation rather than an interval, so it is always
        treated as due -- there is no window in which nobody is watching.
        """
        interval = REASSESS_MINUTES[self.current]
        if interval == 0:
            return -self.since_check(now)
        return interval - self.since_check(now)

    def overdue(self, now: float) -> bool:
        return self.due_in(now) < 0

    def starved(self, now: float) -> bool:
        target = TARGET_MINUTES[self.current]
        if target == 0:
            return False           # already the most urgent category
        return self.waited(now) > target * STARVATION_MULTIPLE


@dataclass
class Board:
    """The waiting room. Add on triage, re-score on new observations, tick on a clock."""

    protocol: Protocol
    patients: dict[str, Waiting] = field(default_factory=dict)
    log: list[Escalation] = field(default_factory=list)

    # ------------------------------------------------------------------ write

    def admit(self, ref: str, category: Category, belief: BeliefState,
              plausible: frozenset[str], now: float = 0.0) -> Waiting:
        used = {w.bed for w in self.patients.values() if w.bed is not None}
        bed = next((n for n in range(1, BED_COUNT + 1) if n not in used), None)
        w = Waiting(
            ref=ref, arrived=now, initial=category, current=category,
            belief=belief, plausible=plausible, last_checked=now, bed=bed,
        )
        self.patients[ref] = w
        return w

    def free_beds(self) -> int:
        used = {w.bed for w in self.patients.values() if w.bed is not None}
        return BED_COUNT - len(used)

    def observe(self, ref: str, belief: BeliefState, now: float) -> Escalation | None:
        """New observations on a waiting patient. Re-scores immediately.

        This is the "vitals re-recorded as worsening" trigger, and it does not
        wait for the timer -- a patient who deteriorates two minutes after triage
        should not sit until their interval elapses.
        """
        w = self.patients[ref]
        fresh = decide(belief, w.plausible)
        self.patients[ref] = replace(w, belief=belief, last_checked=now)

        if fresh.category < w.current:
            e = self._escalate(
                ref, now, Trigger.WORSENED, w.current, fresh.category,
                fresh.explain(self.protocol),
            )
            # Only `tick` was extending the log, so a deterioration caught by
            # re-recorded vitals -- the trigger the brief names first -- was
            # escalating the patient and leaving no record of why.
            self.log.append(e)
            return e
        # Deliberately no branch for fresh.category > w.current. Re-scoring can
        # raise a priority on its own; lowering one needs a named clinician.
        return None

    def tick(self, now: float) -> list[Escalation]:
        """Advance the clock. Returns everything that fired."""
        fired: list[Escalation] = []
        for ref in list(self.patients):
            w = self.patients[ref]

            if w.starved(now) and not w.flagged_starved:
                target = TARGET_MINUTES[w.current]
                self.patients[ref] = replace(w, flagged_starved=True)
                fired.append(Escalation(
                    ref=ref, at=now, trigger=Trigger.STARVED,
                    was=w.current, now=w.current,      # category untouched
                    detail=f"waited {w.waited(now):.0f} min against a "
                           f"{target} min target -- find this patient now",
                ))
                continue

            if w.overdue(now):
                # An overdue re-check is not itself an escalation -- it is a
                # request for one. Re-score against what is already known; if
                # nothing has changed, the category holds and the nurse is
                # prompted rather than the patient silently moved.
                fresh = decide(w.belief, w.plausible)
                if fresh.category < w.current:
                    fired.append(self._escalate(
                        ref, now, Trigger.TIMER, w.current, fresh.category,
                        fresh.explain(self.protocol),
                    ))
                else:
                    fired.append(Escalation(
                        ref=ref, at=now, trigger=Trigger.TIMER,
                        was=w.current, now=w.current,
                        detail=f"{w.since_check(now):.0f} min since last check "
                               f"({REASSESS_MINUTES[w.current]} min interval) "
                               f"-- nurse re-check required",
                    ))
        self.log.extend(fired)
        return fired

    def seen(self, ref: str) -> Waiting | None:
        """Patient reached a clinician. Leaves the queue."""
        return self.patients.pop(ref, None)

    # ------------------------------------------------------------------- read

    def ranked(self, now: float) -> list[Waiting]:
        """Board order: most urgent first, then longest waiting.

        Within a category, a starved patient comes first, then longest waiting.
        Neither ever lets a patient cross a more urgent one -- that is the whole
        point of sorting on category first.
        """
        return sorted(
            self.patients.values(),
            key=lambda w: (w.current, not w.flagged_starved, -w.waited(now)),
        )

    def overdue(self, now: float) -> list[Waiting]:
        return [w for w in self.ranked(now) if w.overdue(now)]

    def escalations(self) -> list[Escalation]:
        """Only the ones that actually moved a category."""
        return [e for e in self.log if e.now < e.was]

    def starved(self, now: float) -> list[Waiting]:
        """Flagged as waiting far past target. Not sicker -- failed."""
        return [w for w in self.ranked(now) if w.flagged_starved]

    def _escalate(self, ref: str, now: float, trigger: Trigger,
                  was: Category, to: Category, detail: str) -> Escalation:
        e = Escalation(ref=ref, at=now, trigger=trigger, was=was, now=to, detail=detail)
        w = self.patients[ref]
        self.patients[ref] = replace(
            w, current=to, last_checked=now, escalations=w.escalations + (e,),
        )
        return e
