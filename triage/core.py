"""Core types for the triage engine.

Nothing here is probabilistic. A Category is produced by walking protocol data
against known discriminator values: same inputs, same output, every time. The
models live upstream and write into a BeliefState -- they never reach this far.
"""

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class Category(IntEnum):
    """Triage priority.

    Lower value means more urgent, so ``min`` picks the winner and the
    "highest plausible branch wins" rule is a one-liner.
    """

    RED = 1
    ORANGE = 2
    YELLOW = 3
    GREEN = 4
    BLUE = 5

    @property
    def label(self) -> str:
        return _LABEL[self.value]

    @property
    def target_minutes(self) -> int:
        """Target time to first clinician assessment."""
        return _TARGET[self.value]


_LABEL = {1: "Immediate", 2: "Very urgent", 3: "Urgent", 4: "Standard", 5: "Non-urgent"}
_TARGET = {1: 0, 2: 10, 3: 60, 4: 120, 5: 240}

LEAST_URGENT = Category.BLUE


class Answer(StrEnum):
    """A discriminator's state.

    UNKNOWN is not a soft FALSE. A model that did not hear something must
    leave it UNKNOWN -- a fabricated negative is the most dangerous output
    this system could produce.
    """

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class Source(StrEnum):
    """How a discriminator gets answered. Drives cost in the selector."""

    RECORD = "record"        # free lookup against prior notes
    MEASURE = "measure"      # a vital sign; dispatchable in parallel
    ASK = "ask"              # an ordinary question to the patient
    SENSITIVE = "sensitive"  # carries real cost in a public waiting room
    OBSERVE = "observe"      # needs a clinician's eyes on the patient


COST: dict[Source, float] = {
    Source.RECORD: 0.01,
    Source.MEASURE: 0.30,
    Source.ASK: 1.00,
    Source.SENSITIVE: 4.00,
    Source.OBSERVE: 6.00,
}


@dataclass(frozen=True, slots=True)
class Confidence:
    """How much the category could still move.

    Not an opinion from a model -- a fact about how much is still unknown. The
    engine assigns a category from CONFIRMED positives only, while the acuity
    distribution accounts for what remains unanswered. The gap between the two
    is the uncertainty.

    The Round 2 brief requires that no score is returned without one, so this is
    a required field on Decision rather than an optional extra: it is enforced by
    the type, not by remembering.
    """

    band: str                 # HIGH | MODERATE | LOW
    assigned: "Category"
    worst_case: "Category"    # most urgent category still carrying real probability
    p_assigned: float         # probability mass on the category we assigned
    unresolved: int           # decision-relevant checks still unanswered

    @property
    def could_escalate(self) -> bool:
        return self.worst_case < self.assigned

    def __str__(self) -> str:
        if not self.could_escalate:
            return f"{self.band} -- nothing unresolved could raise this"
        return (
            f"{self.band} -- could still be {self.worst_case.label} "
            f"({self.unresolved} checks unresolved)"
        )


@dataclass(frozen=True, slots=True)
class Discriminator:
    """One yes/no risk check, shared across however many branches use it."""

    id: str
    text: str          # what it means clinically
    question: str      # fallback phrasing; in production an LLM renders this
    source: Source
    prior: float = 0.05  # base rate before any evidence
    leading: tuple[str, ...] = ()
    """Words that would plant the answer if they appeared in the question.

    Asking "does the pain spread to your jaw?" manufactures the finding -- a
    frightened or deferential patient agrees with whatever is suggested. Ask
    "where exactly do you feel it?" and let them say jaw themselves. Enforced
    in `extract.render`, which rejects and re-renders."""


@dataclass(frozen=True, slots=True)
class Branch:
    """One presentation flowchart.

    ``rules`` is ordered most-urgent-first, which is how the published
    protocols are laid out and how a nurse works down them.
    """

    id: str
    name: str
    rules: tuple[tuple[str, Category], ...]

    def category_for(self, discriminator_id: str) -> Category | None:
        for did, cat in self.rules:
            if did == discriminator_id:
                return cat
        return None

    @property
    def discriminator_ids(self) -> tuple[str, ...]:
        return tuple(did for did, _ in self.rules)


@dataclass(frozen=True, slots=True)
class Protocol:
    """A complete triage protocol: a shared vocabulary plus branch orderings."""

    name: str
    discriminators: dict[str, Discriminator]
    branches: dict[str, Branch]

    def validate(self) -> None:
        """Every rule must reference a discriminator that exists."""
        for branch in self.branches.values():
            for did, _ in branch.rules:
                if did not in self.discriminators:
                    raise ValueError(f"{branch.id} references unknown discriminator {did!r}")

    @property
    def vocabulary_size(self) -> int:
        return len(self.discriminators)
