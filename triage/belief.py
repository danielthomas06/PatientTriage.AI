"""The shared belief state.

One vector over the whole discriminator vocabulary. Every branch reads from
this same object -- that is what lets fifty branches be evaluated at once
instead of one being committed to.

Models write here. Nothing downstream of here is a model.
"""

from dataclasses import dataclass, field, replace

from .core import Answer, Protocol


@dataclass(frozen=True, slots=True)
class Evidence:
    """Why a discriminator holds the value it does."""

    quote: str          # the patient's actual words, or a measured value
    origin: str         # "speech" | "vitals" | "record" | "staff"

    def __str__(self) -> str:
        return f"{self.origin}: {self.quote!r}"


@dataclass(frozen=True, slots=True)
class BeliefState:
    protocol: Protocol
    answers: dict[str, Answer] = field(default_factory=dict)
    evidence: dict[str, Evidence] = field(default_factory=dict)

    def p_true(self, discriminator_id: str) -> float:
        """Probability this check is positive.

        An observed answer is 1.0 or 0.0. An unobserved one falls back to the
        base rate -- never to zero, because "not mentioned" is not "absent".
        """
        match self.answers.get(discriminator_id, Answer.UNKNOWN):
            case Answer.TRUE:
                return 1.0
            case Answer.FALSE:
                return 0.0
            case _:
                return self.protocol.discriminators[discriminator_id].prior

    def is_observed(self, discriminator_id: str) -> bool:
        return self.answers.get(discriminator_id, Answer.UNKNOWN) is not Answer.UNKNOWN

    def unobserved(self) -> list[str]:
        return [d for d in self.protocol.discriminators if not self.is_observed(d)]

    def positives(self) -> list[str]:
        return [d for d, a in self.answers.items() if a is Answer.TRUE]

    def record(
        self,
        discriminator_id: str,
        answer: Answer,
        evidence: Evidence | None = None,
    ) -> "BeliefState":
        """Return a new state with this check answered. Never mutates."""
        if discriminator_id not in self.protocol.discriminators:
            raise KeyError(f"unknown discriminator {discriminator_id!r}")
        answers = self.answers | {discriminator_id: answer}
        ev = self.evidence
        if evidence is not None:
            ev = ev | {discriminator_id: evidence}
        return replace(self, answers=answers, evidence=ev)

    def hypothetical(self, discriminator_id: str, answer: Answer) -> "BeliefState":
        """A throwaway copy used to price a question before asking it."""
        return replace(self, answers=self.answers | {discriminator_id: answer})

    def trace(self) -> list[str]:
        """Human-readable audit line per positive finding."""
        out = []
        for did in self.positives():
            d = self.protocol.discriminators[did]
            ev = self.evidence.get(did)
            out.append(f"{d.text}" + (f"  <- {ev}" if ev else "  <- (no evidence recorded)"))
        return out
