"""The audit trail and clinician overrides.

The brief requires that a recommendation stays overridable, that the override is
captured, and that the system can show what it logged. These tests check the
record is complete and legible enough to survive an incident review, not just
that a function ran.
"""

import pytest

from triage import Answer, BeliefState, Category, Evidence, decide, plausible_set
from triage.audit import (
    DATA_PROTECTION, Direction, EventKind, Ledger, ReasonCode, override_metrics,
)
from triage.charts import PROTOCOL


def a_decision():
    belief = BeliefState(PROTOCOL).record(
        "moderate_pain", Answer.TRUE, Evidence("pain in my upper stomach", "speech")
    )
    return belief, decide(belief, plausible_set({"abdominal_pain": 0.8, "chest_pain": 0.2}))


# ------------------------------------------------------------ the chain

def test_chain_verifies_when_untouched():
    led = Ledger("ED-1")
    led.append(EventKind.OBSERVED, "system", check="shock", value="false")
    led.append(EventKind.SHOWN, "system", to="rn.smith")
    ok, detail = led.verify()
    assert ok, detail


def test_editing_an_event_breaks_the_chain():
    led = Ledger("ED-1")
    led.append(EventKind.OBSERVED, "system", check="shock", value="false")
    led.append(EventKind.SHOWN, "system", to="rn.smith")
    v = led.events[0]
    led.events[0] = type(v)(
        seq=v.seq, at=v.at, kind=v.kind, actor=v.actor,
        payload={**v.payload, "value": "true"},
        prev_hash=v.prev_hash, digest=v.digest,
    )
    ok, detail = led.verify()
    assert not ok and "altered" in detail


def test_deleting_an_event_breaks_the_chain():
    led = Ledger("ED-1")
    for i in range(3):
        led.append(EventKind.OBSERVED, "system", n=i)
    del led.events[1]
    ok, _ = led.verify()
    assert not ok


# ------------------------------------------------------------ overrides

def test_override_records_everything_a_review_needs():
    belief, d = a_decision()
    led = Ledger("ED-1")
    led.record_decision(d, evidence=belief.trace())
    led.append(EventKind.SHOWN, "system", to="rn.k.mensah")
    e = led.record_override(
        clinician="rn.k.mensah",
        recommended=d.category, chosen=Category.RED,
        reason=ReasonCode.CLINICAL_JUDGEMENT,
        note="grey and clammy", decision=d, evidence=belief.trace(),
    )
    p = e.payload
    assert e.actor == "rn.k.mensah"                 # who
    assert e.at                                      # when
    assert p["recommended"] and p["chosen"]          # from -> to
    assert p["reason"] == "clinical_judgement"       # why, structured
    assert p["note"]                                 # why, free text
    assert p["system_said"]["fired"]                 # what the tool was saying
    assert p["evidence_at_the_time"]                 # and on what basis


def test_an_unattributable_override_is_refused():
    """'A nurse changed it' is not an audit record."""
    _, d = a_decision()
    led = Ledger("ED-1")
    for anon in ("nurse", "staff", "unknown", ""):
        with pytest.raises(ValueError, match="identifiable clinician"):
            led.record_override(clinician=anon, recommended=d.category,
                                chosen=Category.RED, reason=ReasonCode.OTHER)


def test_same_category_is_an_acceptance_not_an_override():
    _, d = a_decision()
    with pytest.raises(ValueError, match="acceptance"):
        Ledger("ED-1").record_override(
            clinician="rn.smith", recommended=d.category,
            chosen=d.category, reason=ReasonCode.OTHER)


def test_direction_is_recorded_and_de_escalations_are_findable():
    """The system may raise a priority on its own but never lower one, so every
    de-escalation in the ledger is by construction a human decision."""
    led = Ledger("ED-1")
    led.record_override(clinician="rn.a", recommended=Category.YELLOW,
                        chosen=Category.RED, reason=ReasonCode.CLINICAL_JUDGEMENT)
    led.record_override(clinician="dr.b", recommended=Category.ORANGE,
                        chosen=Category.GREEN, reason=ReasonCode.INFORMATION_WRONG)
    assert led.overrides[0].payload["direction"] == Direction.ESCALATION
    assert led.overrides[1].payload["direction"] == Direction.DE_ESCALATION
    assert len(led.de_escalations) == 1
    assert led.de_escalations[0].actor == "dr.b"


def test_bands_moved_is_recorded():
    led = Ledger("ED-1")
    e = led.record_override(clinician="rn.a", recommended=Category.BLUE,
                            chosen=Category.RED, reason=ReasonCode.CLINICAL_JUDGEMENT)
    assert e.payload["bands_moved"] == 4


# ------------------------------------------------------------ immutability

def test_the_initial_score_is_never_overwritten():
    """CTAS: 'the initial triage score is never changed'. Re-scoring appends."""
    belief, d = a_decision()
    led = Ledger("ED-1")
    led.record_decision(d)
    first = led.initial_decision
    led.append(EventKind.RESCORED, "system", category="RED")
    led.record_override(clinician="rn.a", recommended=d.category,
                        chosen=Category.RED, reason=ReasonCode.CLINICAL_JUDGEMENT)
    assert led.initial_decision == first
    assert led.initial_decision.payload["category"] == d.category.name


def test_logged_confidence_is_plain_values_not_objects():
    """An audit record has to be legible years later, without this codebase."""
    _, d = a_decision()
    led = Ledger("ED-1")
    e = led.record_decision(d)
    c = e.payload["confidence"]
    assert isinstance(c["band"], str)
    assert isinstance(c["worst_case"], str)
    assert isinstance(c["p_assigned"], float)


# ------------------------------------------------------------ adoption metric

def test_override_metrics_break_down_by_reason_and_direction():
    led = Ledger("ED-1")
    for _ in range(4):
        led.append(EventKind.SHOWN, "system", to="rn.a")
    led.record_override(clinician="rn.a", recommended=Category.YELLOW,
                        chosen=Category.ORANGE, reason=ReasonCode.CLINICAL_JUDGEMENT)
    led.record_override(clinician="rn.a", recommended=Category.YELLOW,
                        chosen=Category.GREEN, reason=ReasonCode.INFORMATION_WRONG)
    m = override_metrics([led])
    assert m["shown"] == 4 and m["overrides"] == 2
    assert m["override_rate"] == 0.5
    assert m["escalations"] == 1 and m["de_escalations"] == 1
    assert m["by_reason"]["clinical_judgement"] == 1


def test_jurisdiction_is_stated_not_implied():
    assert "UK GDPR" in DATA_PROTECTION["jurisdiction"]
    assert "NOT the lawful basis" in DATA_PROTECTION["consent"]
