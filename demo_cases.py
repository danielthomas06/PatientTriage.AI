"""The four cases the Round 2 brief names, plus a clinician override.

    python demo_cases.py

The brief asks for at least one ambiguous presentation, one paediatric or
geriatric case, and one zero-history patient; uncertainty surfaced on every
score; and a captured override with what the system logs. One case each, run
against the CTAS pack.
"""

from triage import Answer, BeliefState, Category, Evidence, decide, plausible_set
from triage.audit import DATA_PROTECTION, EventKind, Ledger, ReasonCode
from triage.cohort import Patient, paediatric_vital_category, resolve
from triage.protocols.ctas import PROTOCOL, TARGET_MINUTES

RULE = "=" * 78


def banner(text: str) -> None:
    print(f"\n{RULE}\n  {text}\n{RULE}")


def show(belief, weights, *, note=""):
    plausible = plausible_set(weights)
    d = decide(belief, plausible)
    # Target times come from the PACK, not from Category. Category carries MTS
    # times (Very urgent = 10 min); CTAS says 15. Printing one protocol's targets
    # beside another's categories is exactly the quiet mismatch this design exists
    # to prevent, so the pack is the authority.
    print(f"\n  CATEGORY    {d.category.label}  (CTAS target {TARGET_MINUTES[d.category]} min)")
    if d.fired:
        names = ", ".join(PROTOCOL.branches[b].name for b in d.fired_on)
        print(f"  BECAUSE     {PROTOCOL.discriminators[d.fired].text}")
        print(f"              on: {names}")
    print(f"  CONFIDENCE  {d.confidence}")
    if note:
        print(f"  NOTE        {note}")
    return d


def cohort_block(patient):
    a = resolve(patient)
    print(f"  COHORT      {a.cohort}")
    print(f"  ADULT SCORE {'applicable' if a.may_score_vitals else 'WITHHELD'}")
    for w in a.warnings:
        print(f"  !           {w}")
    return a


# ---------------------------------------------------------------------------
# 1. AMBIGUOUS -- epigastric pain that is cardiac
# ---------------------------------------------------------------------------

def case_ambiguous() -> tuple[BeliefState, dict]:
    banner("CASE 1 -- AMBIGUOUS PRESENTATION")
    print("  68F, walk-in. 'Pain in my upper stomach, feel sick, a bit sweaty.'")

    weights = {"abdominal_pain": 0.80, "chest_pain": 0.20, "general_unwell": 0.10}
    print("\n  Model ranks abdominal_pain first. A nurse would commit to it.")

    belief = BeliefState(PROTOCOL).record(
        "moderate_pain_central", Answer.TRUE,
        Evidence("pain in my upper stomach", "speech"),
    )
    show(belief, weights)

    belief = belief.record(
        "pain_radiating", Answer.TRUE,
        Evidence("it goes up into my jaw a bit", "speech"),
    )
    d = show(belief, weights,
             note="fired from chest_pain, ranked SECOND -- never asked if you commit")

    committed = decide(belief, frozenset({"abdominal_pain"}))
    print(f"\n  Counterfactual, abdominal_pain alone: {committed.category.label}")
    gap = TARGET_MINUTES[committed.category] - TARGET_MINUTES[d.category]
    print(f"  Difference: {gap} minutes of extra wait for an evolving "
          f"myocardial infarction.")
    return belief, weights


# ---------------------------------------------------------------------------
# 2. PAEDIATRIC -- age-banded vitals, and a normal score that is not reassurance
# ---------------------------------------------------------------------------

def case_paediatric() -> None:
    banner("CASE 2 -- PAEDIATRIC")
    print("  14-month-old, febrile, irritable and not consolable.")
    print("  Heart rate 125, respiratory rate 30.")

    child = Patient(age_months=14, looks_unwell=True, suspected_infection=True,
                    temperature=39.1)
    cohort_block(child)

    category, why = paediatric_vital_category(child, resp_rate=30, pulse=125)
    print(f"\n  VITALS      {'within normal for age' if category is None else category.label}")
    for line in why:
        print(f"              {line}")

    print("\n  The same numbers on an adult chart:")
    print("    heart rate 125 -> tachycardic;  respiratory rate 30 -> tachypnoeic")
    print("  Both are normal at 14 months. Scoring a child on the adult table")
    print("  over-triages loudly -- and the reverse error is the silent one.")

    belief = BeliefState(PROTOCOL).record(
        "fever_looks_unwell", Answer.TRUE, Evidence("39.1C, irritable, inconsolable", "staff"),
    )
    show(belief, {"fever": 0.70, "general_unwell": 0.20})


# ---------------------------------------------------------------------------
# 3. GERIATRIC -- the falsely reassuring pulse
# ---------------------------------------------------------------------------

def case_geriatric() -> None:
    banner("CASE 3 -- GERIATRIC, ATYPICAL PHYSIOLOGY")
    print("  81M, 'off legs' two days, productive cough. On bisoprolol.")
    print("  Heart rate 78, temperature 37.4, respiratory rate 24.")

    elder = Patient(age_years=81, frail=True, rate_limiting_meds=True,
                    suspected_infection=True, temperature=37.4)
    cohort_block(elder)

    print("\n  Read naively: pulse normal, afebrile. Nothing alarms.")
    print("  Both readings are unreliable IN THIS PATIENT, and only the record")
    print("  says so -- which is why record lookups are free and always taken first.")

    belief = (
        BeliefState(PROTOCOL)
        .record("rate_limiting_meds", Answer.TRUE, Evidence("bisoprolol 5mg OD", "record"))
        .record("resp_distress_mild", Answer.TRUE, Evidence("respiratory rate 24", "vitals"))
    )
    show(belief, {"general_unwell": 0.55, "fever": 0.20, "shortness_of_breath": 0.15})


# ---------------------------------------------------------------------------
# 4. ZERO HISTORY -- no ID, no record, no shared language
# ---------------------------------------------------------------------------

def case_zero_history() -> None:
    banner("CASE 4 -- ZERO HISTORY")
    print("  Unidentified adult. No record match, minimal shared language,")
    print("  age not established. Gestures to abdomen, visibly distressed.")

    unknown = Patient()
    cohort_block(unknown)

    print("\n  Every record-sourced check stays UNKNOWN. None becomes 'no'.")
    belief = BeliefState(PROTOCOL).record(
        "severe_pain_central", Answer.TRUE,
        Evidence("gestured to abdomen, visibly distressed", "staff"),
    )
    show(belief, {"abdominal_pain": 0.40, "general_unwell": 0.30, "chest_pain": 0.15},
         note="low confidence is the correct output here, not a failure")


# ---------------------------------------------------------------------------
# 5. OVERRIDE -- and what the system logs
# ---------------------------------------------------------------------------

def case_override(belief, weights) -> None:
    banner("CASE 5 -- CLINICIAN OVERRIDE, AND THE AUDIT TRAIL")

    ledger = Ledger(patient_ref="ED-2026-0824-0117")
    for did in belief.positives():
        ledger.append(EventKind.OBSERVED, actor="system",
                      check=did, value="true",
                      evidence=str(belief.evidence.get(did, "")))

    d = decide(belief, plausible_set(weights))
    ledger.record_decision(d, evidence=belief.trace())
    ledger.append(EventKind.SHOWN, actor="system", to="rn.k.mensah")

    print(f"  System recommended: {d.category.label}")
    print("  Nurse escalates to Immediate -- she is grey and clammy, which no")
    print("  check in the vocabulary captures.\n")

    ledger.record_override(
        clinician="rn.k.mensah",
        recommended=d.category,
        chosen=Category.RED,
        reason=ReasonCode.CLINICAL_JUDGEMENT,
        note="grey, clammy, looks unwell beyond what the checks capture",
        decision=d,
        evidence=belief.trace(),
    )

    print("  LEDGER")
    for e in ledger.events:
        print(f"    {e}")

    ok, detail = ledger.verify()
    print(f"\n  CHAIN       {'intact' if ok else 'BROKEN'} -- {detail}")

    print("\n  Tamper check: rewrite one event and re-verify.")
    bad = Ledger(patient_ref=ledger.patient_ref, events=list(ledger.events))
    victim = bad.events[1]
    bad.events[1] = type(victim)(
        seq=victim.seq, at=victim.at, kind=victim.kind, actor=victim.actor,
        payload={**victim.payload, "value": "false"},
        prev_hash=victim.prev_hash, digest=victim.digest,
    )
    ok2, detail2 = bad.verify()
    print(f"  CHAIN       {'intact' if ok2 else 'BROKEN'} -- {detail2}")

    print("\n  The initial score is never overwritten:")
    print(f"    {ledger.initial_decision}")

    print(f"\n  Override rate this encounter: {ledger.override_rate():.0%}")
    print("  Fleet-wide, near-zero is NOT success -- it means staff have stopped")
    print("  reading and are clicking accept. That is the automation-bias failure,")
    print("  and the rate broken down by reason code is how you catch it.")


def main() -> None:
    belief, weights = case_ambiguous()
    case_paediatric()
    case_geriatric()
    case_zero_history()
    case_override(belief, weights)

    banner("DATA PROTECTION")
    for k, v in DATA_PROTECTION.items():
        print(f"  {k:<18} {v}")
    print()


if __name__ == "__main__":
    main()
