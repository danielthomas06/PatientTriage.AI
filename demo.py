"""End-to-end trace of the worked example.

    python demo.py

68-year-old woman, walk-in, epigastric pain. The presentation that gets
triaged as indigestion and turns out to be an inferior myocardial infarction.

Run it to see the selector choose six observations out of thirty-seven, catch
the escalation on a branch nobody selected, and stop the moment it fires.
"""

import time

from triage import (
    Answer, Category, Evidence, Vitals, acuity_distribution,
    decide, plausible_set, rank, score, should_stop, walk_in_baseline,
)

# This walkthrough is scripted against specific discriminator ids, so it names
# the pack that has them rather than taking whatever TRIAGE_PROTOCOL points at.
# The CTAS pack is exercised by run_sim.py and tests/test_ctas.py, which assert
# on behaviour rather than on particular ids.
from triage.charts import PROTOCOL

RULE = "-" * 78


def banner(text: str) -> None:
    print(f"\n{RULE}\n  {text}\n{RULE}")


# What the patient actually said. In production an LLM produces the branch
# weights and the opening checks from this; here they are hand-seeded so the
# trace is reproducible and the engine can be judged on its own.
NARRATIVE = "pain in my upper stomach, feel sick, a bit sweaty"

BRANCH_WEIGHTS = {
    "abdominal_pain": 0.45,
    "chest_pain": 0.25,
    "unwell_adult": 0.15,
    "vomiting": 0.10,
    "breathlessness": 0.03,
    "collapse": 0.01,
    "palpitations": 0.01,
}

# Scripted replies, so the run is deterministic.
SCRIPT: dict[str, tuple[Answer, str]] = {
    "very_low_spo2": (Answer.FALSE, "saturation 96% on air"),
    "low_spo2": (Answer.FALSE, "saturation 96% on air"),
    "shock": (Answer.FALSE, "BP 138/82, capillary refill under 2s"),
    "abnormal_pulse": (Answer.FALSE, "pulse 88, regular"),
    "very_hot": (Answer.FALSE, "temperature 36.9C"),
    "hot": (Answer.FALSE, "temperature 36.9C"),
    "severe_pain": (Answer.FALSE, "she rates it 6 out of 10"),
    "cardiac_pain": (Answer.TRUE, "...a bit, yeah. Into my jaw."),
    "vomiting_blood": (Answer.FALSE, "no blood"),
    "cardiac_history": (Answer.FALSE, "no cardiac history on record"),
    "anticoagulated": (Answer.FALSE, "no anticoagulants on record"),
}

VITALS = Vitals(
    respiratory_rate=18, spo2=96, on_oxygen=False, systolic_bp=138,
    pulse=88, alert=True, temperature=36.9, age_years=68,
)


def show_state(belief, plausible, vitals_score) -> None:
    d = decide(belief, plausible, vitals=vitals_score)
    dist = acuity_distribution(belief, plausible)
    spread = "  ".join(
        f"{c.name}:{p:.2f}" for c, p in sorted(dist.items()) if p > 0.005
    )
    print(f"    now: {d.explain(PROTOCOL)}")
    print(f"    belief: {spread}")


def main() -> None:
    banner("ARRIVAL")
    print(f"  68F, walk-in, unaccompanied.")
    print(f'  Says: "{NARRATIVE}"')

    belief = walk_in_baseline(PROTOCOL)
    print(f"\n  Across-the-room look clears {len(belief.answers)} checks without asking anything.")

    # The narrative itself answers a check. Note what is NOT set: she said
    # "feel sick", which is nausea, not vomiting -- the extractor must not
    # promote that to persistent_vomiting.
    belief = belief.record(
        "moderate_pain", Answer.TRUE, Evidence("pain in my upper stomach", "speech")
    )

    plausible = plausible_set(BRANCH_WEIGHTS)
    print(f"  Plausible branches: {len(plausible)} of {len(PROTOCOL.branches)}")
    for b, w in sorted(BRANCH_WEIGHTS.items(), key=lambda kv: -kv[1]):
        mark = "*" if b in plausible else " "
        print(f"    {mark} {PROTOCOL.branches[b].name:32s} {w:.2f}")
    print("\n  The nurse would commit to Abdominal pain. We keep all four open.")

    vitals_score = None
    banner("ACQUISITION")
    show_state(belief, plausible, vitals_score)

    asked = 0
    started = time.perf_counter()
    while True:
        reason = should_stop(belief, plausible, asked)
        if reason:
            break

        best = rank(belief, plausible, limit=1)[0]
        d = PROTOCOL.discriminators[best.discriminator_id]
        answer, quote = SCRIPT.get(
            best.discriminator_id, (Answer.FALSE, "denied on direct questioning")
        )

        asked += 1
        print(f"\n  [{asked}] {d.question}")
        print(f"      value {best.value:.3f} / cost {best.cost:.2f}  ({d.source.value})")
        print(f"      -> {answer.value.upper()}  \"{quote}\"")

        belief = belief.record(
            best.discriminator_id, answer,
            Evidence(quote, "vitals" if d.source.value == "measure" else "speech"),
        )
        if d.source.value == "measure" and vitals_score is None and asked >= 3:
            vitals_score = score(VITALS)
            print(f"      vital-sign score {vitals_score.total} ({vitals_score.band})"
                  f" -- {vitals_score.explain()}")
        show_state(belief, plausible, vitals_score)

    elapsed = (time.perf_counter() - started) * 1000

    banner("OUTCOME")
    final = decide(belief, plausible, vitals=vitals_score)
    print(f"  {final.explain(PROTOCOL)}")
    print(f"  Target time to clinician: {final.category.target_minutes} min")
    print(f"  Stopped because: {should_stop(belief, plausible, asked)}")
    print(f"\n  {asked} observations taken, out of {len(PROTOCOL.discriminators)} possible.")
    print(f"  Selection time for the whole run: {elapsed:.1f} ms, no network, no model.")

    print("\n  Evidence:")
    for line in belief.trace():
        print(f"    - {line}")

    banner("COUNTERFACTUAL -- what one committed branch would have done")
    committed = frozenset({"abdominal_pain"})
    alt = decide(belief, committed, vitals=vitals_score)
    print(f"  Committed to Abdominal pain: {alt.explain(PROTOCOL)}")
    print(f"  Evaluating all four:         {final.explain(PROTOCOL)}")

    reachable = {
        did for b in committed for did in PROTOCOL.branches[b].discriminator_ids
    }
    print(f"\n  'Pain radiating to jaw, neck or arm' is on the Abdominal pain branch: "
          f"{'cardiac_pain' in reachable}")
    print("  So the question that escalated her would never have been asked,")
    print("  and she would have waited "
          f"{alt.category.target_minutes - final.category.target_minutes} minutes longer "
          f"as {alt.category.label} instead of {final.category.label}.")
    print()


if __name__ == "__main__":
    main()
