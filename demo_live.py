"""End-to-end with whichever model is actually reachable.

    python demo_live.py

No flags. The ladder decides:

    hosted   ANTHROPIC_API_KEY set          richest extraction
    local    ollama serve + a pulled model  no key, no network, no data egress
    keyword  neither                        no model at all

Same guards on every rung, so a weaker model yields more rejections and lower
confidence -- never confidently-wrong triage. Pull the network cable mid-run and
the tier changes; the safety floor does not.
"""

import time

from triage import Source, decide, parse, plausible_set, render, seed
from triage.charts import PROTOCOL

RULE = "=" * 78
NARRATIVE = (
    "I've had this pain in my upper stomach since about six this morning. "
    "I feel sick and I'm a bit sweaty. It's not going away."
)
# Keyed by CHECK, not by question wording. The model rephrases freely -- keyword
# matching on the question handed a "where is the pain" answer to a "how bad is
# it" question, which then correctly parsed as unclear and stalled the run.
REPLIES = {
    "cardiac_pain": "Up here under my ribs. And it goes up into my jaw a bit.",
    "moderate_pain": "About a six.",
    "severe_pain": "About a six, so not the worst I've had.",
    "mild_pain": "About a six.",
    "cannot_complete_sentence": "I'm alright talking.",
    "pleuritic_pain": "No, breathing doesn't change it.",
    "recent_onset": "Started this morning.",
    "persistent_vomiting": "No, I haven't been sick.",
    "vomiting_blood": "No, no blood.",
}


def reply_to(check_id: str) -> str:
    return REPLIES.get(check_id, "I'm not sure really.")


def main() -> None:
    print(RULE)
    print("  68F, walk-in, speaking to the tablet")
    print(RULE)
    print(f'\n  "{NARRATIVE}"\n')

    t0 = time.perf_counter()
    ex = seed(NARRATIVE, PROTOCOL)
    took = (time.perf_counter() - t0) * 1000

    print(f"  TIER        {ex.tier.upper()}{'  (degraded)' if ex.degraded else ''}")
    print(f"  EXTRACTED   in {took:.0f} ms")
    for did in ex.belief.positives():
        d = PROTOCOL.discriminators[did]
        print(f"                {d.text}  <- {ex.belief.evidence.get(did)}")
    if not ex.belief.positives():
        print("                (nothing accepted)")
    for r in ex.rejected[:4]:
        print(f"  REJECTED    {r[:110]}")

    ranked = sorted(ex.branch_weights.items(), key=lambda kv: -kv[1])[:4]
    print("  BRANCHES    " + ", ".join(f"{b} {w:.2f}" for b, w in ranked))

    belief = ex.belief
    plausible = plausible_set(ex.branch_weights)
    d = decide(belief, plausible)
    print(f"\n  CATEGORY    {d.category.label}")
    print(f"  CONFIDENCE  {d.confidence}")

    # One question, chosen by the solver -- not by the model.
    from triage import rank, should_stop

    asked = 0
    while not should_stop(belief, plausible, asked) and asked < 3:
        ranked = rank(belief, plausible, limit=8)
        # Only ASK checks are questions. A saturation probe is a measurement and
        # a conscious level is an observation -- asking a patient for either is
        # both useless and slightly absurd.
        askable = [c for c in ranked
                   if PROTOCOL.discriminators[c.discriminator_id].source is Source.ASK]
        for c in ranked[:3]:
            if c not in askable:
                d0 = PROTOCOL.discriminators[c.discriminator_id]
                print(f"\n  -> {d0.text}  [{d0.source.value}, for staff, not a question]")
        if not askable:
            print("\n  Nothing further the patient can answer.")
            break
        best = askable[0]
        check = PROTOCOL.discriminators[best.discriminator_id]
        question = render(best.discriminator_id, PROTOCOL, age=68)
        answer = reply_to(best.discriminator_id)
        outcome, evidence = parse(best.discriminator_id, question, answer, PROTOCOL)

        asked += 1
        print(f"\n  [{asked}] {question}")
        print(f"      -> \"{answer}\"")
        print(f"      -> {check.id} = {outcome.value.upper()}")

        if outcome.value != "unknown":
            belief = belief.record(best.discriminator_id, outcome, evidence)
        else:
            break

        d = decide(belief, plausible)
        print(f"      {d.category.label}  |  {d.confidence}")

    print(f"\n{RULE}")
    final = decide(belief, plausible)
    print(f"  {final.explain(PROTOCOL)}")
    print(f"  {final.confidence}")
    print(f"\n  Extraction ran on the {ex.tier} tier. The category did not come from")
    print("  a model on any tier -- it is a table walk over confirmed positives.")
    print(RULE)


if __name__ == "__main__":
    main()
