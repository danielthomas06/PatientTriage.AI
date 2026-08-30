"""Generate the nurse-station scenario FROM the engine.

The screen was showing hand-written strings, which makes it a mock-up: nothing
stops the UI drifting away from what the system actually does. This runs the real
encounter through the real engine and emits the categories, confidences and
ledger entries it actually produces.

    python tools/build_station.py > tools/station_data.json

Re-run it after changing a protocol and the screen stays truthful, or breaks
loudly. A UI that cannot drift from its engine is worth the extra file.
"""

import json
import pathlib
import sys

# Run from anywhere; the package lives one level up.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from triage import Answer, BeliefState, Evidence, decide, plausible_set
from triage.audit import EventKind, Ledger, ReasonCode
from triage.charts import PROTOCOL

NARRATIVE = ("I've had this pain in my upper stomach since about six this morning. "
             "I feel sick and I'm a bit sweaty. It's not going away.")

WEIGHTS = {"abdominal_pain": 0.80, "chest_pain": 0.20, "unwell_adult": 0.10}
PLAUSIBLE = plausible_set(WEIGHTS)

# Each entry: what gets recorded at this step, and how the UI should narrate it.
SCRIPT = [
    dict(note="She arrives and speaks. Nothing extracted yet, so the engine has "
              "nothing positive to work with.",
         record=[], said=False),

    dict(note="Extraction runs on the local model. Every check carries the exact "
              "words behind it — a finding whose quote is not in the transcript "
              "is dropped, not trusted.",
         record=[("moderate_pain", Answer.TRUE,
                  "pain in my upper stomach since about six this morning", "spoken")],
         said=True),

    dict(note="An assistant takes vitals alongside the conversation — not the "
              "triage nurse, and not in sequence after her.",
         record=[("very_low_spo2", Answer.FALSE, "96% on air", "measured"),
                 ("shock", Answer.FALSE, "BP 138/82, capillary refill under 2s", "measured")],
         said=True),

    dict(note="The solver picks the next observation. It is a question this time — "
              "a saturation probe would go to a device, a conscious level to staff. "
              "Note the wording: it never names the jaw.",
         record=[], said=True,
         ask=dict(actor="patient",
                  q="Where exactly do you feel the pain? Does it go anywhere else?")),

    dict(note="She answers. The check that fires belongs to Chest pain — ranked "
              "second. A nurse who committed to Abdominal pain would never have "
              "asked it.",
         record=[("cardiac_pain", Answer.TRUE, "it goes up into my jaw a bit", "spoken")],
         said=True, decided=True,
         ask=dict(actor="patient",
                  q="Where exactly do you feel the pain? Does it go anywhere else?",
                  a="\u201cUp here under my ribs. And it goes up into my jaw a bit.\u201d")),
]

SRC_OF = {}
steps = []
belief = BeliefState(PROTOCOL)
ledger = Ledger("ED-2026-0824-0117")
checks = []

for entry in SCRIPT:
    for cid, answer, quote, src in entry["record"]:
        belief = belief.record(cid, answer, Evidence(quote, "speech"))
        SRC_OF[cid] = src
        checks.append(dict(
            name=PROTOCOL.discriminators[cid].text,
            src=src, quote=quote, negative=answer is Answer.FALSE,
        ))
        ledger.append(EventKind.OBSERVED, "system",
                      check=cid, value=answer.value, quote=quote)

    d = decide(belief, PLAUSIBLE)
    if entry.get("decided"):
        ledger.record_decision(d, evidence=belief.trace())
        ledger.append(EventKind.SHOWN, "system", to="rn.k.mensah")

    fired_on = [PROTOCOL.branches[b].name for b in d.fired_on]
    rank_note = ""
    if d.fired_on:
        order = sorted(WEIGHTS, key=lambda b: -WEIGHTS[b])
        pos = min((order.index(b) for b in d.fired_on if b in order), default=None)
        if pos is not None and pos > 0:
            rank_note = f" — ranked #{pos + 1} in plausibility"

    steps.append(dict(
        note=entry["note"],
        said=NARRATIVE if entry["said"] else None,
        checks=list(checks),
        ask=entry.get("ask"),
        cat=d.category.name,
        label=d.category.label,
        band=d.confidence.band,
        why=(f"could still be {d.confidence.worst_case.label} — "
             f"{d.confidence.unresolved} checks unresolved")
            if d.confidence.could_escalate else "nothing unresolved could raise this",
        fired=PROTOCOL.discriminators[d.fired].text if d.fired else None,
        fired_on=", ".join(fired_on) + rank_note if fired_on else None,
        decided=bool(entry.get("decided")),
    ))

final = decide(belief, PLAUSIBLE)
ledger.record_override(
    clinician="rn.k.mensah", recommended=final.category,
    chosen=__import__("triage").Category.RED,
    reason=ReasonCode.CLINICAL_JUDGEMENT,
    note="grey, clammy, looks unwell beyond what the checks capture",
    decision=final, evidence=belief.trace(),
)

out = dict(
    narrative=NARRATIVE,
    steps=steps,
    recommended=final.category.name,
    ledger=[dict(seq=e.seq, kind=str(e.kind), actor=e.actor,
                 payload={k: v for k, v in e.payload.items() if k != "evidence"},
                 prev=e.prev_hash[:8], digest=e.digest[:8])
            for e in ledger.events],
    targets={c.name: c.target_minutes for c in __import__("triage").Category},
)
json.dump(out, sys.stdout, indent=1, default=str)
