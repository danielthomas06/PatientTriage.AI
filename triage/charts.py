"""An illustrative protocol.

IMPORTANT: these branches are authored from general clinical knowledge for
demonstration. They are NOT the Manchester Triage System, which is copyrighted
and licensed to vendors. The engine is protocol-agnostic -- swap this module
for a licensed chart set, or for ESI/CTAS, and nothing else changes.

The shape is what matters:

  * ~10 branches over a vocabulary of ~40 shared discriminators
  * six general discriminators appear on every branch
  * some specific discriminators appear on only one or two

That last point is the entire thesis. ``cardiac_pain`` is on Chest Pain and
not on Abdominal Pain, so a nurse who picks Abdominal Pain never asks it.
"""

from .belief import BeliefState, Evidence
from .core import Answer, Branch, Category, Discriminator, Protocol, Source

R, O, Y, G, B = Category.RED, Category.ORANGE, Category.YELLOW, Category.GREEN, Category.BLUE


def _d(id_, text, question, source, prior=0.05):
    return Discriminator(id=id_, text=text, question=question, source=source, prior=prior)


# Priors are calibrated so that the *prior* acuity distribution resembles a
# real department's case mix -- roughly 2% immediate, 12% very urgent, 35%
# urgent, and the rest standard or below. That calibration is not cosmetic.
#
# With a flat 5% base rate across forty independent checks, "something fires"
# becomes nearly certain and the model implies almost half of all walk-ins are
# Very urgent. When the prior is that alarmed, Very urgent is the best guess no
# matter what any single answer turns out to be -- and a question that cannot
# change the decision has, correctly, zero value. The selector goes silent.
#
# Information is only worth acquiring when it can change what you do, so the
# base rates have to be honest for the selector to work at all.
DISCRIMINATORS = {
    d.id: d
    for d in [
        # --- life threat: observed or measured, never merely asked ---
        _d("airway_compromise", "Airway compromise", "Is the airway obstructed?", Source.OBSERVE, 0.002),
        _d("inadequate_breathing", "Inadequate breathing", "Is breathing inadequate?", Source.OBSERVE, 0.003),
        _d("shock", "Shock", "Are there signs of shock?", Source.MEASURE, 0.003),
        _d("unresponsive", "Unresponsive", "Is the patient unresponsive?", Source.OBSERVE, 0.002),
        _d("currently_fitting", "Currently fitting", "Is the patient fitting now?", Source.OBSERVE, 0.002),
        _d("very_low_spo2", "Very low oxygen saturation", "Oxygen saturation below 92%?", Source.MEASURE, 0.005),

        # --- general discriminators: on every branch ---
        _d("altered_conscious_level", "Altered conscious level", "Are you confused or drowsy?", Source.OBSERVE, 0.015),
        _d("severe_pain", "Severe pain (7-10)", "How bad is the pain, nought to ten?", Source.ASK, 0.040),
        _d("major_haemorrhage", "Uncontrollable major haemorrhage", "Is there heavy bleeding that will not stop?", Source.OBSERVE, 0.004),
        _d("very_hot", "Very hot (39.1C or above)", "Do you have a high fever?", Source.MEASURE, 0.015),
        _d("minor_haemorrhage", "Uncontrollable minor haemorrhage", "Is there bleeding that will not stop?", Source.OBSERVE, 0.015),
        _d("moderate_pain", "Moderate pain (4-6)", "How bad is the pain, nought to ten?", Source.ASK, 0.250),
        _d("hot", "Hot (38.1-39.0C)", "Do you have a fever?", Source.MEASURE, 0.050),
        _d("mild_pain", "Mild pain (1-3)", "How bad is the pain, nought to ten?", Source.ASK, 0.250),
        _d("recent_onset", "Recent problem (under 7 days)", "When did this start?", Source.ASK, 0.550),

        # --- cardiac / chest ---
        _d("cardiac_pain", "Pain radiating to jaw, neck or arm", "Does the pain spread to your jaw, neck or arm?", Source.ASK, 0.020),
        _d("pleuritic_pain", "Pain worse on breathing", "Is the pain worse when you breathe in?", Source.ASK, 0.040),
        _d("cardiac_history", "Known cardiac history", "Any heart problems before?", Source.RECORD, 0.120),
        _d("abnormal_pulse", "Abnormal pulse", "Is the pulse abnormal?", Source.MEASURE, 0.020),

        # --- respiratory ---
        _d("cannot_complete_sentence", "Unable to talk in sentences", "Can you finish a sentence in one breath?", Source.OBSERVE, 0.015),
        _d("low_spo2", "Low oxygen saturation (92-94%)", "Oxygen saturation below 95%?", Source.MEASURE, 0.030),
        _d("wheeze", "Audible wheeze", "Any wheezing?", Source.OBSERVE, 0.030),

        # --- abdominal ---
        _d("rigid_abdomen", "Rigid abdomen", "Is the abdomen rigid?", Source.OBSERVE, 0.008),
        _d("vomiting_blood", "Vomiting blood", "Have you vomited any blood?", Source.ASK, 0.005),
        _d("persistent_vomiting", "Persistent vomiting", "Have you been vomiting repeatedly?", Source.ASK, 0.050),
        _d("pv_bleeding", "Vaginal bleeding", "Any vaginal bleeding?", Source.SENSITIVE, 0.006),

        # --- neurological ---
        _d("new_neuro_deficit", "New neurological deficit", "Any new weakness, numbness or speech trouble?", Source.ASK, 0.012),
        _d("thunderclap_headache", "Sudden severe headache", "Did the headache start suddenly and severely?", Source.ASK, 0.005),
        _d("neck_stiffness", "Neck stiffness", "Is your neck stiff?", Source.ASK, 0.010),
        _d("history_of_unconsciousness", "History of unconsciousness", "Did you black out at all?", Source.ASK, 0.020),

        # --- limb / back ---
        _d("neurovascular_deficit", "Neurovascular deficit in limb", "Is the limb cold, numb or pulseless?", Source.OBSERVE, 0.005),
        _d("gross_deformity", "Gross deformity", "Is the limb visibly deformed?", Source.OBSERVE, 0.015),
        _d("urinary_retention", "Urinary retention or incontinence", "Any trouble passing water?", Source.SENSITIVE, 0.006),

        # --- risk modifiers, mostly free from records ---
        _d("anticoagulated", "On anticoagulants", "Are you on blood thinners?", Source.RECORD, 0.100),
        _d("immunosuppressed", "Immunosuppressed", "Any condition or treatment affecting immunity?", Source.RECORD, 0.040),
        _d("recent_attendance_72h", "Attended in the last 72 hours", "Have you been here in the last three days?", Source.RECORD, 0.050),
        _d("diabetic", "Diabetic", "Are you diabetic?", Source.RECORD, 0.100),
    ]
}

# Six general discriminators sit on every branch, in the same relative order.
_LIFE_THREAT = (
    ("airway_compromise", R),
    ("inadequate_breathing", R),
    ("shock", R),
    ("unresponsive", R),
    ("currently_fitting", R),
    ("very_low_spo2", R),
)
_GENERAL_URGENT = (
    ("altered_conscious_level", O),
    ("major_haemorrhage", O),
    ("severe_pain", O),
    ("very_hot", O),
)
_GENERAL_LOWER = (
    ("minor_haemorrhage", Y),
    ("moderate_pain", Y),
    ("hot", Y),
    ("mild_pain", G),
    ("recent_onset", G),
)


def _branch(id_, name, specific_urgent=(), specific_lower=()):
    """Assemble a branch: life threat, then specifics, then general, in order."""
    return Branch(
        id=id_,
        name=name,
        rules=_LIFE_THREAT + tuple(specific_urgent) + _GENERAL_URGENT
        + tuple(specific_lower) + _GENERAL_LOWER,
    )


BRANCHES = {
    b.id: b
    for b in [
        _branch("chest_pain", "Chest pain",
                specific_urgent=(("cardiac_pain", O), ("cannot_complete_sentence", O), ("abnormal_pulse", O)),
                specific_lower=(("pleuritic_pain", Y), ("cardiac_history", Y))),

        _branch("abdominal_pain", "Abdominal pain in adults",
                specific_urgent=(("rigid_abdomen", O), ("vomiting_blood", O), ("pv_bleeding", O)),
                specific_lower=(("persistent_vomiting", Y),)),

        _branch("breathlessness", "Shortness of breath in adults",
                specific_urgent=(("cannot_complete_sentence", O), ("low_spo2", O)),
                specific_lower=(("wheeze", Y), ("pleuritic_pain", Y))),

        _branch("unwell_adult", "Unwell adult",
                specific_urgent=(),
                specific_lower=(("recent_attendance_72h", Y),)),

        _branch("headache", "Headache",
                specific_urgent=(("thunderclap_headache", O), ("neck_stiffness", O), ("new_neuro_deficit", O)),
                specific_lower=(("history_of_unconsciousness", Y),)),

        _branch("collapse", "Collapse",
                specific_urgent=(("new_neuro_deficit", O), ("abnormal_pulse", O)),
                specific_lower=(("history_of_unconsciousness", Y),)),

        _branch("palpitations", "Palpitations",
                specific_urgent=(("abnormal_pulse", O), ("cardiac_pain", O)),
                specific_lower=(("cardiac_history", Y),)),

        _branch("vomiting", "Vomiting",
                specific_urgent=(("vomiting_blood", O),),
                specific_lower=(("persistent_vomiting", Y),)),

        _branch("limb_problems", "Limb problems",
                specific_urgent=(("neurovascular_deficit", O),),
                specific_lower=(("gross_deformity", Y),)),

        _branch("back_pain", "Back pain",
                specific_urgent=(("new_neuro_deficit", O), ("urinary_retention", O)),
                specific_lower=()),
    ]
}

PROTOCOL = Protocol(
    name="Illustrative presentation protocol (not MTS)",
    discriminators=DISCRIMINATORS,
    branches=BRANCHES,
)
PROTOCOL.validate()


# Resolved by looking, not by asking.
_ACROSS_THE_ROOM = (
    "airway_compromise",
    "inadequate_breathing",
    "unresponsive",
    "currently_fitting",
)

# The same observation under CTAS naming. CTAS folds airway and breathing into
# one graded respiratory-distress modifier, so the ids differ even though the
# clinical act -- looking at a patient who just walked in -- is identical.
_ACROSS_THE_ROOM_CTAS = (
    "resp_distress_severe",
    "unconscious",
    "shock",
)


def walk_in_baseline(protocol: Protocol | None = None) -> BeliefState:
    """The across-the-room look, expressed as data.

    A patient who walks in and speaks in sentences has a patent airway, is
    breathing, and is responsive. Real triage settles that in the first seconds
    by looking; it is never a queued question.

    Seeding it is not a convenience. Left unresolved, the life-threat checks
    dominate the selector purely through the cost of being wrong about them,
    and every ranking opens with "is this patient in respiratory arrest?" --
    correct arithmetic, useless product.

    Note what is deliberately *not* cleared: shock and oxygen saturation cannot
    be judged across a room. They stay open, so the selector's first move is to
    ask for vital signs, which is exactly right.
    """
    protocol = protocol or PROTOCOL
    belief = BeliefState(protocol)
    seen = Evidence("walked in, talking in sentences", "staff")
    for did in _ACROSS_THE_ROOM:
        # Packs name these differently — CTAS folds airway and breathing into
        # graded respiratory distress, so only seed what this pack actually has.
        if did in protocol.discriminators:
            belief = belief.record(did, Answer.FALSE, seen)
    for did in _ACROSS_THE_ROOM_CTAS:
        if did in protocol.discriminators:
            belief = belief.record(did, Answer.FALSE, seen)
    return belief
