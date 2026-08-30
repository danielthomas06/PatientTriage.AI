"""CTAS -- the Canadian Triage and Acuity Scale, transcribed from source.

Unlike the illustrative chart set in ``triage/charts.py``, this is a real,
published, nationally-endorsed protocol. Provenance and the copyright position
are in CTAS_PROVENANCE.md next to this file; read it before shipping anything.

Two things from the manual shaped the engine, not just this data file.

The combination rule is CTAS's own instruction, quoted verbatim:

    "When there are multiple complaints, or conflicting complaints are noted,
     the complaint that will result in the highest appropriate CTAS score is
     the one to be used."

So taking the most urgent result across plausible branches is not a safety
heuristic we bolted on -- it is what the protocol says to do. The only thing we
change is that a nurse must choose the complaint before knowing anything, and
we do not have to.

And the reassessment intervals below are published, not invented. The Round 2
brief mandates monitoring the waiting queue; this is the citation for it.

STRUCTURAL NOTE -- graded modifiers become ordered booleans.

CTAS modifiers are graded (respiratory distress is severe/moderate/mild/none),
where MTS-style discriminators are yes/no. Each grade with a level becomes its
own boolean check here, ordered most-urgent-first within a branch. Because a
branch takes the first positive and the engine takes the max across branches,
that encoding is faithful: "severe distress -> level 1" is exactly
"resp_distress_severe is true -> RED".
"""

from ..core import Branch, Category, Discriminator, Protocol, Source

R, O, Y, G, B = Category.RED, Category.ORANGE, Category.YELLOW, Category.GREEN, Category.BLUE


def _d(id_, text, question, source, prior=0.05):
    return Discriminator(id=id_, text=text, question=question, source=source, prior=prior)


# Priors follow the same discipline as charts.py: they have to resemble a real
# case mix or the value-of-information selector goes silent. If the prior says
# everyone is probably Very urgent, no answer can change the decision, and a
# question that cannot change the decision is correctly worth nothing.
DISCRIMINATORS = {
    d.id: d
    for d in [
        # ---- first-order vital sign modifiers, CTAS level 1 -------------------
        _d("resp_distress_severe", "Severe respiratory distress",
           "Is the patient in severe respiratory distress?", Source.OBSERVE, 0.003),
        _d("spo2_under_90", "Oxygen saturation below 90%",
           "What is the oxygen saturation?", Source.MEASURE, 0.004),
        _d("shock", "Shock -- severe end-organ hypoperfusion",
           "Are there signs of shock?", Source.OBSERVE, 0.003),
        _d("unconscious", "Unconscious (GCS 3-9), unable to protect airway",
           "Is the patient rousable?", Source.OBSERVE, 0.002),

        # ---- first-order, CTAS level 2 ---------------------------------------
        _d("resp_distress_moderate", "Moderate respiratory distress",
           "Is there increased work of breathing?", Source.OBSERVE, 0.012),
        _d("spo2_under_92", "Oxygen saturation below 92%",
           "What is the oxygen saturation?", Source.MEASURE, 0.008),
        _d("haemodynamic_compromise", "Haemodynamic compromise",
           "Is perfusion borderline?", Source.OBSERVE, 0.015),
        _d("altered_loc", "Altered level of consciousness (GCS 10-13)",
           "Is the patient orientated to person, place and time?", Source.OBSERVE, 0.015),

        # ---- first-order, CTAS level 3 ---------------------------------------
        _d("resp_distress_mild", "Mild respiratory distress",
           "Any shortness of breath?", Source.OBSERVE, 0.060),
        _d("spo2_92_to_94", "Oxygen saturation 92-94%",
           "What is the oxygen saturation?", Source.MEASURE, 0.030),
        _d("vitals_at_limits", "Vital signs at the limits of normal for age",
           "Are the vitals at the edge of normal?", Source.MEASURE, 0.090),

        # ---- pain: the composite table, flattened ----------------------------
        # CTAS grades pain on severity x location x onset. Only the acute rows
        # are encoded as discriminators; chronic pain sits one band lower and is
        # carried by `pain_chronic` below.
        _d("severe_pain_central", "Severe central pain (8-10)",
           "How bad is the pain, nought to ten?", Source.ASK, 0.030),
        _d("severe_pain_peripheral", "Severe peripheral pain (8-10)",
           "How bad is the pain, nought to ten?", Source.ASK, 0.035),
        _d("moderate_pain_central", "Moderate central pain (4-7)",
           "How bad is the pain, nought to ten?", Source.ASK, 0.120),
        _d("moderate_pain_peripheral", "Moderate peripheral pain (4-7)",
           "How bad is the pain, nought to ten?", Source.ASK, 0.180),
        _d("mild_pain_central", "Mild central pain (1-3)",
           "How bad is the pain, nought to ten?", Source.ASK, 0.140),
        _d("mild_pain_peripheral", "Mild peripheral pain (1-3)",
           "How bad is the pain, nought to ten?", Source.ASK, 0.200),
        _d("pain_chronic", "Long-standing rather than new pain",
           "Is this new, or a long-standing problem?", Source.ASK, 0.250),

        # ---- fever, adults over 17 -------------------------------------------
        # SIRS at triage is only ever temperature, heart rate and respiratory
        # rate. Bloods do not exist yet, so the white-cell criterion is out of
        # reach and the manual says so.
        _d("fever_immunocompromised", "Fever in an immunocompromised patient",
           "Any condition or treatment affecting immunity?", Source.RECORD, 0.008),
        _d("fever_looks_septic", "Febrile and looks septic (3 SIRS criteria, or organ dysfunction)",
           "Does the patient look septic?", Source.OBSERVE, 0.010),
        _d("fever_looks_unwell", "Febrile, 1-2 SIRS criteria, looks ill",
           "Does the patient look unwell?", Source.OBSERVE, 0.040),
        _d("fever_looks_well", "Febrile but comfortable and in no distress",
           "Do you have a fever?", Source.MEASURE, 0.070),

        # ---- bleeding and mechanism ------------------------------------------
        _d("bleed_life_or_limb", "Life or limb threatening bleed with a bleeding disorder",
           "Is there uncontrolled bleeding?", Source.OBSERVE, 0.003),
        _d("bleed_moderate_minor", "Moderate or minor bleed with a bleeding disorder",
           "Is there bleeding that will not stop?", Source.OBSERVE, 0.010),
        Discriminator(
            id="high_risk_moi", text="High-risk mechanism of injury",
            question="What happened?", source=Source.ASK, prior=0.015,
            leading=("serious", "severe", "badly", "high speed", "high-risk"),
        ),

        # ---- second-order: glucose -------------------------------------------
        _d("glucose_abnormal_symptomatic", "Blood glucose under 3 or over 18, with symptoms",
           "What is the blood glucose?", Source.MEASURE, 0.008),
        _d("glucose_abnormal_silent", "Blood glucose under 3 or over 18, no symptoms",
           "What is the blood glucose?", Source.MEASURE, 0.012),

        # ---- second-order: complaint specific --------------------------------
        Discriminator(
            id="pain_radiating", text="Pain radiating to jaw, neck or arm",
            question="Where exactly do you feel the pain? Does it go anywhere else?",
            source=Source.ASK, prior=0.020,
            leading=("jaw", "neck", "arm", "shoulder", "radiate", "radiating", "spread to"),
        ),
        _d("cardiac_history", "Known cardiac disease, MI, stents or bypass",
           "Any heart problems before?", Source.RECORD, 0.120),
        _d("cannot_complete_sentence", "Unable to talk in sentences",
           "Listen: can the patient finish a sentence in one breath?", Source.OBSERVE, 0.015),
        _d("new_neuro_deficit", "New neurological deficit",
           "Any new weakness, numbness or speech trouble?", Source.ASK, 0.012),
        _d("thunderclap_headache", "Sudden severe headache",
           "Did the headache start suddenly?", Source.ASK, 0.005),
        _d("persistent_vomiting", "Persistent vomiting",
           "Have you been vomiting repeatedly?", Source.ASK, 0.050),
        _d("self_harm_risk", "Risk of self-harm",
           "Any thoughts of harming yourself?", Source.SENSITIVE, 0.010),
        _d("pregnant", "Possible pregnancy",
           "Is there any chance of pregnancy?", Source.SENSITIVE, 0.030),
        _d("infant_fever_under_3m", "Fever in an infant under three months",
           "How old is the baby, and what is the temperature?", Source.MEASURE, 0.002),

        # ---- risk modifiers, free from the record ----------------------------
        _d("anticoagulated", "On anticoagulants or has a bleeding disorder",
           "Are you on blood thinners?", Source.RECORD, 0.100),
        _d("rate_limiting_meds", "On beta-blockers or other rate-limiting medication",
           "Are you on any heart-rate medication?", Source.RECORD, 0.120),
        _d("recent_attendance_72h", "Attended in the last 72 hours",
           "Have you been here in the last three days?", Source.RECORD, 0.050),
        _d("immunosuppressed", "Immunosuppressed",
           "Any condition or treatment affecting immunity?", Source.RECORD, 0.040),

        # ---- second-order: the 8 branches added to reach 20 -------------------
        #
        # Provenance is mixed and each line says which kind it is. Where the
        # manual gives a specific, citable example this is a direct source --
        # marked SOURCED, with the section. Everywhere else it is built from
        # general, well-established emergency-medicine red flags in the same
        # style as the sourced modifiers, because the manual used for this
        # transcription (the Participant's Manual v2.5b) states its own
        # detailed complaint-by-complaint second-order tables live in a
        # separate document: "Of 165 CEDIS complaints, 95 Adult and 102
        # Paediatric complaints have 2nd order modifiers" (Sec 4.3) -- this
        # manual only publishes worked examples for a handful of them. Marked
        # AUTHORED. Neither tier has been checked by a clinician; see
        # CTAS_PROVENANCE.md.

        # Collapse / Syncope
        # SOURCED -- Sec 4.2, Selected Special Complaints, Level 2:
        # "Syncope/presyncope (no prodromal symptoms)"
        _d("syncope_no_warning", "Fainted with no warning symptoms beforehand",
           "Did you have any warning before you fainted, or did it happen "
           "without notice?", Source.ASK, 0.015),
        # AUTHORED -- exertional syncope is a well-established marker for a
        # cardiac cause (e.g. outflow obstruction, exercise-triggered
        # arrhythmia), not a CTAS-quoted level for this complaint.
        _d("syncope_exertional", "Fainted during physical exertion",
           "Were you exercising or exerting yourself when it happened?",
           Source.ASK, 0.008),

        # Vertigo / Dizziness
        # SOURCED -- Sec 4.2, Level 3:
        # "Vertigo (positional, no other neuro symptoms)"
        _d("vertigo_positional_only", "Dizziness only with head movement, no other neuro symptoms",
           "Does it only happen when you move your head, with nothing else "
           "going on -- no weakness, no slurred speech, no numbness?",
           Source.ASK, 0.060),

        # Palpitations
        # SOURCED -- Sec 4.2, Level 2:
        # "Palpitations/irregular heart beat (history of documented lethal)"
        _d("palpitations_lethal_history", "History of a documented dangerous heart rhythm",
           "Have you ever been told you have a dangerous heart rhythm?",
           Source.RECORD, 0.006),

        # Allergic reaction
        # AUTHORED -- airway/breathing/circulation involvement is the
        # standard anaphylaxis red-flag pattern (WHO / Resuscitation Council
        # UK consensus, not CTAS-specific); the shared first-order checks
        # (shock, resp_distress_severe) already catch the confirmed case.
        # This catches the reported-symptom stage, before it is observed.
        _d("throat_or_tongue_swelling", "Throat or tongue swelling, or feels the throat closing",
           "Does your throat or tongue feel swollen, or like it's closing up?",
           Source.ASK, 0.006),
        _d("widespread_hives", "Hives or swelling spreading over the body",
           "Is the rash or swelling spreading, or in more than one place?",
           Source.ASK, 0.035),

        # Rash
        # AUTHORED -- non-blanching rash is a standard meningococcal-sepsis
        # red flag in general EM teaching, not a CTAS-quoted level.
        _d("non_blanching_rash", "Rash does not fade under pressure (non-blanching)",
           "Does the rash still show when you press a glass against it?",
           Source.OBSERVE, 0.004),

        # Eye problems
        # SOURCED -- Sec 4.2, Level 2: "Chemical exposure, eye"
        _d("chemical_eye_exposure", "Chemical splash or exposure to the eye",
           "Did any chemical get into your eye?", Source.ASK, 0.004),
        # AUTHORED -- sudden painless vision loss is a standard ophthalmic
        # emergency marker (retinal detachment / central retinal artery
        # occlusion / giant cell arteritis), not a CTAS-quoted level.
        _d("sudden_vision_loss", "Sudden loss of vision",
           "Did your vision suddenly get worse or go dark?", Source.ASK, 0.005),

        # Substance misuse / overdose / withdrawal
        # AUTHORED -- the dangerous end (reduced consciousness, respiratory
        # depression) is already caught by the shared first-order checks
        # (unconscious, altered_loc, resp_distress_severe). This adds the
        # triage-relevant question a nurse actually needs answered: what, and
        # how much, which is standard toxicology-triage practice rather than
        # a CTAS-quoted level.
        _d("high_risk_ingestion", "Specific high-risk substance, unknown substance, or mixed ingestion",
           "Do you know what was taken, how much, and whether it was more "
           "than one thing?", Source.ASK, 0.010),
        _d("withdrawal_signs", "Tremor or agitation consistent with withdrawal",
           "Any shaking, sweating or agitation that started after stopping "
           "a substance?", Source.OBSERVE, 0.020),

        # Vaginal bleeding / gynaecological, and pregnancy >20 weeks
        # SOURCED -- Sec 4.3.2, the manual's own worked table for
        # "Obstetrical patients with complications of pregnancy > 20 weeks
        # gestation" (adapted from Murray, Bullard, Grafstein & CEDIS NWG,
        # Can J Emerg Med 2004; 6(6):421-7), reproduced here as individual
        # boolean checks rather than the manual's table layout:
        Discriminator(
            id="prolapsed_cord_or_presenting_parts",
            text="Presenting fetal parts or a prolapsed cord",
            question="Can you feel anything at the entrance, like a cord or "
                     "a part of the baby?", source=Source.OBSERVE, prior=0.001),
        _d("bleeding_third_trimester", "Vaginal bleeding, third trimester of pregnancy",
           "How many weeks pregnant are you, and how much bleeding?",
           Source.ASK, 0.003),
        _d("no_fetal_movement_or_heart_tones", "No fetal movement or fetal heart tones",
           "Have you felt the baby move recently?", Source.ASK, 0.002),
        _d("active_labour_frequent", "Active labour, contractions two minutes apart or less",
           "How far apart are the contractions?", Source.OBSERVE, 0.005),
        _d("active_labour", "Active labour, contractions more than two minutes apart",
           "How far apart are the contractions?", Source.OBSERVE, 0.010),
        _d("possible_ruptured_membranes", "Possible leaking amniotic fluid",
           "Has your water broken, or is there fluid leaking?", Source.ASK, 0.006),
        # AUTHORED -- for vaginal bleeding NOT in the >20-week obstetric
        # context above (early pregnancy, or not pregnant at all), the
        # manual gives no specific table. Bleeding heavy enough to soak a
        # pad hourly is a standard general-gynae severity marker, not
        # CTAS-quoted; haemodynamic compromise itself is already caught by
        # the shared `shock` first-order check.
        _d("heavy_bleeding", "Bleeding heavily enough to soak a pad every hour or faster",
           "How heavy is the bleeding -- are you soaking a pad within an hour?",
           Source.ASK, 0.015),
    ]
}


# CTAS first-order modifiers apply to essentially every presenting complaint,
# which is exactly the shared-vocabulary property the engine exploits.
_FIRST_ORDER_L1 = (
    ("resp_distress_severe", R),
    ("spo2_under_90", R),
    ("shock", R),
    ("unconscious", R),
)
_FIRST_ORDER_L2 = (
    ("resp_distress_moderate", O),
    ("spo2_under_92", O),
    ("haemodynamic_compromise", O),
    ("altered_loc", O),
    ("fever_immunocompromised", O),
    ("fever_looks_septic", O),
    ("bleed_life_or_limb", O),
)
_FIRST_ORDER_L3 = (
    ("resp_distress_mild", Y),
    ("spo2_92_to_94", Y),
    ("vitals_at_limits", Y),
    ("fever_looks_unwell", Y),
    ("bleed_moderate_minor", Y),
)
_FIRST_ORDER_LOWER = (
    ("fever_looks_well", G),
    ("pain_chronic", B),
)

# Which locality a branch's pain checks fall under, per the manual's own
# definition (Sec 2.4.2, quoted in full in triage/pain.py): central pain
# "originates within a body cavity (head, chest, abdomen) or organ (eye,
# testicle, deep soft tissue compartment)"; peripheral pain "originates in
# the skin, soft tissues, axial skeleton or superficial organs". Note the
# manual names the eye as a central organ explicitly -- easy to miss and
# easy to get wrong the other way.
#
# This is not cosmetic. Severe central pain scores Orange; the SAME 8-10
# score scored as peripheral scores Yellow -- a full category lower. Pain
# checks are therefore assembled PER BRANCH from this table rather than
# shared globally: putting every locality's pain checks on every branch (an
# earlier version of this pack did exactly that) meant a headache patient's
# "how bad is the pain" answer settled the central checks correctly, but left
# the never-relevant peripheral checks sitting on the branch as still-
# reachable, still-unresolved candidates -- so the ranker would surface the
# identically-worded pain question a second time for a locality nobody was
# ever asked about. Scoping the checks to the branch that actually uses them
# fixes that at the source: an irrelevant locality's checks are no longer
# reachable on that branch at all, not merely deprioritised.
#
# Ambiguous branches (general_unwell, fever, mental_health, substance_misuse)
# default to "central". That is the safer direction: central outranks
# peripheral at the same severity, so genuine uncertainty over-triages rather
# than under-triages, the same asymmetry the rest of the engine is built
# around.
PAIN_LOCALITY: dict[str, str] = {
    "chest_pain": "central",
    "abdominal_pain": "central",
    "shortness_of_breath": "central",
    "general_unwell": "central",
    "fever": "central",
    "altered_loc_complaint": "central",
    "headache": "central",              # head -- explicitly a central site
    "vomiting": "central",
    "seizure": "central",
    "extremity_injury": "peripheral",   # limb -- axial skeleton
    "laceration": "peripheral",         # skin
    "mental_health": "central",
    "collapse": "central",
    "vertigo": "central",               # head-origin
    "palpitations": "central",
    "allergic_reaction": "central",     # airway/systemic risk
    "rash": "peripheral",               # skin, the manual's own example
    "eye_problems": "central",          # eye -- explicitly a central organ
    "substance_misuse": "central",
    "vaginal_bleeding": "central",      # pelvic organ
}

# Checks that presuppose a pregnancy or a vagina to ask about at all. This is
# a genuine exclusion -- not a deprioritisation -- and it is the ONE place in
# this pack that works that way, so the reasoning is worth stating plainly.
#
# Everywhere else, "wider is safer" rules: plausible_set() never fully
# excludes a branch a weak extraction under-weighted, because the cost of
# wrongly excluding a true positive is a missed danger signal. That
# reasoning does not transfer here. Sex, once actually recorded, is not
# noisy extraction the way a branch weight from a keyword match is -- and
# the failure mode of asking anyway is not hypothetical: the branch-weight
# discount in triage/voi.py softens a low-relevance check, it does not
# suppress one whose raw information value is high enough to survive a 10x
# discount, which is exactly what a Red-tier, low-prior obstetric check is
# by design. Observed directly: a 34F headache case had "how many weeks
# pregnant are you" surface as the FIRST follow-up question. For a patient
# recorded male, the same mechanism would ask it regardless of complaint.
#
# What this does NOT do: it does not touch plausible_set(), decide(), or
# the category computation. A record lookup, a direct observation, or the
# patient volunteering it in narrative can still set any of these checks --
# this only stops the interview loop from proactively asking. And it only
# fires on an explicit "M" on record, or an age too young for pregnancy to be
# a routine consideration (see MINIMUM_PLAUSIBLE_PREGNANCY_AGE below), never
# on blank/unknown, because the whole point is that this triggers on an
# affirmative fact, not a guess -- excluding on a guess would be the same
# mistake the rest of the engine is built to avoid, aimed the other way.
#
# The age condition exists because sex alone wasn't the whole gap. Recorded
# case: age 6 months, sex F, "baby is having a high fever" -- and the
# system's own reasoning step asked about vaginal bleeding, justified as "a
# critical safety rule-out for pregnancy complications". Sex=F correctly
# didn't exclude it; nothing was checking age at all. The model was told the
# patient's age and still reasoned past it -- it isn't reliable enough on its
# own to be the only thing standing between an infant and a pregnancy
# question, the same lesson as the original sex-only gap.
OBSTETRIC_GYNAE_ONLY: frozenset[str] = frozenset({
    "pregnant",
    "prolapsed_cord_or_presenting_parts",
    "bleeding_third_trimester",
    "no_fetal_movement_or_heart_tones",
    "active_labour_frequent",
    "active_labour",
    "possible_ruptured_membranes",
    "heavy_bleeding",
})

MINIMUM_PLAUSIBLE_PREGNANCY_AGE = 9
"""Years. Below this, pregnancy is not a routine triage consideration -- the
youngest medically documented cases are pathological outliers around age 5,
not something a general interview should be built around. Deliberately low
rather than tied to a typical menarche age, so this never wrongly excludes a
genuinely plausible young patient; a true case below this age is vanishingly
rare and stays reachable exactly like any other check the interview doesn't
proactively raise -- a direct observation or record can still set it."""


def excludes_obstetric_gynae(sex: str, age: float | None) -> bool:
    """Should the interview loop skip OBSTETRIC_GYNAE_ONLY checks for this
    patient? Two independent, fact-based reasons, either sufficient alone --
    see the block comment above for why this is a genuine exclusion and why
    it only fires on a known fact, never a guess."""
    if sex == "M":
        return True
    if age is not None and age < MINIMUM_PLAUSIBLE_PREGNANCY_AGE:
        return True
    return False

# (severity band -> category) for each locality, from the discriminator
# priors above: severe 8-10, moderate 4-7, mild 1-3.
_PAIN_BANDS = {
    "central": (("severe_pain_central", O), ("moderate_pain_central", Y),
                ("mild_pain_central", G)),
    "peripheral": (("severe_pain_peripheral", Y), ("moderate_pain_peripheral", G),
                   ("mild_pain_peripheral", B)),
}


def _pain_rules(id_: str) -> tuple[tuple[str, Category], ...]:
    return _PAIN_BANDS[PAIN_LOCALITY.get(id_, "central")]


def _branch(id_, name, second_order_urgent=(), second_order_lower=()):
    """Assemble a branch: first-order modifiers first, second-order interleaved.

    The manual's own ordering -- vital sign modifiers are considered first and
    are what the "critical look" is for; second-order modifiers apply after
    first-order have failed to assign a higher acuity.
    """
    return Branch(
        id=id_,
        name=name,
        # Position within the tuple does not affect the decision -- decide()
        # takes the most urgent category per discriminator id across all
        # plausible branches, not the first match -- so the branch's own
        # locality-scoped pain rules are simply appended.
        rules=_FIRST_ORDER_L1
        + tuple(second_order_urgent)
        + _FIRST_ORDER_L2
        + tuple(second_order_lower)
        + _FIRST_ORDER_L3
        + _FIRST_ORDER_LOWER
        + _pain_rules(id_),
    )


BRANCHES = {
    b.id: b
    for b in [
        # THE AMBIGUITY CASE. An inferior MI commonly presents as epigastric pain
        # and is routed to abdominal_pain, where pain_radiating is never asked.
        _branch("chest_pain", "Chest pain",
                second_order_urgent=(("pain_radiating", O), ("cannot_complete_sentence", O)),
                second_order_lower=(("cardiac_history", Y),)),

        _branch("abdominal_pain", "Abdominal pain",
                second_order_urgent=(("persistent_vomiting", O),),
                second_order_lower=(("pregnant", Y),)),

        _branch("shortness_of_breath", "Shortness of breath",
                second_order_urgent=(("cannot_complete_sentence", O),),
                second_order_lower=(("cardiac_history", Y),)),

        # THE GERIATRIC TRAP. Deliberately weak discriminators, and the branch
        # elderly non-specific presentations fall into. Afebrile sepsis and a
        # beta-blocked pulse both read as reassuring here.
        _branch("general_unwell", "General weakness / unwell adult",
                second_order_lower=(("recent_attendance_72h", Y), ("immunosuppressed", Y))),

        _branch("fever", "Fever",
                second_order_urgent=(("infant_fever_under_3m", O),),
                second_order_lower=(("immunosuppressed", Y),)),

        _branch("altered_loc_complaint", "Altered level of consciousness",
                second_order_urgent=(("glucose_abnormal_symptomatic", O),),
                second_order_lower=(("glucose_abnormal_silent", Y),)),

        _branch("headache", "Headache",
                second_order_urgent=(("thunderclap_headache", O), ("new_neuro_deficit", O)),
                second_order_lower=(("anticoagulated", Y),)),

        _branch("vomiting", "Vomiting and/or nausea",
                second_order_urgent=(("persistent_vomiting", O),),
                second_order_lower=(("pregnant", Y),)),

        _branch("seizure", "Seizure",
                second_order_urgent=(("glucose_abnormal_symptomatic", O),),
                second_order_lower=(("glucose_abnormal_silent", Y),)),

        _branch("extremity_injury", "Extremity injury",
                second_order_urgent=(("high_risk_moi", O),),
                second_order_lower=(("anticoagulated", Y),)),

        _branch("laceration", "Laceration / wound",
                second_order_urgent=(("high_risk_moi", O),),
                second_order_lower=(("anticoagulated", Y),)),

        _branch("mental_health", "Mental health concern",
                second_order_urgent=(("self_harm_risk", O),)),

        # ---- the 8 added to bring this pack from 12 to 20 branches --------
        # See the discriminator block above for which lines are sourced from
        # the manual and which are authored in the same style.

        _branch("collapse", "Collapse / syncope",
                second_order_urgent=(("syncope_no_warning", O),
                                      ("syncope_exertional", O)),
                second_order_lower=(("cardiac_history", Y),)),

        _branch("vertigo", "Vertigo / dizziness",
                second_order_urgent=(("new_neuro_deficit", O),),
                second_order_lower=(("vertigo_positional_only", Y),)),

        _branch("palpitations", "Palpitations / irregular heart beat",
                second_order_urgent=(("palpitations_lethal_history", O),),
                second_order_lower=(("cardiac_history", Y),)),

        _branch("allergic_reaction", "Allergic reaction",
                second_order_urgent=(("throat_or_tongue_swelling", O),),
                second_order_lower=(("widespread_hives", Y),)),

        _branch("rash", "Rash",
                second_order_urgent=(("non_blanching_rash", O),)),

        _branch("eye_problems", "Eye problems",
                second_order_urgent=(("chemical_eye_exposure", O),
                                      ("sudden_vision_loss", O))),

        _branch("substance_misuse", "Substance misuse / intoxication / withdrawal",
                second_order_urgent=(("high_risk_ingestion", O),),
                second_order_lower=(("withdrawal_signs", Y),)),

        # THE OBSTETRIC ESCALATION. prolapsed_cord and 3rd-trimester bleeding
        # sit at Level 1 in the manual's own table (Sec 4.3.2) -- this is the
        # one branch in the pack where a second-order modifier outranks
        # every first-order one except the shared Resuscitation list.
        _branch("vaginal_bleeding", "Vaginal bleeding / pregnancy complication",
                second_order_urgent=(("prolapsed_cord_or_presenting_parts", R),
                                      ("bleeding_third_trimester", R),
                                      ("no_fetal_movement_or_heart_tones", O),
                                      ("active_labour_frequent", O)),
                second_order_lower=(("active_labour", Y),
                                     ("possible_ruptured_membranes", Y),
                                     ("heavy_bleeding", Y),
                                     ("pregnant", Y))),
    ]
}


# CTAS target times to physician assessment, and the published reassessment
# intervals for patients still waiting. Both differ from the MTS values baked
# into Category, which is why they live on the protocol.
TARGET_MINUTES = {R: 0, O: 15, Y: 30, G: 60, B: 120}
REASSESSMENT_MINUTES = {R: 0, O: 15, Y: 30, G: 60, B: 120}   # R = continuous

TRIAGE_TARGET_MINUTES = 15   # "triage patients within 10 to 15 minutes of arrival"

# The initial score is a permanent record. Reassessments append; they never
# overwrite. Straight from the manual: "the triage nurse documents the
# reassessment findings and any changes in the patient's acuity score, however,
# the initial triage score is never changed."
INITIAL_SCORE_IS_IMMUTABLE = True


PROTOCOL = Protocol(
    name="CTAS (Canadian Triage and Acuity Scale)",
    discriminators=DISCRIMINATORS,
    branches=BRANCHES,
)
PROTOCOL.validate()
