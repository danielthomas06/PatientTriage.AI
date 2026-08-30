# Protocol pack: CTAS (Canadian Triage and Acuity Scale)

## Source

Transcribed from publicly available CTAS documentation:

- **CTAS Participant's Manual v2.5b, November 2013** — "Combined Adult/Paediatric
  Educational Program". http://ctas-phctas.ca/wp-content/uploads/2018/05/participant_manual_v2.5b_november_2013_0.pdf
- **Revisions to the CTAS Guidelines 2016**, Bullard MJ et al., CJEM.
  https://ctas-phctas.ca/wp-content/uploads/2018/05/revisions_to_the_canadian_emergency_department_triage_and_acuity_scale_ctas_guidelines_2016.pdf
- Paediatric heart/respiratory rate normals in the manual cite:
  Fleming S, Thompson M, Stevens R, Heneghan C. *Normal ranges of heart rate and
  respiratory rate in children from birth to 18 years of age: a systematic review of
  observational studies.* The Lancet 2011; 377(9770): 1011-1019.

## Copyright and permitted use

The CTAS Participant's Manual carries this notice:

> Copyright 2012 Canadian Association of Emergency Physicians (CAEP) with the consent
> of the CTAS National Working Group (NWG). This material cannot be copied or used to
> instruct educational courses without the express permission of CAEP and the CTAS NWG.

**Our position for this prototype:**

- The underlying clinical facts (e.g. "SpO2 below 90% indicates severe respiratory
  distress") are not copyrightable. The manual's specific wording, table layouts and
  the curated complaint x modifier compilation are protected expression.
- We have transcribed a **subset** (20 of the CEDIS presenting complaint list) into our own
  schema, for a non-commercial research prototype, with attribution.
- The total list size has two figures in the manual itself, both quoted rather than picked
  between: the manual's own prose says "Of **165** CEDIS complaints, 95 Adult and 102
  Paediatric complaints have 2nd order modifiers" (Sec 4.3), while mechanically counting
  every line in the manual's own Appendix B (the full complaint list by system) gives
  **177**. The discrepancy is most likely paediatric-specific complaints counted
  differently between the two, not an error in either figure -- we have not resolved it
  further and neither number should be quoted as exact without checking the primary CEDIS
  publication.
- We do not reproduce CAEP tables verbatim as UI assets.
- **Production deployment would require express permission from CAEP and the CTAS NWG.**
- An openly licensed alternative pack (SATS, published by EMSSA for broad adoption) is
  provided so the engine can be demonstrated without this dependency.

## Paediatric ranges: transcribed

`paediatric_vitals.py` carries Appendix G in full -- 25 age rows for respiratory
rate and 25 for heart rate, birth to 18 years.

Extracted mechanically (`pdftotext -raw`) rather than retyped, so there is no
hand-transcription step to get wrong. `-layout` mode interleaved the half-step
sub-rows and shifted the high-side columns out of alignment; `-raw` recovers one
clean row per age. Both tables parsed with zero malformed rows.

Verified two ways, independently of the extraction:

- At birth the heart-rate normal band centres on 127, and the respiratory band on
  44. Those are exactly the medians Fleming et al. report at birth -- the source
  CTAS itself cites. A shifted column would not land on both.
- Every row's six boundaries are monotonic, and rates decline with age.

## The second transcription pass: 12 branches to 20

Eight branches added: collapse/syncope, vertigo/dizziness, palpitations, allergic reaction,
rash, eye problems, substance misuse/overdose/withdrawal, vaginal bleeding & pregnancy
complications >20 weeks.

Provenance is mixed within this pass, and each discriminator in `ctas.py` carries a comment
saying which kind it is:

- **SOURCED** -- a specific, citable example from the manual's Module Four (Sec 4.2
  "Selected Special Complaints" and Sec 4.3.2 "Obstetrical patients... > 20 weeks
  gestation"). Nine discriminators are sourced this way: `syncope_no_warning`,
  `vertigo_positional_only`, `palpitations_lethal_history`, `chemical_eye_exposure`, and the
  six late-pregnancy modifiers (`prolapsed_cord_or_presenting_parts`,
  `bleeding_third_trimester`, `no_fetal_movement_or_heart_tones`, `active_labour_frequent`,
  `active_labour`, `possible_ruptured_membranes`). The pregnancy table is itself adapted in
  the manual from Murray, Bullard, Grafstein & CEDIS NWG, *Can J Emerg Med* 2004;
  6(6):421-7.
- **AUTHORED** -- general, well-established emergency-medicine red flags (anaphylaxis
  airway/circulation involvement, non-blanching rash as a sepsis marker, sudden painless
  vision loss, high-risk/unknown ingestion), built in the same style but not a line from
  this manual. Why: the manual states its own complaint-by-complaint second-order tables
  live in a separate document -- "Of 165 CEDIS complaints, 95 Adult and 102 Paediatric
  complaints have 2nd order modifiers" (Sec 4.3) -- and only publishes worked examples for
  a handful of complaints, reproduced above as SOURCED. The Mental Health second-order
  table (Sec 4.3.3) is also genuinely in the manual and richly detailed, but the existing
  `mental_health` branch does not yet use it beyond `self_harm_risk` -- worth a future pass.

## Still not transcribed

- Remaining ~157 CEDIS complaints beyond the 20 now encoded (see the count discussion
  above for why that number has some uncertainty).
- Dehydration second-order modifier.
- The Mental Health Sec 4.3.3 table, beyond the single `self_harm_risk` check already used.
- A dedicated major-trauma fast path (the manual's Level 1 Critical Look list includes
  cardiac/respiratory arrest and actively-seizing, already covered by the shared
  first-order checks; mechanism-based major trauma criteria -- fall from height, high-speed
  collision, penetrating torso injury -- are not yet promoted out of the single
  `high_risk_moi` checkbox on `extremity_injury`/`laceration`).

## Full CEDIS list, checked against what's encoded

[CEDIS_FULL_LIST.md](CEDIS_FULL_LIST.md) -- every line of the manual's own Appendix B,
mechanically extracted, with the complaints this pack currently encodes checked off. Use
this rather than re-deriving a count by hand when deciding what to transcribe next.

## Clinical review status

**NOT YET REVIEWED BY A CLINICIAN.** This is a transcription by non-clinicians and must
be checked by someone with emergency care training before any claim of clinical validity.
