# PatientTriage.ai

An emergency-department triage assistant built around one rule: **no model
ever gets a path to a priority.** Models read speech, photos, and records and
write findings into a belief state; a deterministic engine turns confirmed
findings into a category. A wrong model answer flows through the same
auditable path as a wrong human answer, gets the same staff verification,
and traces back to the exact words (or pixels) that produced it.

```bash
pip install -r requirements.txt
python serve.py                       # the live app -- http://localhost:8000
python -m pytest tests/ -q            # 216 tests, none of which call a model
python run_sim.py --all               # evaluation: headline, sensitivity, surge, ablation
python scripts/run_demo_dataset.py    # 18 patients, twice (normal + 3x surge), scored
```

Full "run this on another machine" instructions are near the bottom. This
README is long because the project is; skip to whatever section you need.

## Problem statement → what we built

This was built against a specific brief. Rather than paraphrase it, here's
what it asked for, quoted directly from the requirements (these same quotes
are cited in the modules that implement them, so nothing here is asserted
without a pointer to code and a test):

> "Vital sign thresholds and symptom weights differ significantly across
> pediatric, adult, and geriatric populations... Solutions that apply a
> single adult-calibrated scoring model across all age groups introduce
> silent safety risk."

→ `triage/cohort.py` classifies every patient into a cohort (neonate / infant
/ child / obstetric / adult / unresolved) before anything is scored, and
`triage/protocols/paediatric_vitals.py` scores children against an age-banded
table from birth (CTAS Appendix G), never the adult chart. An *unknown* age
is a fail-safe — adult thresholds are withheld, not assumed. `flacc.py` adds
a staff-observed pain scale for patients too young to self-report a number.

> "Clinical accountability and liability mean any recommendation must remain
> reviewable and overridable by a licensed clinician, with a clear audit
> trail... Capture at least one clinician override and show what the system
> logs."

→ `triage/audit.py`: a hash-chained ledger (tamper-evident — editing or
deleting an event breaks the chain, and that's tested), a named-clinician
override that can never be anonymous, and a UK-GDPR-shaped data-protection
statement. The initial score is never overwritten, only appended to.

> "The system must monitor patients already in the waiting queue and trigger
> re-assessment if wait time exceeds safe thresholds for their severity
> level or if vitals are re-recorded as worsening."

→ `triage/monitor.py`'s `Board`: a per-severity re-check interval (RED is
continuous, BLUE is every 4 hours — an eightfold difference, not one flat
number), and re-scoring the moment new observations arrive rather than
waiting for the timer. The board never de-escalates on its own — only a
named clinician can lower a priority.

> "...must be usable by non-specialist staff... [and demonstrate] at least
> one ambiguous presentation, one pediatric or geriatric case, and one
> zero-history patient."

→ `demo_cases.py` runs exactly those four cases plus an override, end to
end. The 18-patient dataset in `data/demo_patients.json` (see below) extends
this to a full scored set, including a deteriorating patient and a 3x-surge
replay.

> "...overwhelmed emergency department"

→ `sim/`'s default scenario runs at 110% of clinician capacity on purpose —
see *Evaluation* below for why that's load-bearing, not incidental.

## Two protocol packs

The engine imports neither pack by name — swap the data file and it triages
under a different standard.

| pack | what it is |
|---|---|
| `triage/charts.py` | Illustrative, authored for demonstration. 10 branches, 37 checks. Not the Manchester Triage System, which is copyrighted. |
| `triage/protocols/ctas.py` | **The live app's pack.** Transcribed from the published Canadian Triage and Acuity Scale manual where cited, authored in the same style elsewhere. **20 branches, 58 checks.** See `triage/protocols/CTAS_PROVENANCE.md` for exactly which lines are sourced vs. authored, and the copyright position — **not yet reviewed by a clinician**, and that file says so in those words. |

## The boundary

```
   speech, photos, records
            |
        [ models ]                probabilistic, may be wrong
            |
            v
   ┌────────────────────┐
   │    BeliefState     │        ~58 yes/no checks + the words behind each
   └────────────────────┘
            |
        [ engine ]                deterministic, reproducible, offline
            |
            v
        Category                  Immediate / Very urgent / Urgent / Standard / Non-urgent
```

Every place a model is allowed near this system, it proposes into the belief
state and a human (or the deterministic engine) decides — never the reverse.
That includes the two newest input modes, photos and voice: a photo's
caption is a proposal a nurse must confirm before it becomes a check, and
voice only ever fills the same text box typing would have.

## Modules

| file | does |
|---|---|
| `core.py` | `Category`, `Answer`, `Discriminator`, `Branch`, `Protocol` |
| `charts.py` | Illustrative protocol: 10 branches over 37 shared checks |
| `protocols/ctas.py` | CTAS, transcribed from source: 20 branches over 58 checks |
| `cohort.py` | Age stratification, and the refusal to score an unknown cohort |
| `protocols/paediatric_vitals.py` | CTAS Appendix G, extracted from source: HR and RR by age |
| `flacc.py` | Staff-observed pain scale for patients too young to self-report |
| `pain.py` | Central/peripheral pain-locality split — the same 8-10 score is Orange centrally, Yellow peripherally |
| `observations.py` | Measured vitals → checks, both directions (a normal SpO2 clears the low-sat check, not just leaves it unasked) |
| `monitor.py` | The waiting queue: per-severity re-check, deterioration, escalate-only |
| `audit.py` | Hash-chained ledger, clinician overrides, data-protection posture |
| `actions.py` | Suggested first-line actions per branch/category — always "confirm before acting" |
| `ollama.py` | Local text-model backend — stdlib only, no key, no network |
| `local_stt.py` | Local speech-to-text (Vosk) — the offline fallback voice tier |
| `vision.py` | Photo → proposed caption + candidate check — hosted or local vision model, never a diagnosis |
| `identity_extract.py` | One spoken self-introduction → name/age/sex, for the kiosk's single-utterance check-in |
| `news2.py` | Adult vital-sign scoring — a lookup table, refuses children and pregnancy |
| `belief.py` | The shared belief state. Immutable; every write returns a new one |
| `engine.py` | Category from confirmed positives; probability of each category |
| `voi.py` | Which observation to acquire next, and when to stop |
| `extract.py` | Narrative extraction, question phrasing, reply parsing — three bounded model calls + evidence guards |
| `tripwire.py` | Deterministic red flags. Runs beside the model, never after it |

## The live app: a kiosk and a nurse station, at the same time

`python serve.py`, then:

- **`http://localhost:8000/kiosk.html`** — patient-facing. Name, age, sex,
  a free-text description of what's wrong, optional photo, optional voice.
  **Never shows a category, a confidence band, a branch, or a "why" a
  question was chosen** — the page's own source comment says this is a
  UI-level boundary, not an authentication one, and that's worth knowing
  before treating it as a security control.
- **`http://localhost:8000/`** — nurse-facing. Full clinical detail: every
  check with its provenance, plausible branches, confidence, the audit
  ledger, suggested first-line actions, and a "Kiosk check-ins — not yet
  reviewed" queue that any patient who checked themselves in on a kiosk
  shows up in until a nurse looks at them.

**Multiple patients, multiple kiosks, at once.** Each patient gets their own
`Encounter` guarded by its own lock; a slow model call for one patient never
blocks a second kiosk's keystroke. A shared lock guards only the session
directory and the board itself, and is never held across a model call.
`tests/test_concurrent_sessions.py` proves this with real threads, not a
mock.

Extraction runs on a three-tier ladder everywhere a model is used — hosted
Claude, then a local Ollama model, then a keyword-only fallback that never
calls a network. The nurse screen shows which tier answered; a weaker tier
means more rejections and lower confidence, never confidently-wrong triage.

## Voice input

Every text box a patient or nurse would otherwise type into can be spoken
instead — narrative, replies, and (kiosk only) a single spoken
self-introduction that fills name/age/sex at once via `identity_extract.py`.

Two tiers, tried in order:

1. **Web Speech API** — live, in-browser, free, needs internet (the browser
   ships your audio to its vendor's servers to transcribe it).
2. **Vosk, local** — a small offline model, no network at all. Used
   automatically if the first tier is unavailable or errors.

Speech only ever fills the same box typing would have; nothing is ever
auto-submitted from voice, and a misheard word is one click to clear and
retry (`attachClearButton` pairs with every mic button).

## Photo capture

A patient (or a nurse, directly) can take or upload a photo of a visible
symptom. A vision model — hosted Claude, then a local `qwen2.5vl:7b` — reads
it into a plain-language caption and, only where the CTAS vocabulary already
has a matching visual check (non-blanching rash, widespread hives, active
bleeding), a candidate check. **There is no keyword fallback for a photo** —
if neither vision tier is reachable, the feature says so plainly rather than
inventing a caption.

Nothing from a photo reaches the belief state until a nurse clicks Confirm
on the nurse screen's Photos panel — the same rule as everywhere else a
model gets near this system. Photos are viewable there for as long as the
patient is on the board, even after the kiosk session that captured them has
moved on to the next patient.

"Take a photo" opens a real in-page camera (`getUserMedia` + a live preview
+ a capture button) rather than relying on `<input capture>`, which most
desktop browsers simply ignore.

## The demo dataset

`data/demo_patients.json` — 18 patient records covering the brief's named
cases (ambiguous, paediatric, geriatric, zero-history) plus a spread across
every acuity band and one scripted deterioration, replayed through the
**real** HTTP API exactly like a kiosk would.

```bash
python scripts/run_demo_dataset.py --wave both \
    --csv data/demo_results.csv \
    --presentation-csv data/demo_presentation.csv
```

- `--wave baseline` runs all 18 one at a time.
- `--wave surge` replays the same 18 concurrently at 3x the normal arrival
  rate, dispatched across real worker threads — this is what actually
  exercises the per-encounter locking above, not just a claim about it.
- Each record's `expected.category` is a **floor**, not an exact
  prediction: a `staff_observations` entry (posted via the same
  `/api/observe` a nurse uses) guarantees the category regardless of which
  model tier answers; the live model reading the narrative can only push the
  result *more* urgent, never less, by the engine's own design. The script
  reports that distinction rather than flagging an honest, more-cautious
  model read as a failure.
- `--presentation-csv` produces one row per patient — name, age, gender,
  chief complaint, vitals, the real Q&A the app conducted, expected vs.
  actual category, confidence — meant to be read directly, not pivoted
  first.

## The model layer

Three bounded calls in `extract.py`, each with a fixed contract:

```
seed(transcript)       narrative     ->  risk checks + branch weights
choose_next(shortlist)  ranked checks ->  which one to ask, and how to phrase it
parse(check, reply)     free text     ->  true / false / unclear
```

The solver (`voi.py`) decides *what's worth asking*; a model only picks
among an already-ranked shortlist and phrases the question, or reads the
reply back. Question *choice* never varies between runs — only wording
does, and varying wording is harmless. `identity_extract.py` and
`vision.py` add two more bounded calls, same guards, different inputs
(a self-introduction, a photo).

Two guards make every one of these safe to trust:

- **The schema has no "unknown" case.** A check the model didn't hear is
  simply absent from the response, so it is structurally unable to emit a
  fabricated negative — the most dangerous output this system could
  produce.
- **Every finding carries a verbatim quote, verified against the
  transcript.** Findings whose evidence can't be found are dropped and
  counted in `Extraction.rejected`. Hallucinated evidence fails closed.

`test_paraphrase_is_rejected` pins the behaviour that matters: *"patient
reports crushing chest pain radiating to the jaw"* is rejected against a
transcript that only said *"my chest feels tight."*

## Degradation, verified

The ladder is real, not a slide. With no credentials present:

```
TIER 1  hosted model             -> unavailable, reason recorded
TIER 2  local model (Ollama)     -> tried; if that's also unreachable...
TIER 3  keyword extraction       -> cardiac_pain found, marked degraded
        tripwire (never needs a model) -> ORANGE, "goes into my jaw"
        engine                    -> Very urgent
        FINAL                     -> Very urgent
```

The patient is still correctly escalated with the network unplugged and no
model in the loop. Degraded mode may know *less*; it must never claim to
know *more* — `test_keyword_fallback_never_asserts_a_negative` pins this.

## Real bugs the tests (and live testing) caught

**The tripwire missed the flagship case.** The original `cardiac_radiation`
pattern required the word "chest" near "jaw." But *"pain in my upper
stomach that goes into my jaw"* contains no cardiac word at all — exactly
the inferior MI this system exists to catch, and the pattern sailed past
it. Fixed by anchoring on the **radiation verb** ("goes into," "spreads
to," "radiates") rather than the body part.

**Linear branch-weight discounting wasn't enough.** A real "feeling dizzy,
haven't eaten" case still surfaced eye-exposure and allergy questions
before the relevant vertigo one, because a floor-weighted unrelated
Orange-tier check could nearly keep pace with the true branch under a
linear discount. Fixed by cubing the relevance ratio in `voi.py`.

**Sex-only exclusion missed age.** A 6-month-old, recorded female, with
"baby has a high fever" was asked about vaginal bleeding — the model's own
reasoning cited "a critical safety rule-out for pregnancy complications."
Sex correctly excluded nothing; nothing was checking age at all. Fixed with
an age floor (`MINIMUM_PLAUSIBLE_PREGNANCY_AGE`) *and* a defense-in-depth
fix — the model had literally been told the patient's age as "0.5" and
reasoned past it; `_describe_age()` now renders it as "6 months old (an
infant)" everywhere a model sees an age.

**Accepted kiosk patients never left the intake queue.** `Encounter`
objects are never removed from memory at Accept, only reset when that same
kiosk starts its next patient — so `queue_rows()` needed an explicit
`ref not in board.patients` exclusion, which it didn't have.

**A re-checked, already-admitted patient never reached the waiting
board.** `Board.observe()` existed and was fully implemented, but nothing
in `serve.py` ever called it — a nurse recording a fresh, more severe
finding on a patient already on the board had no effect on their place in
the queue. Every write path now calls `Encounter._sync_board()`.

## Why fifty branches cost no more than one

Within a branch the first match wins, ordered most-urgent-first — so a
branch returns its *most urgent* matching check. Across plausible branches,
take the most urgent result. The two rules collapse:

> acuity = the most urgent category among all checks that are positive and
> appear on at least one plausible branch

Ordering inside a branch stops mattering once you take a max across
branches. A fifty-way tree walk becomes one pass over a vocabulary.

## Measured

Pure Python, one core, no optimisation:

| | |
|---|---|
| `decide()` — produce the priority | **19 µs** |
| `acuity_distribution()` | **23 µs** |
| `rank()` — price all remaining checks | **4.5 ms** |

Scoring is microseconds; selection is milliseconds. Both run with the
network unplugged.

## Two calibrations that are load-bearing

**Base rates.** A flat 5% prior across independent checks implies far too
many walk-ins are Very urgent by default — when the prior is that alarmed,
Very urgent is the best guess whatever any answer turns out to be, and
information that cannot change the decision has, correctly, zero value.
`test_prior_case_mix_is_plausible` guards this.

**Loss shape.** Under-triage is penalised at `10^bands`, over-triage at
`2.5^bands`. Asymmetric, not absolute — over-triage is not free, because
calling everyone Immediate in a full department displaces someone who
needed the bay.

Both were found by tests failing, not by inspection.

## Evaluation

`sim/` is a discrete-event department: finite triage nurses and clinicians,
a waiting room, patients who leave unseen. Both arms consume the
**identical** patient stream; only the triage policy differs, and both call
the real engine — this measures the shipped code, not a mock of it.

The default scenario is deliberately **overwhelmed** — 110% of clinician
capacity. An earlier version ran at 73% utilisation, where no queue forms,
priority ordering is inert, and 129 of 138 deteriorating patients were
already seen before they deteriorated: confident numbers on mechanisms the
harness never exercised. `test_default_department_is_actually_overwhelmed`
prevents that regressing.

### Headline (2000 arrivals, seed 20260817)

| | baseline | assistant |
|---|---|---|
| under-triage | 3.2% | **0.1%** |
| critical under-triage | 1.1% | **0.0%** |
| over-triage | 12.9% | 24.1% |
| deteriorations missed | 23 | **2** |
| wait, Very urgent | 47.4 min | **9.7 min** |
| wait, Standard | 45.8 min | 52.2 min |

Two rows go the wrong way, and they stay in the table. Over-triage nearly
doubles — the direct cost of taking the most urgent category any plausible
branch returns. Lower-acuity patients wait *longer*, because urgent ones
correctly move ahead of them. Any version of this table showing only
improvements would be measuring something other than triage.

### Ablation — where the gain actually comes from

```
                                      critical   deteriorations
                                    under-triage        missed
baseline (one branch, no re-check)      1.1%              23
assistant, full                         0.0%               3
  ... re-check off                      0.3%              32
  ... branch parallelism off            0.7%              10
  ... faster triage off                 0.3%               7
  ... all three off                     1.1%              23   <- exactly baseline
```

Three mechanisms, not two — a shorter nurse encounter is pure throughput, a
real gain and a different claim from the headline. The last row landing
exactly on baseline is what says the harness grants no free advantage;
`test_ablation_closes` is the regression.

### Sensitivity

`p_atypical` — how often the presenting complaint points at a branch that
doesn't carry the danger check — has no solid public figure, so the honest
form of the claim is a curve:

```
 p_atypical   baseline   assistant     delta
        0%      0.1%       0.1%     -0.1pp
       12%      1.1%       0.0%     -1.1pp
       30%      2.8%       0.1%     -2.7pp
```

At 0% the arms are identical — with no atypical presentations there's no
wrong branch to be caught on. The gap scaling cleanly with the parameter is
the evidence that the mechanism, not an artefact, produces the result.

### Surge (275% utilisation, one triage nurse)

Under-triage 7.5% → 0.0%, deteriorations missed 51 → 5, Very urgent wait
379 → 132 min. The advantage grows as the department gets worse.

## Environment variables

| variable | default | what it does |
|---|---|---|
| `TRIAGE_PORT` | `8000` | port for `serve.py` |
| `ANTHROPIC_API_KEY` | unset | hosted model tier, if you want it |
| `TRIAGE_MODEL` | `claude-opus-5` | hosted text model |
| `TRIAGE_VISION_MODEL` | `claude-opus-5` | hosted vision model |
| `OLLAMA_HOST` | `http://localhost:11434` | local model server — point this at a remote box if you're not running Ollama on the same machine |
| `OLLAMA_MODEL` | `qwen2.5:7b` | local text model |
| `OLLAMA_VISION_MODEL` | `qwen2.5vl:7b` | local vision model |
| `OLLAMA_TIMEOUT` | `180` | seconds before a local call gives up |
| `VOSK_MODEL_PATH` | `models/vosk-model-small-en-us-0.15` | local speech-to-text model directory |

## Running this on another machine

```bash
git clone <this-repo>
cd accenture_hackathon
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Pick your model tier(s).** The app works with none of these set — it
falls all the way to keyword extraction — but degrades gracefully as you
add capability:

```bash
# Hosted (optional) -- best quality, needs a key and internet
export ANTHROPIC_API_KEY=sk-ant-...

# Local (optional, recommended) -- needs Ollama installed and running
ollama pull qwen2.5:7b          # text extraction
ollama pull qwen2.5vl:7b        # photo captioning

# Local voice fallback (optional) -- only needed if you want offline voice
pip install vosk
curl -LO https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip -d models/
```

**Run it.**

```bash
python serve.py
# open http://localhost:8000/kiosk.html   -- patient check-in
# open http://localhost:8000/             -- nurse dashboard
```

**Verify it.**

```bash
python -m pytest tests/ -q                 # 216 tests, ~4s, no model calls
python scripts/run_demo_dataset.py         # 18 patients through the real API, scored
```

If Ollama is on a different machine on your network:

```bash
export OLLAMA_HOST="http://<that machine's IP>:11434"
export OLLAMA_MODEL="qwen2.5:7b"
export OLLAMA_VISION_MODEL="qwen2.5vl:7b"
python serve.py
```

## Not built yet

- **Real push delivery for critical patients**, beyond sort-to-top-of-queue
  and a visual flash on the nurse screen — no SMS/pager/audible alert.
- **Deployment beyond `localhost`.** Voice and camera capture both require
  a browser "secure context" (HTTPS, or `localhost`) — serving the kiosk
  from a bare HTTP LAN address will silently lose both. This is a
  TLS/deployment question, not a code one.
- **Data at rest.** `ENCOUNTERS` and `BOARD_RECORDS`/`BOARD_PHOTOS` are
  in-memory only, wiped on restart — deliberate for a prototype, not a
  decision for production without a real data-retention story.
- **A medical-imaging-specific vision model.** The current tier uses
  general-purpose vision models (Claude, Qwen2.5-VL); a model fine-tuned on
  clinical photography would likely caption more usefully but wasn't
  evaluated here.
- **Fitting the synthetic evaluation population against real data.** Every
  number in *Evaluation* comes from a population this repo invented; the
  simulator can show the *mechanism* works and how sensitive the result is
  to its assumptions, but `p_atypical` and the case mix aren't grounded in
  a real dataset. MIMIC-IV-ED would be the natural next step.
- **Clinician review of the CTAS transcription.** `CTAS_PROVENANCE.md`
  says this in as many words: not yet reviewed by a clinician.

## Licensing

`charts.py` is authored from general clinical knowledge **for
demonstration**. It is not the Manchester Triage System, which is
copyrighted and licensed to vendors.

`protocols/ctas.py` transcribes selected content from the published CTAS
manual (© Canadian Association of Emergency Physicians). See
`triage/protocols/CTAS_PROVENANCE.md` for exactly which lines are sourced
vs. authored in the same style, and the fair-use position taken for this
prototype — read it before using this pack for anything beyond a demo.

The engine itself is protocol-agnostic: charts are data, the engine is
code. Swap in a licensed set, or ESI, and nothing else changes.

NEWS2 is RCP-owned; free to use with attribution, commercial use and
derivatives need permission.
