"""Replay data/demo_patients.json through the REAL application and print a
scorecard, then (optionally) replay it again at N times the arrival rate to
show queueing behaviour under surge.

    python scripts/run_demo_dataset.py                    # baseline only
    python scripts/run_demo_dataset.py --wave surge        # surge only
    python scripts/run_demo_dataset.py --wave both          # both (default)

Requires serve.py already running (python serve.py) -- this hits its real
HTTP API exactly the way the kiosk does, so every result on screen is what
the nurse dashboard at http://localhost:8000 is showing too. Nobody is
accepted onto the board; every patient stays in the "not yet reviewed" kiosk
intake queue, which is the panel built for exactly this.

Deliberately stdlib-only (urllib), matching serve.py's own dependency-free
posture -- this is a demo script, not something worth a requirements.txt for.
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASET = Path(__file__).resolve().parent.parent / "data" / "demo_patients.json"
MAX_QUESTIONS = 4    # a safety cap on the follow-up loop -- should_stop() ends it
                      # sooner for anything Orange/Red; Yellow-and-below records
                      # (nothing left for should_stop to halt on) run the full loop,
                      # so this also bounds worst-case runtime against a slow remote model
CALL_TIMEOUT = 180   # a real remote model (e.g. a 27B model over a home network) can be slow

# Ordinal urgency, most urgent first -- matches Category's own ordering (RED is
# smallest). `expected.category` is a FLOOR, not an exact prediction: extra
# positives the live model reads directly out of free narrative (which this
# script cannot script the absence of) can only ever push decide() to something
# AT LEAST as urgent, never less, by the engine's own never-lowers design. A
# result strictly more urgent than expected is a real, honest model reading --
# not a bug -- so it is reported separately from an actual under-triage.
_ORDER = ["RED", "ORANGE", "YELLOW", "GREEN", "BLUE"]


def _call(base_url: str, method: str, path: str, body: dict | None = None) -> dict:
    url = f"{base_url}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                  headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=CALL_TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read())


def _post(base_url: str, path: str, body: dict) -> dict:
    return _call(base_url, "POST", path, body)


def _get(base_url: str, path: str) -> dict:
    return _call(base_url, "GET", path)


# --------------------------------------------------------------------------
# one patient, start to finish
# --------------------------------------------------------------------------

def run_patient(base_url: str, record: dict, *, verbose: bool = True) -> dict:
    origin = record.get("origin", "kiosk")
    ref = _post(base_url, "/api/patient/new", {"origin": origin})["ref"]

    identity = dict(record["identity"])
    _post(base_url, "/api/identity", {"ref": ref, **identity})
    _post(base_url, "/api/narrative", {"ref": ref, "text": record["narrative"]})

    if "vitals" in record:
        _post(base_url, "/api/vitals", {"ref": ref, **record["vitals"]})
    if "flacc" in record:
        _post(base_url, "/api/flacc", {"ref": ref, **record["flacc"]})
    for obs in record.get("staff_observations", []):
        _post(base_url, "/api/observe", {
            "ref": ref, "check": obs["check_id"],
            "positive": obs["positive"], "note": obs["note"],
        })

    scripted = record.get("scripted_answers", {})
    default_reply = record.get("default_reply", "No.")
    asked = []
    for _ in range(MAX_QUESTIONS):
        nxt = _get(base_url, f"/api/next?ref={ref}").get("next") or {}
        if nxt.get("stopped") or not nxt.get("question"):
            break
        check = nxt["check"]
        reply = scripted.get(check, default_reply)
        asked.append({"check": check, "question": nxt["question"], "reply": reply})
        _post(base_url, "/api/answer", {"ref": ref, "check": check, "reply": reply})

    state = _get(base_url, f"/api/state?ref={ref}")["encounter"]
    expected = record["expected"]["category"]
    actual = state["category"]
    at_least = _ORDER.index(actual) <= _ORDER.index(expected)
    result = {
        "id": record["id"], "ref": ref, "tags": record.get("tags", []),
        "expected": expected,
        "actual": actual, "label": state["label"],
        "confidence": state["confidence"]["band"], "tier": state["tier"],
        "questions_asked": asked,
        "match": actual == expected,
        "at_least_as_urgent": at_least,
        "over_triaged": at_least and actual != expected,
    }

    deterioration = record.get("deterioration")
    if deterioration:
        time.sleep(deterioration.get("delay_seconds", 1))
        for obs in deterioration.get("staff_observations", []):
            _post(base_url, "/api/observe", {
                "ref": ref, "check": obs["check_id"],
                "positive": obs["positive"], "note": obs["note"],
            })
        state2 = _get(base_url, f"/api/state?ref={ref}")["encounter"]
        result["deterioration"] = {
            "expected": deterioration["expected"]["category"],
            "actual": state2["category"], "label": state2["label"],
            "match": state2["category"] == deterioration["expected"]["category"],
        }

    if verbose:
        if result["match"]:
            mark = "OK"
        elif result["over_triaged"]:
            mark = "over-triaged (safer direction, not a failure)"
        else:
            mark = "**UNDER-TRIAGED**"
        print(f"  {result['id']:<32s} expected {result['expected']:<7s} "
              f"-> actual {result['actual']:<7s} ({result['label']:<12s})  {mark}")
        if "deterioration" in result:
            d = result["deterioration"]
            dmark = "OK" if d["match"] else "**MISMATCH**"
            print(f"    {'':<30s} after deterioration: expected {d['expected']:<7s} "
                  f"-> actual {d['actual']:<7s} ({d['label']:<12s})  {dmark}")
    return result


# --------------------------------------------------------------------------
# waves
# --------------------------------------------------------------------------

def run_baseline(base_url: str, patients: list[dict], delay: float) -> list[dict]:
    print(f"\n{'='*78}\n  BASELINE -- {len(patients)} patients, one at a time, "
          f"real narrative -> extraction -> decide() pipeline\n{'='*78}")
    results = []
    for i, record in enumerate(patients):
        results.append(run_patient(base_url, record))
        if i < len(patients) - 1:
            time.sleep(delay)
    return results


def run_surge(base_url: str, patients: list[dict], baseline_rate: float,
              multiplier: float, speed: float, workers: int) -> list[dict]:
    """Replay the same 18 patients at `multiplier`x the normal arrival rate,
    dispatched concurrently so a slow model call on one patient's kiosk can
    never block another's -- the per-encounter locking built earlier in this
    project is exactly what makes this safe to fire this way."""
    interval = 3600.0 / (baseline_rate * multiplier)   # real hospital seconds/arrival
    interval /= speed                                    # compressed for a watchable demo

    print(f"\n{'='*78}\n  SURGE -- {multiplier:.0f}x normal volume "
          f"({baseline_rate * multiplier:.0f}/hr vs {baseline_rate:.0f}/hr baseline), "
          f"{len(patients)} arrivals, one every {interval:.1f}s on screen\n{'='*78}")

    results: list[dict] = []
    lock = threading.Lock()

    def go(record):
        r = run_patient(base_url, record)
        with lock:
            results.append(r)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record in patients:
            pool.submit(go, record)
            time.sleep(interval)
    return results


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def summarise(label: str, results: list[dict]) -> None:
    total = len(results)
    exact = sum(1 for r in results if r["match"])
    at_least = sum(1 for r in results if r["at_least_as_urgent"])
    critical = sum(1 for r in results if r["actual"] in ("RED", "ORANGE"))
    print(f"\n  {label}: {exact}/{total} exact match, {at_least}/{total} at least "
          f"as urgent as the guaranteed floor ({critical} flagged critical)")
    for r in results:
        if r["over_triaged"]:
            print(f"    ~ {r['id']}: floor was {r['expected']}, live model read it as "
                  f"{r['actual']} -- more cautious, not wrong ({r['tier']} tier)")
        elif not r["at_least_as_urgent"]:
            print(f"    ! {r['id']}: UNDER-TRIAGED -- floor was {r['expected']}, "
                  f"got {r['actual']} ({r['tier']} tier)")


CSV_COLUMNS = [
    "wave", "id", "tags", "expected_category", "actual_category", "label",
    "exact_match", "at_least_as_urgent", "over_triaged", "confidence_band",
    "tier", "questions_asked", "ref",
    "deteriorated", "deterioration_expected", "deterioration_actual",
    "deterioration_match",
]


def csv_rows(all_results: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for wave, results in all_results.items():
        for r in results:
            det = r.get("deterioration")
            rows.append({
                "wave": wave, "id": r["id"], "tags": ";".join(r["tags"]),
                "expected_category": r["expected"], "actual_category": r["actual"],
                "label": r["label"], "exact_match": r["match"],
                "at_least_as_urgent": r["at_least_as_urgent"],
                "over_triaged": r["over_triaged"], "confidence_band": r["confidence"],
                "tier": r["tier"], "questions_asked": len(r["questions_asked"]),
                "ref": r["ref"],
                "deteriorated": det is not None,
                "deterioration_expected": det["expected"] if det else "",
                "deterioration_actual": det["actual"] if det else "",
                "deterioration_match": det["match"] if det else "",
            })
    return rows


def write_csv(all_results: dict[str, list[dict]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows(all_results))


# --------------------------------------------------------------------------
# a single wide, human-readable table -- one row per patient, meant to be
# read directly rather than pivoted first
# --------------------------------------------------------------------------

PRESENTATION_COLUMNS = [
    "patient_id", "ref", "wave", "name", "age", "gender", "chief_complaint",
    "vitals", "clinical_observations", "questions_and_answers",
    "expected_category", "actual_category", "actual_label", "confidence",
    "result", "model_tier",
]

_VITAL_LABELS = {
    "respiratory_rate": "RR", "spo2": "SpO2 (%)", "on_oxygen": "On O2",
    "systolic_bp": "Systolic BP", "pulse": "Pulse", "temperature": "Temp (C)",
    "alert": "Alert (AVPU)", "pain_score": "Pain score", "looks_unwell": "Looks unwell",
}


def _label(check_id: str) -> str:
    return check_id.replace("_", " ").capitalize()


def _format_age(identity: dict) -> str:
    months = identity.get("age_months")
    if months not in (None, ""):
        return f"{months} months"
    years = identity.get("age")
    if years in (None, ""):
        return "unknown"
    return str(years)


def _format_vitals(vitals: dict) -> str:
    if not vitals:
        return ""
    return "; ".join(f"{_VITAL_LABELS.get(k, k)}: {v}" for k, v in vitals.items())


def _format_observations(observations: list[dict], flacc: dict | None) -> str:
    parts = [
        f"{_label(o['check_id'])} ({'positive' if o['positive'] else 'negative'}) -- {o['note']}"
        for o in observations
    ]
    if flacc:
        parts.append("FLACC (staff-observed pain, pre-verbal patient): " +
                      ", ".join(f"{k}={v}" for k, v in flacc.items()))
    return " | ".join(parts)


def _format_qa(asked: list[dict]) -> str:
    if not asked:
        return "(none -- category settled before any follow-up question was needed)"
    return " | ".join(f"Q: {a['question']}  ->  A: {a['reply']}" for a in asked)


def presentation_rows(patients: list[dict], all_results: dict[str, list[dict]]) -> list[dict]:
    by_id = {p["id"]: p for p in patients}
    rows = []
    for wave, results in all_results.items():
        for r in results:
            record = by_id[r["id"]]
            identity = record["identity"]
            if r["match"]:
                result = "Exact match"
            elif r["over_triaged"]:
                result = "Over-triaged (safer direction)"
            elif r["at_least_as_urgent"]:
                result = "At/above floor"
            else:
                result = "UNDER-TRIAGED"
            rows.append({
                "patient_id": r["id"], "ref": r["ref"], "wave": wave,
                "name": identity.get("name") or "Unidentified",
                "age": _format_age(identity),
                "gender": identity.get("sex") or "unknown",
                "chief_complaint": record["narrative"],
                "vitals": _format_vitals(record.get("vitals", {})),
                "clinical_observations": _format_observations(
                    record.get("staff_observations", []), record.get("flacc")),
                "questions_and_answers": _format_qa(r["questions_asked"]),
                "expected_category": r["expected"],
                "actual_category": r["actual"],
                "actual_label": r["label"],
                "confidence": r["confidence"],
                "result": result,
                "model_tier": r["tier"],
            })
    return rows


def write_presentation_csv(patients: list[dict], all_results: dict[str, list[dict]],
                            path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PRESENTATION_COLUMNS)
        writer.writeheader()
        writer.writerows(presentation_rows(patients, all_results))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--wave", choices=["baseline", "surge", "both"], default="both")
    ap.add_argument("--delay", type=float, default=2.0,
                     help="seconds between baseline arrivals")
    ap.add_argument("--surge-multiplier", type=float, default=3.0)
    ap.add_argument("--surge-speed", type=float, default=20.0,
                     help="compress real hospital timing by this much, purely for watchability")
    ap.add_argument("--surge-workers", type=int, default=4,
                     help="concurrent kiosks -- matches the per-encounter locking design")
    ap.add_argument("--patients", type=Path, default=DATASET)
    ap.add_argument("--out", type=Path, default=None,
                     help="optional path to save full JSON results")
    ap.add_argument("--csv", type=Path, default=None,
                     help="optional path to save a flat CSV scorecard")
    ap.add_argument("--presentation-csv", type=Path, default=None,
                     help="optional path to save a wide, one-row-per-patient CSV "
                          "(name/age/gender/complaint/vitals/Q&A/category) for a demo audience")
    args = ap.parse_args()

    dataset = json.loads(args.patients.read_text())
    patients = dataset["patients"]
    baseline_rate = dataset.get("baseline_arrivals_per_hour", 15.0)

    try:
        _get(args.base_url, "/api/config")
    except Exception as exc:
        raise SystemExit(
            f"Can't reach {args.base_url} -- is `python serve.py` running? ({exc})"
        )

    all_results: dict[str, list[dict]] = {}
    if args.wave in ("baseline", "both"):
        all_results["baseline"] = run_baseline(args.base_url, patients, args.delay)
        summarise("BASELINE", all_results["baseline"])
    if args.wave in ("surge", "both"):
        all_results["surge"] = run_surge(
            args.base_url, patients, baseline_rate,
            args.surge_multiplier, args.surge_speed, args.surge_workers,
        )
        summarise("SURGE", all_results["surge"])

    print(f"\n{'='*78}\n  Open {args.base_url}/ (the nurse dashboard) and look at "
          f"\"Kiosk check-ins -- not yet reviewed\" to see every result above laid "
          f"out live, colour-coded by category. Nobody was accepted onto the "
          f"board, so all of them are still sitting in that queue.\n{'='*78}")

    if args.out:
        args.out.write_text(json.dumps(all_results, indent=2))
        print(f"\n  Full results saved to {args.out}")
    if args.csv:
        write_csv(all_results, args.csv)
        print(f"  CSV scorecard saved to {args.csv}")
    if args.presentation_csv:
        write_presentation_csv(patients, all_results, args.presentation_csv)
        print(f"  Presentation CSV saved to {args.presentation_csv}")


if __name__ == "__main__":
    main()
