"""Run the evaluation.

    python run_sim.py                 # headline comparison
    python run_sim.py --sweep         # sensitivity to the atypical rate
    python run_sim.py --surge         # mass-casualty scenario
    python run_sim.py --sanity        # checks the comparison isn't rigged

Both arms see the identical patient stream. Only the triage policy differs.
"""

import argparse
from dataclasses import replace

from sim import Assistant, Baseline, Cohort, Department, compare, generate, run, summarise

RULE = "-" * 74


def banner(text: str) -> None:
    print(f"\n{RULE}\n  {text}\n{RULE}")


def assumptions(cohort: Cohort, dept: Department) -> None:
    util = cohort.arrivals_per_hour / dept.capacity_per_hour
    print(f"  {cohort.n} arrivals at {cohort.arrivals_per_hour}/hour  "
          f"| {dept.triage_nurses} triage nurses, {dept.clinicians} clinicians")
    print(f"  clinician capacity {dept.capacity_per_hour:.1f}/hour  ->  "
          f"utilisation {util:.0%}"
          f"{'  (overwhelmed — the scenario under test)' if util >= 1.0 else ''}")
    print(f"  atypical presentations {cohort.p_atypical:.0%}   "
          f"deteriorate while waiting {cohort.p_deteriorate:.0%}   "
          f"seed {cohort.seed}")


def one_run(cohort: Cohort, dept: Department, p_branch_missed: float = 0.05):
    patients = generate(cohort)
    base = summarise("baseline", run(patients, Baseline(), dept))
    asst = summarise(
        "assistant",
        run(patients, Assistant(p_branch_missed=p_branch_missed), dept),
    )
    return base, asst


def headline(cohort: Cohort, dept: Department) -> None:
    banner("HEADLINE")
    assumptions(cohort, dept)
    base, asst = one_run(cohort, dept)
    print()
    print(base.report())
    print()
    print(asst.report())
    banner("DELTA")
    print(compare(base, asst))


def sweep(cohort: Cohort, dept: Department) -> None:
    banner("SENSITIVITY — the result as a function of its weakest assumption")
    print("  p_atypical is how often the presenting complaint points at a branch")
    print("  that does not carry the danger check. We have no solid public number")
    print("  for it, so the honest form of the claim is this curve, not one figure.\n")
    print(f"  {'p_atypical':>11s}  {'baseline':>9s}  {'assistant':>10s}  {'delta':>8s}")
    for p in (0.0, 0.04, 0.08, 0.12, 0.20, 0.30):
        base, asst = one_run(replace(cohort, p_atypical=p), dept)
        d = (asst.critical_under_triage - base.critical_under_triage) * 100
        print(f"  {p:>10.0%}  {base.critical_under_triage:>8.1%}  "
              f"{asst.critical_under_triage:>9.1%}  {d:>+7.1f}pp")
    print("\n  At p_atypical = 0 the two arms should be near-identical: with no")
    print("  atypical presentations there is no wrong branch to be caught on.")


def surge(cohort: Cohort, dept: Department) -> None:
    banner("SURGE — the worst case, not the average one")
    hot = replace(cohort, arrivals_per_hour=30.0, n=1200)
    thin = Department(triage_nurses=1, clinicians=4, seed=dept.seed)
    assumptions(hot, thin)
    base, asst = one_run(hot, thin)
    print()
    print(compare(base, asst))


def sanity(cohort: Cohort, dept: Department) -> None:
    banner("ABLATION — which mechanism is doing the work?")
    print("  Turn each mechanism off and watch the advantage disappear. If it")
    print("  doesn't, the harness is measuring something other than it claims.\n")

    patients = generate(cohort)
    base = summarise("baseline", run(patients, Baseline(), dept))

    rows = [
        ("baseline (one branch, no re-check)", base.critical_under_triage, base.missed_deteriorations),
    ]
    slow = Baseline().triage_minutes
    for label, kwargs in [
        ("assistant, full", dict()),
        ("  ... re-check off", dict(recheck_every=0.0, per_category=False)),
        ("  ... flat 15 min re-check", dict(per_category=False)),
        ("  ... branch parallelism off", dict(parallel=False)),
        ("  ... faster triage off", dict(triage_minutes=slow)),
        ("  ... all three off", dict(parallel=False, recheck_every=0.0, triage_minutes=slow)),
    ]:
        m = summarise("assistant", run(patients, Assistant(**kwargs), dept))
        rows.append((label, m.critical_under_triage, m.missed_deteriorations))

    print(f"  {'':<36s}{'critical':>10s}{'deteriorations':>17s}")
    print(f"  {'':<36s}{'under-triage':>10s}{'missed':>17s}")
    for label, crit, missed in rows:
        print(f"  {label:<36s}{crit:>9.1%}{missed:>17d}")
    print("\n  The flat-15 row is a finding, not a defeat. CTAS publishes an")
    print("  interval per severity and that is what a department can schedule,")
    print("  but checking EVERYONE every 15 minutes does catch more — and the")
    print("  honest reason is that it buys safety with nursing time. Under surge")
    print("  the two converge exactly, because the nurse is saturated and")
    print("  re-checks slip whatever the schedule says. The interval only matters")
    print("  while there is slack.")

    print("\n  Three mechanisms, not two. The third — a shorter nurse encounter —")
    print("  is pure throughput: patients reach a clinician sooner, so fewer of")
    print("  them deteriorate before contact. It is a real gain and a different")
    print("  claim from the headline, so it gets its own row rather than hiding")
    print("  inside the other two. With all three off the assistant should land")
    print("  exactly on the baseline; any gap left is a free advantage.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--surge", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260817)
    args = ap.parse_args()

    cohort = Cohort(n=args.n, seed=args.seed)
    dept = Department()

    if args.sweep or args.all:
        sweep(cohort, dept)
    if args.surge or args.all:
        surge(cohort, dept)
    if args.sanity or args.all:
        sanity(cohort, dept)
    if not (args.sweep or args.surge or args.sanity):
        headline(cohort, dept)
    print()


if __name__ == "__main__":
    main()
