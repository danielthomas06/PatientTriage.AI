"""Tests for the evaluation harness.

A simulator that quietly favours the thing you are selling is worse than no
simulator, so most of these test *fairness* rather than behaviour. The one
that matters is `test_ablation_closes`: with every mechanism disabled the
assistant must land exactly on the baseline. Any gap is an advantage the
harness is handing over for free, and every number in the report inherits it.
"""

from dataclasses import replace

import pytest

from sim import Assistant, Baseline, Cohort, Department, generate, run, summarise
from triage import Category

SMALL = Cohort(n=400, seed=99)
DEPT = Department()


# ------------------------------------------------------------- reproducibility

def test_same_seed_gives_identical_patients():
    a, b = generate(SMALL), generate(SMALL)
    assert [p.true_category for p in a] == [p.true_category for p in b]
    assert [p.arrival for p in a] == [p.arrival for p in b]
    assert [p.obvious_branch for p in a] == [p.obvious_branch for p in b]


def test_different_seed_gives_a_different_cohort():
    a = generate(SMALL)
    b = generate(replace(SMALL, seed=SMALL.seed + 1))
    assert [p.true_category for p in a] != [p.true_category for p in b]


def test_runs_are_deterministic():
    patients = generate(SMALL)
    first = summarise("x", run(patients, Baseline(), DEPT))
    second = summarise("x", run(patients, Baseline(), DEPT))
    assert first.under_triage == second.under_triage
    assert first.wait_by_truth == second.wait_by_truth


# -------------------------------------------------------------------- fairness

def test_ablation_closes():
    """The load-bearing test. Strip every mechanism; land on the baseline.

    Three mechanisms separate the arms -- parallel branch evaluation,
    re-scoring while waiting, and a shorter nurse encounter. Turn all three
    off and the assistant is the baseline, so the reported gains are entirely
    attributable and none of them is a gift from the harness.
    """
    patients = generate(SMALL)
    base = summarise("baseline", run(patients, Baseline(), DEPT))
    stripped = summarise(
        "assistant",
        run(
            patients,
            Assistant(
                parallel=False,
                recheck_every=0.0,
                triage_minutes=Baseline().triage_minutes,
            ),
            DEPT,
        ),
    )
    assert stripped.critical_under_triage == base.critical_under_triage
    assert stripped.under_triage == base.under_triage
    assert stripped.missed_deteriorations == base.missed_deteriorations
    assert stripped.seen == base.seen


def test_no_atypical_presentations_means_no_branch_advantage():
    """With nothing to be caught on, parallel evaluation should buy nothing."""
    cohort = replace(SMALL, p_atypical=0.0, p_deteriorate=0.0)
    patients = generate(cohort)
    slow = Baseline().triage_minutes
    base = summarise("baseline", run(patients, Baseline(), DEPT))
    asst = summarise(
        "assistant",
        run(patients, Assistant(triage_minutes=slow, recheck_every=0.0), DEPT),
    )
    assert abs(asst.critical_under_triage - base.critical_under_triage) < 0.01


def test_both_arms_see_the_same_patients():
    patients = generate(SMALL)
    a = run(patients, Baseline(), DEPT)
    b = run(patients, Assistant(), DEPT)
    assert [r.patient.id for r in a] == [r.patient.id for r in b]
    assert [r.patient.true_category for r in a] == [r.patient.true_category for r in b]


# ------------------------------------------------------------------- scenario

def test_default_department_is_actually_overwhelmed():
    """The brief says overwhelmed. An idle department measures nothing.

    Below 100% utilisation no queue forms, so priority ordering is inert and
    nobody waits long enough to deteriorate before being seen -- the harness
    would report on mechanisms it never exercised.
    """
    assert Cohort().arrivals_per_hour > DEPT.capacity_per_hour


def test_deterioration_actually_bites_before_contact():
    patients = generate(SMALL)
    records = run(patients, Baseline(), DEPT)
    exposed = [
        r for r in records
        if r.patient.deteriorates_at is not None
        and r.seen_at is not None
        and r.seen_at > r.patient.deteriorates_at
    ]
    assert exposed, "no patient deteriorated before being seen; re-scoring is untested"


# -------------------------------------------------------------------- metrics

def test_under_triage_is_measured_at_contact_not_arrival():
    """A patient who worsened unnoticed counts as under-triaged."""
    patients = generate(SMALL)
    records = run(patients, Baseline(), DEPT)
    worsened = [
        r for r in records
        if r.patient.deteriorates_at is not None
        and r.seen_at is not None
        and r.seen_at > r.patient.deteriorates_at
    ]
    assert any(r.truth_at_contact() < r.patient.true_category for r in worsened)


def test_over_triage_is_reported_alongside_under_triage():
    """Neither number can be gamed without the other moving."""
    patients = generate(SMALL)
    m = summarise("assistant", run(patients, Assistant(), DEPT))
    assert m.over_triage > 0.0, "an arm with zero over-triage is suspicious"
    assert 0.0 <= m.under_triage <= 1.0


def test_assistant_never_lowers_a_priority():
    """Re-scoring escalates only. There is no de-escalation path."""
    policy = Assistant()
    patients = generate(SMALL)
    for p in patients[:120]:
        for assigned in Category:
            raised = policy.recheck(p, assigned, p.arrival + 60.0)
            if raised is not None:
                assert raised < assigned
