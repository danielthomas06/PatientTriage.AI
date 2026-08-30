"""The waiting queue.

The brief mandates monitoring with two triggers -- an interval keyed to severity,
and re-recorded vitals that worsen. These check both, and check that the board
cannot do the one thing only a human may do.
"""

import pytest

from triage import Answer, BeliefState, Category, Evidence, plausible_set
from triage.charts import PROTOCOL
from triage.monitor import (
    REASSESS_MINUTES, STARVATION_MULTIPLE, Board, Trigger,
)

PLAUSIBLE = plausible_set({"abdominal_pain": 0.8, "chest_pain": 0.2})


def board_with(category=Category.GREEN, checks=()):
    b = BeliefState(PROTOCOL)
    for cid in checks:
        b = b.record(cid, Answer.TRUE, Evidence("simulated", "staff"))
    board = Board(PROTOCOL)
    board.admit("bed-1", category, b, PLAUSIBLE, now=0.0)
    return board


# ---------------------------------------------------- intervals by severity

def test_intervals_differ_by_severity():
    """A single interval for everyone is the wrong shape -- the brief says
    'for their severity level', and the top and bottom differ eightfold."""
    assert REASSESS_MINUTES[Category.ORANGE] == 15
    assert REASSESS_MINUTES[Category.BLUE] == 120
    assert REASSESS_MINUTES[Category.ORANGE] < REASSESS_MINUTES[Category.YELLOW]


def test_red_is_never_between_checks():
    """Continuous nursing care, not an interval -- there is no window in which
    nobody is watching a resuscitation patient."""
    board = board_with(Category.RED)
    assert board.patients["bed-1"].overdue(now=0.1)


@pytest.mark.parametrize("cat,before,after", [
    (Category.ORANGE, 14, 16),
    (Category.YELLOW, 29, 31),
    (Category.GREEN, 59, 61),
])
def test_overdue_fires_at_the_published_interval(cat, before, after):
    board = board_with(cat)
    assert not board.patients["bed-1"].overdue(before)
    assert board.patients["bed-1"].overdue(after)


# ---------------------------------------------------- trigger: worsening

def test_worsening_escalates_without_waiting_for_the_timer():
    """A patient who deteriorates two minutes after triage should not sit until
    their interval elapses."""
    board = board_with(Category.GREEN)
    worse = BeliefState(PROTOCOL).record(
        "cardiac_pain", Answer.TRUE, Evidence("into my jaw", "speech"))
    e = board.observe("bed-1", worse, now=2.0)
    assert e is not None
    assert e.trigger is Trigger.WORSENED
    assert e.now < e.was
    assert board.patients["bed-1"].current is Category.ORANGE


def test_improving_never_lowers_a_priority():
    """The board escalates on its own; de-escalation needs a named clinician."""
    board = board_with(Category.ORANGE, checks=["cardiac_pain"])
    better = BeliefState(PROTOCOL)          # nothing positive at all
    assert board.observe("bed-1", better, now=5.0) is None
    assert board.patients["bed-1"].current is Category.ORANGE


# ---------------------------------------------------- trigger: timer

def test_an_overdue_check_with_nothing_new_prompts_rather_than_moves():
    board = board_with(Category.GREEN)
    fired = board.tick(now=61)
    assert len(fired) == 1
    assert fired[0].trigger is Trigger.TIMER
    assert fired[0].now == fired[0].was          # prompted, not silently moved
    assert "nurse re-check required" in fired[0].detail


def test_an_overdue_check_that_rescores_worse_does_move():
    board = board_with(Category.GREEN, checks=["cardiac_pain"])
    fired = board.tick(now=61)
    assert fired[0].now is Category.ORANGE


# ---------------------------------------------------- trigger: starvation

def test_starvation_flags_but_does_not_change_the_category():
    """The failure that fills serious-incident reports: triaged correctly as low
    acuity, then forgotten.

    It raises a FLAG, not the category. An earlier version escalated a band,
    which fell apart on a full board -- every Orange passes its 3x multiple
    inside an hour, so a busy shift turned the whole queue Immediate, and a board
    where everyone is Immediate says nothing.
    """
    board = board_with(Category.GREEN)               # 60 min target
    assert not board.patients["bed-1"].starved(120)
    assert board.patients["bed-1"].starved(60 * STARVATION_MULTIPLE + 1)

    fired = board.tick(now=60 * STARVATION_MULTIPLE + 1)
    starved = [e for e in fired if e.trigger is Trigger.STARVED]
    assert len(starved) == 1
    assert starved[0].now == starved[0].was          # category untouched
    assert "find this patient now" in starved[0].detail
    assert board.patients["bed-1"].current is Category.GREEN
    assert board.patients["bed-1"].flagged_starved


def test_starvation_fires_once_not_on_every_tick():
    board = board_with(Category.GREEN)
    t = 60 * STARVATION_MULTIPLE + 1
    first = [e for e in board.tick(t) if e.trigger is Trigger.STARVED]
    again = [e for e in board.tick(t + 30) if e.trigger is Trigger.STARVED]
    assert len(first) == 1 and not again


def test_a_starved_patient_sorts_to_the_top_of_their_own_band():
    """Impossible to miss, without corrupting the acuity signal."""
    b = BeliefState(PROTOCOL)
    board = Board(PROTOCOL)
    board.admit("green-forgotten", Category.GREEN, b, PLAUSIBLE, now=0)
    board.admit("green-recent", Category.GREEN, b, PLAUSIBLE, now=200)
    board.admit("orange", Category.ORANGE, b, PLAUSIBLE, now=201)
    board.tick(now=202)

    order = [w.ref for w in board.ranked(now=202)]
    assert order[0] == "orange"                      # urgency still wins
    assert order[1] == "green-forgotten"             # flagged, so first in band
    assert [w.ref for w in board.starved(202)] == ["green-forgotten"]


def test_the_most_urgent_category_cannot_starve():
    board = board_with(Category.RED)
    assert not board.patients["bed-1"].starved(10_000)


# ---------------------------------------------------- board order

def test_waiting_time_orders_within_a_category_but_never_across_one():
    board = Board(PROTOCOL)
    b = BeliefState(PROTOCOL)
    board.admit("green-old", Category.GREEN, b, PLAUSIBLE, now=0)
    board.admit("orange-new", Category.ORANGE, b, PLAUSIBLE, now=100)
    board.admit("green-new", Category.GREEN, b, PLAUSIBLE, now=90)

    order = [w.ref for w in board.ranked(now=101)]
    assert order[0] == "orange-new"              # urgency first, always
    assert order[1:] == ["green-old", "green-new"]   # then longest waiting


def test_the_initial_category_survives_every_escalation():
    """CTAS: the initial triage score is never changed. Escalations append."""
    board = board_with(Category.GREEN)
    worse = BeliefState(PROTOCOL).record(
        "cardiac_pain", Answer.TRUE, Evidence("into my jaw", "speech"))
    board.observe("bed-1", worse, now=2.0)
    w = board.patients["bed-1"]
    assert w.initial is Category.GREEN
    assert w.current is Category.ORANGE
    assert len(w.escalations) == 1


def test_seen_removes_from_the_queue():
    board = board_with()
    assert board.seen("bed-1") is not None
    assert board.patients == {}


# ---------------------------------------------------------------- beds

def test_admission_assigns_the_lowest_free_bed():
    from triage.monitor import BED_COUNT
    board = Board(PROTOCOL)
    b = BeliefState(PROTOCOL)
    w1 = board.admit("p1", Category.GREEN, b, PLAUSIBLE, now=0.0)
    w2 = board.admit("p2", Category.GREEN, b, PLAUSIBLE, now=1.0)
    assert (w1.bed, w2.bed) == (1, 2)
    assert board.free_beds() == BED_COUNT - 2


def test_a_freed_bed_is_reused_not_skipped():
    board = Board(PROTOCOL)
    b = BeliefState(PROTOCOL)
    board.admit("p1", Category.GREEN, b, PLAUSIBLE, now=0.0)   # bed 1
    board.admit("p2", Category.GREEN, b, PLAUSIBLE, now=0.0)   # bed 2
    board.seen("p1")                                           # frees bed 1
    w3 = board.admit("p3", Category.GREEN, b, PLAUSIBLE, now=0.0)
    assert w3.bed == 1


def test_the_board_never_turns_a_patient_away_at_capacity():
    """A full board still admits -- into the queue, with no bed, rather than
    refusing the patient outright. That's the honest state of a real
    department at capacity, not a reason to stop triaging."""
    from triage.monitor import BED_COUNT
    board = Board(PROTOCOL)
    b = BeliefState(PROTOCOL)
    for i in range(BED_COUNT):
        board.admit(f"p{i}", Category.GREEN, b, PLAUSIBLE, now=0.0)
    assert board.free_beds() == 0

    overflow = board.admit("p-overflow", Category.YELLOW, b, PLAUSIBLE, now=0.0)
    assert overflow.bed is None
    assert "p-overflow" in board.patients   # queued, not rejected
