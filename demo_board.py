"""Two hours in the waiting room.

    python demo_board.py

Initial triage is a snapshot. This is the part that keeps watching, and it is
where the serious-incident reports actually come from: a patient triaged
CORRECTLY as low acuity who deteriorates in a corridor at hour three while
nobody looks at them again.

Three things fire here, and only the first two are in the protocol:

    TIMER      the published re-check interval for that category elapsed
    WORSENED   new observations re-score to something more urgent
    STARVED    waited far past their own target -- our policy, not CTAS's

Only the first two change a category. Starvation raises a flag and sorts the
patient to the top of their own band, because acuity is how sick someone is and
waiting time is how badly the department is failing them -- and a board where
everyone has been escalated to Immediate says nothing at all.
"""

from triage import Answer, BeliefState, Category, Evidence, plausible_set
from triage.charts import PROTOCOL
from triage.monitor import REASSESS_MINUTES, STARVATION_MULTIPLE, Board, Trigger

RULE = "-" * 78
CLOCK_START = 14 * 60          # 14:00, in minutes past midnight


def hhmm(minutes: float) -> str:
    m = int(CLOCK_START + minutes)
    return f"{m // 60 % 24:02d}:{m % 60:02d}"


def belief_with(*checks) -> BeliefState:
    b = BeliefState(PROTOCOL)
    for cid in checks:
        b = b.record(cid, Answer.TRUE, Evidence("observed at triage", "staff"))
    return b


PLAUSIBLE = {
    "chest": plausible_set({"chest_pain": 0.8, "abdominal_pain": 0.2}),
    "abdo": plausible_set({"abdominal_pain": 0.8, "chest_pain": 0.2}),
    "limb": plausible_set({"limb_problems": 0.9}),
    "unwell": plausible_set({"unwell_adult": 0.7, "breathlessness": 0.3}),
}

board = Board(PROTOCOL)


def show(now: float) -> None:
    print(f"\n  {hhmm(now)}")
    print(f"  {'patient':<26}{'category':<14}{'waited':>8}{'re-check':>12}")
    for w in board.ranked(now):
        due = w.due_in(now)
        if REASSESS_MINUTES[w.current] == 0:
            due_text = "continuous"
        elif due < 0:
            due_text = f"OVERDUE {-due:.0f}m"
        else:
            due_text = f"in {due:.0f}m"
        mark = ""
        if w.current != w.initial:
            mark = f"  <- raised from {w.initial.label}"
        if w.flagged_starved:
            mark += "  [FIND THIS PATIENT]"
        print(f"  {w.ref:<26}{w.current.label:<14}"
              f"{w.waited(now):>6.0f}m{due_text:>12}{mark}")


def fired(events) -> None:
    for e in events:
        arrow = f"{e.was.label} -> {e.now.label}" if e.now < e.was else "prompt only"
        print(f"      [{e.trigger}]  {e.ref}: {arrow}")
        print(f"         {e.detail}")


def main() -> None:
    print(RULE)
    print("  WAITING ROOM -- one afternoon")
    print(RULE)
    print(f"  Re-check intervals, from CTAS: " +
          ", ".join(f"{c.label} {REASSESS_MINUTES[c]}m" for c in Category
                    if REASSESS_MINUTES[c]))
    print(f"  Starvation guard: {STARVATION_MULTIPLE:g}x the category target (our policy, not CTAS)")

    # ---- arrivals -------------------------------------------------------
    board.admit("bed-02 chest pain", Category.ORANGE,
                belief_with("cardiac_pain"), PLAUSIBLE["chest"], now=0)
    board.admit("bed-05 laceration", Category.GREEN,
                belief_with("minor_haemorrhage"), PLAUSIBLE["limb"], now=6)
    board.admit("bed-07 unwell adult", Category.GREEN,
                belief_with("recent_onset"), PLAUSIBLE["unwell"], now=10)
    board.admit("bed-09 ankle", Category.GREEN,
                belief_with("mild_pain"), PLAUSIBLE["limb"], now=18)
    show(20)

    # ---- 35 minutes: bed-02 is overdue on a 15 minute interval ----------
    print(f"\n{RULE}\n  {hhmm(35)}  the clock catches bed-02")
    fired(board.tick(35))
    print("      Nothing about him has changed, so the category holds and the")
    print("      nurse is prompted. The board does not move a patient silently.")

    # ---- 52 minutes: bed-07 deteriorates -------------------------------
    print(f"\n{RULE}\n  {hhmm(52)}  an assistant re-records bed-07")
    print("      Respiratory rate up, now unable to finish a sentence.")
    worse = belief_with("recent_onset", "cannot_complete_sentence")
    e = board.observe("bed-07 unwell adult", worse, now=52)
    fired([e] if e else [])
    print("      Triaged Standard 42 minutes ago and correct at the time. This")
    print("      is the patient the initial score cannot help.")
    show(52)

    # ---- 3 hours: bed-09 has been forgotten -----------------------------
    print(f"\n{RULE}\n  {hhmm(190)}  bed-09 has been waiting three hours")
    fired(board.tick(190))
    show(190)

    # ---- summary --------------------------------------------------------
    print(f"\n{RULE}")
    moved = board.escalations()
    flagged = board.starved(190)
    print(f"  {len(board.log)} re-assessment events")
    print(f"  {len(moved)} moved a category:")
    for e in moved:
        print(f"    {hhmm(e.at)}  {e}")
    print(f"  {len(flagged)} flagged as waiting far past target:")
    for w in flagged:
        print(f"    {w.ref} -- {w.waited(190):.0f} min, still {w.current.label}")
    print()
    print("  Every category change is an escalation. The board raises a priority")
    print("  on its own and never lowers one -- a de-escalation needs a named")
    print("  clinician, so every downgrade is attributable by construction and")
    print("  this module cannot produce one.")
    print(RULE)


if __name__ == "__main__":
    main()
