"""Choosing the next question: expected-risk-reduction, and relevance.

A real run surfaced questions that looked wrong for the chief complaint -- a
headache narrative producing a chest-pain-only question. The cause: every
branch starts "plausible" at a floor specifically so a weak extraction never
wrongly excludes one (documented in extract.py), but plausible_set() then
treats membership as binary -- a branch at the bare floor competed for the
next question on exactly equal footing with the one the patient actually
described. `rank(..., branch_weights=...)` fixes the ordering without
touching the safety property: nothing is excluded, a floor-level branch's
questions are simply discounted relative to the current top weight.
"""

from triage import BeliefState, plausible_set
from triage.charts import PROTOCOL
from triage.core import Source
from triage.voi import rank, should_stop

FLOOR = 0.06


def ask_only(candidates):
    return [c for c in candidates
            if PROTOCOL.discriminators[c.discriminator_id].source is Source.ASK]


def weights_favoring(branch: str, w: float = 0.6) -> dict[str, float]:
    out = {b: FLOOR for b in PROTOCOL.branches}
    out[branch] = w
    return out


def test_a_branch_specific_check_is_discounted_by_its_relative_weight():
    """cardiac_pain lives only on chest_pain/palpitations. With headache
    weighted 10x higher than the floor, cardiac_pain's ranking value should
    drop by the CUBE of that ratio -- not to zero, just decisively out of the
    way. Cubed rather than linear because a linear 10x wasn't enough in a
    real case: a genuinely plausible branch's own low-value checks could
    still lose to an unrelated branch's high-value one. See the comment in
    voi.py's rank() for the live example that motivated this."""
    belief = BeliefState(PROTOCOL)
    weights = weights_favoring("headache")
    plausible = plausible_set(weights)

    unweighted = {c.discriminator_id: c.value for c in ask_only(rank(belief, plausible))}
    weighted = {c.discriminator_id: c.value
                for c in ask_only(rank(belief, plausible, branch_weights=weights))}

    assert weighted["cardiac_pain"] == unweighted["cardiac_pain"] * (FLOOR / 0.6) ** 3


def test_general_checks_are_never_discounted():
    """Checks on every branch (recent_onset, the pain ladder) must keep their
    full value regardless of which branch is weighted -- they matter no
    matter what the chief complaint turns out to be, which is the same
    reasoning that puts life-threat checks first."""
    belief = BeliefState(PROTOCOL)
    weights = weights_favoring("headache")
    plausible = plausible_set(weights)

    unweighted = {c.discriminator_id: c.value for c in ask_only(rank(belief, plausible))}
    weighted = {c.discriminator_id: c.value
                for c in ask_only(rank(belief, plausible, branch_weights=weights))}

    for general in ("recent_onset", "severe_pain", "moderate_pain", "mild_pain"):
        assert weighted[general] == unweighted[general]


def test_the_weighted_branchs_own_checks_are_never_discounted():
    """headache is the top-weighted branch here, so its own specific checks
    must come through at full value -- only OTHER branches get discounted."""
    belief = BeliefState(PROTOCOL)
    weights = weights_favoring("headache")
    plausible = plausible_set(weights)

    unweighted = {c.discriminator_id: c.value for c in ask_only(rank(belief, plausible))}
    weighted = {c.discriminator_id: c.value
                for c in ask_only(rank(belief, plausible, branch_weights=weights))}

    for headache_specific in ("thunderclap_headache", "neck_stiffness", "new_neuro_deficit"):
        assert weighted[headache_specific] == unweighted[headache_specific]


def test_nothing_is_ever_fully_excluded():
    """The whole point of the floor is that a branch is deprioritised, never
    dropped -- a floor-level branch's check must still have a nonzero value
    when it is the only thing left with any raw gain."""
    belief = BeliefState(PROTOCOL)
    weights = weights_favoring("headache")
    plausible = plausible_set(weights)

    cardiac = next(c for c in ask_only(rank(belief, plausible, branch_weights=weights))
                    if c.discriminator_id == "cardiac_pain")
    assert cardiac.value > 0


def test_raw_gain_is_untouched_so_should_stop_stays_exact():
    """should_stop reads raw_gain, not value, to decide whether anything can
    still move the category. Relevance weighting must not leak into that --
    a check being on a deprioritised branch is not the same as it being
    unable to change the decision."""
    belief = BeliefState(PROTOCOL)
    weights = weights_favoring("headache")
    plausible = plausible_set(weights)

    unweighted_gain = {c.discriminator_id: c.raw_gain for c in rank(belief, plausible)}
    weighted_gain = {c.discriminator_id: c.raw_gain
                      for c in rank(belief, plausible, branch_weights=weights)}
    assert unweighted_gain == weighted_gain

    # should_stop's own call site never passes branch_weights -- confirm that
    # is still a valid, working call after the signature change.
    assert should_stop(belief, plausible, asked=0) is None


def test_no_weights_passed_behaves_exactly_as_before():
    """Every existing caller (should_stop, demo.py, demo_live.py) calls rank()
    without branch_weights. That path must be untouched."""
    belief = BeliefState(PROTOCOL)
    plausible = plausible_set(weights_favoring("headache"))
    assert rank(belief, plausible) == rank(belief, plausible, branch_weights=None)


def test_ranking_is_deterministic_across_hash_seeds():
    """Found as a real, reproducible flake while adding branches: the same
    encounter produced a different 4th question depending on the interpreter
    process's (randomised) string hash seed. Root cause was in
    effective_categories() iterating a frozenset directly -- content was
    order-independent, but insertion order wasn't, and that order reached a
    floating-point product downstream, close enough to flip a near-tie.
    Runs the actual subprocess boundary that exposed it rather than mocking
    hash behaviour, so a regression here would be a real repro, not a stub.
    """
    import subprocess
    import sys

    script = (
        "import serve\n"
        "serve.USE_MODEL = False\n"
        "e = serve.Encounter('t', age=54)\n"
        "e.add_narrative(\"I've had a headache since this morning and I feel a bit off.\")\n"
        "print(e.next_step()['question'])\n"
    )
    seen = set()
    for seed in ("1", "2", "3", "12345"):
        out = subprocess.run(
            [sys.executable, "-c", script],
            cwd=__file__.rsplit("tests", 1)[0],
            env={"PYTHONHASHSEED": seed, "PATH": __import__("os").environ.get("PATH", "")},
            capture_output=True, text=True, timeout=30,
        )
        assert out.returncode == 0, out.stderr
        seen.add(out.stdout.strip())
    assert len(seen) == 1, f"question varied by hash seed: {seen}"
