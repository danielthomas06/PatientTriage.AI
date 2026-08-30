"""First-line actions: a lookup table, not a diagnosis.

The one property worth protecting here is the boundary: this table is keyed
on (branch, category) -- both already produced by the deterministic engine --
and nothing here lets a model infer what's wrong with the patient.
"""

from triage.actions import BRANCH_ACTIONS, first_line_actions
from triage.core import Category


def test_actions_scale_with_urgency_not_against_it():
    """A RED patient must see at least everything a YELLOW patient sees.

    If a more urgent category produced FEWER actions, that would be a table
    bug with real consequences -- the whole point of the threshold scheme is
    that urgency only adds orders, never removes them.
    """
    for branch in BRANCH_ACTIONS:
        red = set(first_line_actions(branch, Category.RED))
        yellow = set(first_line_actions(branch, Category.YELLOW))
        blue = set(first_line_actions(branch, Category.BLUE))
        assert blue <= yellow <= red, branch


def test_unmapped_branch_returns_nothing():
    """Silence, not a guess -- same discipline as an unresolved cohort."""
    assert first_line_actions("palpitations", Category.RED) == []
    assert first_line_actions("not_a_real_branch", Category.RED) == []


def test_every_action_defers_to_local_policy_or_is_a_plain_observation():
    """Nothing in the table is phrased as a standing instruction on its own.

    Anything that names a medicine or a step outside plain nursing skill must
    say "per local PGD"/"per local protocol" or explicitly flag the clinician
    -- the table cannot be the sole authority for a prescribing decision. Plain
    nursing skills (IV cannulation, observations, immobilising a limb) need no
    such gate; they are already within a nurse's ordinary scope.
    """
    risky_markers = ("mg", "aspirin", "ct pathway", "x-ray")
    safe_markers = ("per local pgd", "per local protocol", "flag", "clinician")
    for branch, entry in BRANCH_ACTIONS.items():
        items = list(entry.get("always", ()))
        for _, more in entry.get("thresholds", ()):
            items.extend(more)
        for text in items:
            low = text.lower()
            if any(m in low for m in risky_markers):
                assert any(m in low for m in safe_markers), (branch, text)


def test_serve_surfaces_actions_for_a_headache_case():
    """End-to-end: a narrative that fires the headache branch should carry a
    first-line-actions block, gated on the leading branch actually having one."""
    import serve
    serve.USE_MODEL = False
    e = serve.Encounter("t", age=54)
    e.add_narrative("Worst headache of my life, came on suddenly an hour ago.")
    actions = e.state()["actions"]
    assert actions is not None
    assert actions["branch"] == "Headache"
    assert len(actions["items"]) >= 1


def test_serve_shows_nothing_for_an_empty_encounter():
    """No narrative, no branch weights -- the panel must stay hidden rather
    than guess at what to suggest."""
    import serve
    e = serve.Encounter("t", age=30)
    assert e.state()["actions"] is None
