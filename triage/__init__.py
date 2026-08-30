"""PatientTriage.ai -- deterministic triage core.

Layering, strictly:

    models  ->  BeliefState  ->  engine  ->  Category

Everything above BeliefState may be probabilistic and may be wrong. Nothing
below it is. No model has a path to a Category, which is the whole safety
argument and the reason the rest of the system can afford to be ambitious.

Protocols are data, not code:

    charts.PROTOCOL          illustrative, authored for demonstration
    protocols.ctas.PROTOCOL  transcribed from the published CTAS manual

The engine imports neither. Swap the pack and it triages under a different
standard -- which is how one assistant covers hospitals that use different
protocols, or none at all.

Extraction degrades in tiers, and every tier reads the same vocabulary and the
same evidence check:

    extract.seed        hosted model
    ollama              a model on this machine -- no key, no network
    extract.keyword_seed  no model at all
"""

import os as _os

from . import audit, monitor, ollama, protocols
from .protocols import ctas as _ctas
from .belief import BeliefState, Evidence
from .cohort import Assessment, Cohort, Patient, resolve
from .charts import PROTOCOL as _ILLUSTRATIVE
from .charts import walk_in_baseline

# Which pack the package-level PROTOCOL refers to. The engine never imports a
# protocol itself; this exists so the simulator and demo can be pointed at a
# different standard without threading a parameter through every call site.
#
#   TRIAGE_PROTOCOL=ctas python run_sim.py --all
#
# Default stays illustrative: the published CTAS tables are CAEP copyright and
# the priors in that pack have not been calibrated against a real case mix, so
# the headline evaluation numbers should not silently change protocol.
_PACKS = {"illustrative": _ILLUSTRATIVE, "ctas": _ctas.PROTOCOL}
PROTOCOL = _PACKS[_os.environ.get("TRIAGE_PROTOCOL", "illustrative").strip().lower()]
from .core import Answer, Branch, Category, Confidence, Discriminator, Protocol, Source
from .engine import Decision, acuity_distribution, decide, effective_categories, plausible_set
from .extract import (
    Extraction, Unavailable, choose_next, keyword_seed, keyword_parse, parse, render, seed,
)
from .news2 import NotApplicable, Score, Vitals, score
from .tripwire import Flag, ceiling, scan
from .voi import Candidate, bayes_risk, loss, rank, should_stop, value_of

__all__ = [
    "protocols", "ollama", "audit", "monitor",
    "Assessment", "Cohort", "Patient", "resolve",
    "PROTOCOL", "walk_in_baseline",
    "Answer", "Branch", "Category", "Confidence", "Discriminator", "Protocol", "Source",
    "BeliefState", "Evidence",
    "Decision", "acuity_distribution", "decide", "effective_categories", "plausible_set",
    "NotApplicable", "Score", "Vitals", "score",
    "Candidate", "bayes_risk", "loss", "rank", "should_stop", "value_of",
    "Extraction", "Unavailable", "choose_next", "keyword_seed", "parse", "render", "seed",
    "Flag", "ceiling", "scan",
]
