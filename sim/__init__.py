"""Discrete-event ED simulator and evaluation harness.

Exists because you cannot evaluate a triage system on real patients in a
weekend, and because a demo is not evidence. It converts "here is a thing that
works" into "across N simulated arrivals, under-triage fell from X to Y, and
here is the assumption that result depends on."
"""

from .hospital import Department, Record, run
from .metrics import Metrics, compare, summarise
from .patients import Cohort, Patient, generate
from .policies import Assistant, Baseline, Outcome

__all__ = [
    "Cohort", "Patient", "generate",
    "Baseline", "Assistant", "Outcome",
    "Department", "Record", "run",
    "Metrics", "summarise", "compare",
]
