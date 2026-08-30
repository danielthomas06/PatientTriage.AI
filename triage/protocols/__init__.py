"""Protocol packs.

The engine imports nothing protocol-specific. Swap the pack and it triages
under a different standard, which is the scalability answer: one assistant
across hospitals that use different protocols, or none.

    mts_illustrative  authored from general clinical knowledge for demonstration.
                      NOT the Manchester Triage System, which is copyrighted.
    ctas              transcribed from the published CTAS manual. Real, sourced,
                      nationally endorsed. See CTAS_PROVENANCE.md for the
                      copyright position before shipping.
"""

from . import ctas

__all__ = ["ctas"]
