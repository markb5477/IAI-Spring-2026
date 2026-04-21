"""AGM expansion:  B + phi.

Dumb operation: add phi to B with a rank-derived priority. No consistency check,
no entailment query.  Can produce an inconsistent base - that is intentional;
handling inconsistency is revision's job.
"""

from belief_base import priority_of


def expand(B, phi):
    return B + [(phi, priority_of(phi))]
