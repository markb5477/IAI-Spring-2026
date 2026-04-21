"""AGM contraction:  B / phi    (priority-guided partial meet).

Return a subset B' of B such that B' does not entail phi, keeping as many
high-priority formulas as possible.  Prefer a surgical single-formula cut;
if no single removal breaks entailment, drop the weakest formula
unconditionally and loop.

Ties on priority are broken uniformly at random.
"""

import random

from entailment import entails


def contract(B, phi):
    if not entails(B, phi):             # Vacuity - phi was never derivable
        return B
    if entails([], phi):                # phi is a tautology - cannot be contracted
        return B

    while entails(B, phi):
        # Candidates: each one, removed alone, would break entailment of phi.
        candidates = [pair for pair in B
                      if not entails([x for x in B if x != pair], phi)]

        pool = candidates if candidates else B       # fallback: drop weakest overall
        lowest = min(p for _, p in pool)
        pick = random.choice([c for c in pool if c[1] == lowest])

        B = [x for x in B if x != pick]

    return B
