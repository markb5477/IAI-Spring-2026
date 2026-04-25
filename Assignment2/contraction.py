"""AGM contraction:  B / phi    (priority-guided partial meet).

Return a subset B' of B such that B' does not entail phi, keeping as many
high-priority formulas as possible.  Prefer a surgical single-formula cut;
if no single removal breaks entailment, drop the weakest formula
unconditionally and loop.

Ties on priority are broken deterministically by insertion order:
among equal-priority beliefs, the most recently added one is dropped first.
This ensures contraction is deterministic, which is required for
Extensionality and Inclusion to hold reliably.
"""

from entailment import entails


def contract(B, phi):
    if not entails(B, phi):         # Vacuity — phi was never derivable
        return B
    if entails([], phi):            # phi is a tautology — cannot be contracted
        return B

    while entails(B, phi):
        # Candidates: each one, removed alone, would break entailment of phi.
        candidates = [pair for pair in B
                      if not entails([x for x in B if x != pair], phi)]

        pool = candidates if candidates else B   # fallback: drop weakest overall

        # Find the lowest priority (most complex) in the pool
        lowest = min(p for _, p in pool)
        weakest = [c for c in pool if c[1] == lowest]

        # Deterministic tiebreak: drop the one that appears LATEST in B
        # (highest index in B = most recently added = least stable)
        pick = max(weakest, key=lambda pair: B.index(pair))

        B = [x for x in B if x != pick]

    return B
