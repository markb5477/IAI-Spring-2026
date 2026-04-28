"""AGM expansion:  B + phi.

Add phi to B with a rank-derived priority.  No consistency check; that is
revision's job.  Skip if phi is already syntactically present or logically
equivalent to an existing belief (Extensionality).
"""

from __future__ import annotations

from formula import Formula
from belief_base import BeliefBase, make
from resolution import entails


def equivalent(phi: Formula, psi: Formula) -> bool:
    """Logical equivalence:  φ ⊨ ψ  and  ψ ⊨ φ."""
    return entails([phi], psi) and entails([psi], phi)


def expand(B: BeliefBase, phi: Formula) -> BeliefBase:
    for b in B:
        if b.formula == phi or equivalent(b.formula, phi):
            return B
    return B + [make(phi)]
