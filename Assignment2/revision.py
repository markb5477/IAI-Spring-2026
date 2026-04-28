"""AGM revision:  B * phi     (Levi identity).

    B * phi  =  (B ÷ ¬phi)  +  phi

Contract first to make room (so the result will not entail false once phi is
added), then expand with phi.

When ``phi`` is unsatisfiable we return an explicitly inconsistent base
``[Belief(Bot, 0)]`` rather than the input.  This is the AGM-correct choice:
Success then holds (Bot ⊨ phi for any phi), and Consistency holds vacuously
(its premise "phi is satisfiable" fails).
"""

from __future__ import annotations

from formula import Formula, BOT, neg
from belief_base import BeliefBase, make
from resolution import entails
from contraction import contract
from expansion import expand


def revise(B: BeliefBase, phi: Formula) -> BeliefBase:
    if entails([], neg(phi)):                     # phi is unsatisfiable
        return [make(BOT)]
    return expand(contract(B, neg(phi)), phi)
