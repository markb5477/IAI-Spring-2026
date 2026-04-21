"""AGM revision:  B * phi     (Levi identity).

    B * phi  =  (B / !phi)  +  phi

Contract first to make room (so the result will not entail false once phi
is added), then expand with phi.
"""

from formula import neg
from entailment import entails
from contraction import contract
from expansion import expand


def revise(B, phi):
    if entails([], neg(phi)):          # phi is a contradiction (Consistency)
        return B
    B1 = contract(B, neg(phi))         # make room
    return expand(B1, phi)             # add the new info
