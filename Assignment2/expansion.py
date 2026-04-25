"""AGM expansion:  B + phi.

If adding phi would make B inconsistent, we fall back to revision
(Levi identity) instead of blindly expanding.  This keeps the belief
base consistent at all times.

Consistent expansion:   B + phi  (phi does not contradict B)
Inconsistent fallback:  B * phi  (revision via Levi identity)
"""

from belief_base import priority_of
from entailment  import entails
from formula     import neg


def expand(B, phi):
    """
    Add phi to B.

    - If B ∪ {phi} would be inconsistent, delegate to revise() instead.
    - If phi is already entailed by B, return B unchanged (no redundant add).
    - Otherwise append (phi, priority) to B.

    Parameters
    ----------
    B   : list of (formula_ast, priority) pairs
    phi : formula AST

    Returns
    -------
    list of (formula_ast, priority) pairs
    """
    # Already entailed — nothing new to add
    if entails(B, phi):
        return B

    # Check if adding phi would cause inconsistency: B ∪ {phi} ⊨ ⊥
    # i.e. the expanded base entails both phi and ¬phi
    tentative = B + [(phi, priority_of(phi))]
    if entails(tentative, neg(phi)):
        # Inconsistent — fall back to revision (imported lazily to avoid circular import)
        from revision import revise
        return revise(B, phi)

    return tentative
