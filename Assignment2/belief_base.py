"""Belief base: list of (formula, priority) pairs.

Higher priority means more entrenched - less likely to be dropped in contraction.
Priority is derived from the formula's *syntactic rank*: simpler formulas
(literals, atoms) are more entrenched than complex ones.
"""

from ranking import rank


def priority_of(phi):
    """Priority for a formula.  Higher = more entrenched.

    Using -rank(phi) so argmin-by-priority in contraction drops the most
    complex formula first, matching the chosen "simpler = more entrenched" policy.
    """
    return -rank(phi)
