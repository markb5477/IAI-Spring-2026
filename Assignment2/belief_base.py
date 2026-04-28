"""Belief base: ordered list of (formula, priority) pairs.

Higher priority means more entrenched - less likely to be dropped during
contraction.  We derive priority from the formula's *syntactic rank* (simpler
formulas are more entrenched), negated so that argmax-by-priority drops the
most complex formula first.
"""

from __future__ import annotations

from dataclasses import dataclass

from formula import Formula
from ranking import rank


@dataclass(frozen=True, slots=True)
class Belief:
    """A single belief: a formula together with its entrenchment priority."""
    formula: Formula
    priority: int


# Type alias used in signatures.  We deliberately keep the structural list type
# rather than introducing a wrapper class - operations on belief bases are
# immutable list transformations, and a class would obscure that.
BeliefBase = list[Belief]


def priority_of(phi: Formula) -> int:
    """Higher = more entrenched.  -rank(phi) so simpler formulas win ties."""
    return -rank(phi)


def make(phi: Formula) -> Belief:
    """Construct a Belief with the default rank-derived priority."""
    return Belief(phi, priority_of(phi))


def formulas_of(B: BeliefBase) -> list[Formula]:
    """Project a belief base to its bare formulas (for the entailment layer)."""
    return [b.formula for b in B]
