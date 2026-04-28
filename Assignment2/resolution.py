"""Resolution-based entailment.

Lecture 10:  premises ⊨ phi  iff  premises ∧ ¬phi  is unsatisfiable.  Convert
to CNF and saturate under binary resolution; the empty clause witnesses
unsatisfiability.

This module is the only place where actual propositional reasoning happens;
expansion / contraction / revision are layered on top of ``entails``.

The signature is deliberately narrow: ``entails`` takes an iterable of bare
``Formula`` values - never a belief base.  Belief-base callers must project
explicitly via ``formulas_of``.
"""

from __future__ import annotations

from typing import Iterable

from formula import Formula, neg
from cnf import to_cnf, is_tautology, negate_literal, Clause, CNF


def resolve(c1: Clause, c2: Clause) -> set[Clause]:
    """All binary resolvents of c1 and c2.

    For each literal ℓ ∈ c1 with ¬ℓ ∈ c2, produce ``(c1 ∪ c2) \\ {ℓ, ¬ℓ}``.
    Tautological resolvents are skipped.
    """
    out: set[Clause] = set()
    for lit in c1:
        nlit = negate_literal(lit)
        if nlit in c2:
            r: Clause = frozenset((c1 | c2) - {lit, nlit})
            if not is_tautology(r):
                out.add(r)
    return out


def entails(premises: Iterable[Formula], phi: Formula) -> bool:
    """Resolution refutation.  True iff ``premises ⊨ phi``.  Read-only."""
    S: CNF = set()
    # Collect all premise clauses
    for f in premises:
        S |= to_cnf(f)
    # Add ¬phi for refutation: premises ⊨ phi iff premises ∧ ¬phi is unsatisfiable
    S |= to_cnf(neg(phi))

    if frozenset() in S:                          # already contradictory
        return True

    while True:
        new: set[Clause] = set()
        clauses = list(S)
        n = len(clauses)
        for i in range(n):
            for j in range(i + 1, n):
                for r in resolve(clauses[i], clauses[j]):
                    if not r:
                        return True               # empty clause -> ⊨
                    new.add(r)
        if new <= S:                              # saturated, no progress
            return False
        S = S | new
