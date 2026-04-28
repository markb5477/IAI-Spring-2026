"""AGM contraction:  B ÷ phi    (priority-based partial meet, Lecture 11).

    B ÷ φ  =  ⋂ γ(B ⊥ φ)

where  B ⊥ φ  is the set of inclusion-maximal subsets of B that do not entail
phi (the "remainders"), and γ selects the best ones.  We instantiate γ with
epistemic entrenchment proxied by the priority sum of each remainder.

Naive remainder enumeration is O(2^|B|) - acceptable for assignment-sized bases.
"""

from __future__ import annotations

from itertools import combinations

from formula import Formula
from belief_base import Belief, BeliefBase, formulas_of
from resolution import entails


def remainders(B: BeliefBase, phi: Formula) -> list[BeliefBase]:
    """All inclusion-maximal sublists of B that do NOT entail phi.

    Walks the powerset largest-first and prunes any subset already covered by
    a maximal superset already accepted.
    """
    n = len(B)
    R: list[BeliefBase] = []
    R_sets: list[frozenset[int]] = []
    for size in range(n, -1, -1):
        for combo in combinations(range(n), size):
            S = frozenset(combo)
            if any(S < T for T in R_sets):        # covered by a maximal superset
                continue
            sub: BeliefBase = [B[i] for i in combo]
            if not entails(formulas_of(sub), phi):
                R.append(sub)
                R_sets.append(S)
    return R


def _score(remainder: BeliefBase) -> int:
    """Sum of priorities (higher = remainder retains more entrenched beliefs)."""
    return sum(b.priority for b in remainder)


def _gamma(R: list[BeliefBase]) -> list[BeliefBase]:
    """Pick the remainders that maximise the priority-sum score."""
    if not R:
        return []
    best = max(_score(c) for c in R)
    return [c for c in R if _score(c) == best]


def _intersection(rs: list[BeliefBase]) -> BeliefBase:
    """Intersect remainders by Belief identity, preserving the order of the first."""
    if not rs:
        return []
    common: set[Belief] = set(rs[0])
    for r in rs[1:]:
        common &= set(r)
    seen: set[Belief] = set()
    out: BeliefBase = []
    for b in rs[0]:
        if b in common and b not in seen:
            out.append(b)
            seen.add(b)
    return out


def contract(B: BeliefBase, phi: Formula) -> BeliefBase:
    if entails([], phi):                          # tautology - Success exception
        return B
    if not entails(formulas_of(B), phi):          # Vacuity
        return B
    R = remainders(B, phi)
    Rsel = _gamma(R)
    return _intersection(Rsel)
