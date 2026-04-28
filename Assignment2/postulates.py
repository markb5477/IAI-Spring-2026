"""AGM revision postulate checkers (Lecture 11).

Each predicate takes ``(B, phi, revise)`` (or ``(B, phi, psi, revise)``) and
returns True iff the postulate holds.  Membership is checked at the closure
(entailment) level rather than as syntactic set membership, so the priority
annotations on belief-base entries don't break equality comparisons.
"""

from __future__ import annotations

from typing import Callable

from formula import Formula, neg, BOT
from belief_base import BeliefBase, formulas_of
from resolution import entails
from expansion import expand, equivalent


ReviseOp = Callable[[BeliefBase, Formula], BeliefBase]


def check_success(B: BeliefBase, phi: Formula, revise: ReviseOp) -> bool:
    """Success:  φ ∈ Cn(B * φ)."""
    return entails(formulas_of(revise(B, phi)), phi)


def check_inclusion(B: BeliefBase, phi: Formula, revise: ReviseOp) -> bool:
    """Inclusion:  Cn(B * φ) ⊆ Cn(B + φ)."""
    R = revise(B, phi)
    E_forms = formulas_of(expand(B, phi))
    return all(entails(E_forms, b.formula) for b in R)


def check_vacuity(B: BeliefBase, phi: Formula, revise: ReviseOp) -> bool:
    """Vacuity:  if  ¬φ ∉ Cn(B), then  B * φ  ≡  B + φ."""
    if entails(formulas_of(B), neg(phi)):
        return True                               # premise fails -> vacuously holds
    R_forms = formulas_of(revise(B, phi))
    E_forms = formulas_of(expand(B, phi))
    return (all(entails(E_forms, f) for f in R_forms) and
            all(entails(R_forms, f) for f in E_forms))


def check_consistency(B: BeliefBase, phi: Formula, revise: ReviseOp) -> bool:
    """Consistency:  if  φ  is satisfiable, then  B * φ  is satisfiable."""
    if entails([], neg(phi)):                     # phi is unsatisfiable -> premise fails
        return True
    return not entails(formulas_of(revise(B, phi)), BOT)


def check_extensionality(
    B: BeliefBase, phi: Formula, psi: Formula, revise: ReviseOp,
) -> bool:
    """Extensionality:  if  φ ≡ ψ,  then  Cn(B * φ) = Cn(B * ψ)."""
    if not equivalent(phi, psi):
        return True
    R1_forms = formulas_of(revise(B, phi))
    R2_forms = formulas_of(revise(B, psi))
    return (all(entails(R2_forms, f) for f in R1_forms) and
            all(entails(R1_forms, f) for f in R2_forms))
