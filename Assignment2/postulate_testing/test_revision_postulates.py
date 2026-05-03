"""Hand-checked tests for the AGM revision postulates.

Each test pairs a small belief base with a formula and asserts the postulate
checker returns the expected value (True for "postulate holds", False for a
documented failure).  Numbering follows the standard AGM presentation
(K*1 .. K*6 from Gärdenfors / Hansson).
"""

from __future__ import annotations

import os
import sys
import unittest

# Make the Assignment2 package modules importable when this file is run
# directly or via unittest discovery.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formula import parse
from belief_base import make
from revision import revise
from postulates import (
    check_success,
    check_inclusion,
    check_vacuity,
    check_consistency,
    check_extensionality,
)


def B_from(*formula_strs: str):
    """Build a BeliefBase from a list of concrete-syntax formula strings."""
    return [make(parse(s)) for s in formula_strs]


class TestSuccess(unittest.TestCase):
    """K*2 (Success):  phi  in  Cn(B * phi)."""

    def test_no_conflict(self):
        # B = {p}, revise by q.  q is unrelated -> revise just expands.
        self.assertTrue(check_success(B_from("p"), parse("q"), revise))

    def test_conflict(self):
        # B = {p}, revise by !p.  Success forces !p into the result.
        self.assertTrue(check_success(B_from("p"), parse("!p"), revise))

    def test_empty_base(self):
        self.assertTrue(check_success(B_from(), parse("p"), revise))

    def test_unsat_phi(self):
        # phi unsatisfiable -> revise() returns [Bot]; Bot entails everything.
        self.assertTrue(check_success(B_from("p"), parse("q & !q"), revise))


class TestInclusion(unittest.TestCase):
    """K*3 (Inclusion):  Cn(B * phi)  subset of  Cn(B + phi)."""

    def test_no_conflict(self):
        # No conflict: B*phi should equal B+phi, so trivially included.
        self.assertTrue(check_inclusion(B_from("p"), parse("q"), revise))

    def test_conflict(self):
        # B+!p is inconsistent -> Cn(B+!p) is everything; included is trivial.
        self.assertTrue(check_inclusion(B_from("p"), parse("!p"), revise))

    def test_with_implication(self):
        self.assertTrue(
            check_inclusion(B_from("p", "p -> q"), parse("!q"), revise)
        )


class TestVacuity(unittest.TestCase):
    """K*4 (Vacuity):  if !phi not in Cn(B), then  B * phi  ===  B + phi."""

    def test_premise_holds(self):
        # B = {p}, phi = q.  !q not entailed -> revise == expand.
        self.assertTrue(check_vacuity(B_from("p"), parse("q"), revise))

    def test_premise_fails(self):
        # B = {p}, phi = !p.  !!p in Cn(B) -> premise fails -> vacuously holds.
        self.assertTrue(check_vacuity(B_from("p"), parse("!p"), revise))

    def test_independent_extension(self):
        self.assertTrue(check_vacuity(B_from("p", "q"), parse("r"), revise))


class TestConsistency(unittest.TestCase):
    """K*5 (Consistency):  if phi is satisfiable, then  B * phi  is consistent."""

    def test_satisfiable_conflict(self):
        # Even when phi conflicts with B, the result must stay consistent.
        self.assertTrue(check_consistency(B_from("p"), parse("!p"), revise))

    def test_unsat_phi_vacuous(self):
        # phi unsatisfiable -> premise fails -> vacuously holds.
        self.assertTrue(check_consistency(B_from("p"), parse("p & !p"), revise))

    def test_independent_phi(self):
        self.assertTrue(check_consistency(B_from("p"), parse("q"), revise))


class TestExtensionality(unittest.TestCase):
    """K*6 (Extensionality):  if phi === psi, then Cn(B * phi) = Cn(B * psi)."""

    def test_double_negation(self):
        # p === !!p
        self.assertTrue(
            check_extensionality(B_from("p", "q"), parse("p"), parse("!!p"), revise)
        )

    def test_de_morgan(self):
        # !(p & q) === (!p | !q)
        self.assertTrue(
            check_extensionality(
                B_from("r"), parse("!(p & q)"), parse("!p | !q"), revise
            )
        )

    def test_implication_disjunction(self):
        # (p -> q) === (!p | q)
        self.assertTrue(
            check_extensionality(
                B_from("p", "r"), parse("p -> q"), parse("!p | q"), revise
            )
        )

    def test_not_equivalent_vacuous(self):
        # p and q are not equivalent -> premise fails -> vacuously holds.
        self.assertTrue(
            check_extensionality(B_from("r"), parse("p"), parse("q"), revise)
        )


class TestRevisionEdgeCases(unittest.TestCase):
    """Boundary inputs: empty base, Top/Bot phi, iterated revision, equivalence
    in disguise, long implication chains.  All postulates should hold for every
    case below (or hold vacuously when the premise fails)."""

    _ALL = (check_success, check_inclusion, check_vacuity, check_consistency)

    def _assert_all_postulates(self, B, phi):
        for chk in self._ALL:
            self.assertTrue(chk(B, phi, revise),
                            f"{chk.__name__} failed on B={B}, phi={phi}")

    def test_empty_base_with_atom(self):
        # Revising the empty base by an atom: everything should hold.
        self._assert_all_postulates(B_from(), parse("p"))

    def test_empty_base_with_tautology(self):
        # Revising empty base by Top.  Top is satisfiable, so Consistency
        # is non-vacuous and must produce a consistent result.
        self._assert_all_postulates(B_from(), parse("Top"))

    def test_revise_by_bot(self):
        # phi = Bot is unsatisfiable -> revise() returns [Bot] explicitly.
        # Success holds (Bot |= anything); Consistency holds vacuously.
        B = B_from("p")
        self.assertTrue(check_success(B, parse("Bot"), revise))
        self.assertTrue(check_consistency(B, parse("Bot"), revise))
        self.assertTrue(check_inclusion(B, parse("Bot"), revise))

    def test_revise_by_top(self):
        # phi = Top is a tautology and trivially consistent with anything.
        self._assert_all_postulates(B_from("p", "q"), parse("Top"))

    def test_revise_by_already_entailed(self):
        # Revising by something already in B: should be a no-op at the
        # closure level.
        self._assert_all_postulates(B_from("p", "q"), parse("p"))

    def test_iterated_revision_same_phi(self):
        # (B * p) * p still satisfies Success for p (idempotence at the
        # postulate level, not necessarily syntactic).
        B = revise(B_from("q"), parse("p"))
        self._assert_all_postulates(B, parse("p"))

    def test_revise_then_revert(self):
        # (B * p) * !p — the second revision must still drive !p in.
        B = revise(B_from("q"), parse("p"))
        self._assert_all_postulates(B, parse("!p"))

    def test_long_implication_chain(self):
        # B = {p, p->q, q->r}.  Revising by !r forces contraction along the
        # chain; partial meet has to drop one of the implications.
        B = B_from("p", "p -> q", "q -> r")
        self._assert_all_postulates(B, parse("!r"))

    def test_extensionality_triple_negation(self):
        # !!!p === !p
        self.assertTrue(check_extensionality(
            B_from("q"), parse("!!!p"), parse("!p"), revise
        ))

    def test_extensionality_top_disjunction(self):
        # Top === (p | !p) — both tautologies.
        self.assertTrue(check_extensionality(
            B_from("q"), parse("Top"), parse("p | !p"), revise
        ))

    def test_extensionality_bot_conjunction(self):
        # Bot === (p & !p) — both unsatisfiable.  Both revisions yield [Bot].
        self.assertTrue(check_extensionality(
            B_from("q"), parse("Bot"), parse("p & !p"), revise
        ))

    def test_redundant_beliefs_in_base(self):
        # B contains p and !!p (logically equivalent, syntactically distinct).
        # Postulates must not be confused by syntactic duplicates.
        self._assert_all_postulates(B_from("p", "!!p"), parse("q"))

    def test_biconditional_phi(self):
        # phi is a complex biconditional rather than a literal.
        self._assert_all_postulates(B_from("p"), parse("p <-> q"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
