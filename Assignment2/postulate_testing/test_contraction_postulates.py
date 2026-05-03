"""Hand-checked tests for the AGM contraction postulates.

Numbering follows the standard AGM presentation (K-2 .. K-6).  Recovery
(K-5) is *not* satisfied by belief-base partial-meet contraction in general;
the failing case is asserted with assertFalse so a future change in the
construction is caught.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formula import parse
from belief_base import make
from contraction import contract
from postulates import (
    check_contract_inclusion,
    check_contract_vacuity,
    check_contract_success,
    check_contract_recovery,
    check_contract_extensionality,
)


def B_from(*formula_strs: str):
    return [make(parse(s)) for s in formula_strs]


class TestContractInclusion(unittest.TestCase):
    """K-2 (Inclusion):  Cn(B / phi)  subset of  Cn(B)."""

    def test_basic(self):
        # B = {p, q}, contract by p -> result is {q}, included in Cn(B).
        self.assertTrue(check_contract_inclusion(B_from("p", "q"), parse("p"), contract))

    def test_no_op(self):
        # phi not entailed -> Vacuity short-circuit -> result == B.
        self.assertTrue(check_contract_inclusion(B_from("p"), parse("q"), contract))

    def test_empty_base(self):
        self.assertTrue(check_contract_inclusion(B_from(), parse("p"), contract))

    def test_with_implication(self):
        self.assertTrue(
            check_contract_inclusion(B_from("p", "p -> q"), parse("q"), contract)
        )


class TestContractVacuity(unittest.TestCase):
    """K-3 (Vacuity):  if phi not in Cn(B), then  B / phi  ===  B."""

    def test_premise_holds(self):
        # q not entailed by {p} -> contraction is a no-op.
        self.assertTrue(check_contract_vacuity(B_from("p"), parse("q"), contract))

    def test_premise_fails(self):
        # p in Cn({p}) -> premise fails -> vacuously holds.
        self.assertTrue(check_contract_vacuity(B_from("p"), parse("p"), contract))

    def test_independent_atoms(self):
        self.assertTrue(check_contract_vacuity(B_from("p", "q"), parse("r"), contract))


class TestContractSuccess(unittest.TestCase):
    """K-4 (Success):  if phi is not a tautology, then phi not in Cn(B / phi)."""

    def test_basic(self):
        # B = {p}, contract by p -> result must not entail p.
        self.assertTrue(check_contract_success(B_from("p"), parse("p"), contract))

    def test_with_other_beliefs(self):
        self.assertTrue(check_contract_success(B_from("p", "q"), parse("p"), contract))

    def test_via_implication(self):
        # B = {p, p -> q}: q is entailed.  After contract by q, q must not
        # be entailed (one of {p, p->q} must be dropped).
        self.assertTrue(
            check_contract_success(B_from("p", "p -> q"), parse("q"), contract)
        )

    def test_tautology_vacuous(self):
        # phi tautological -> premise fails -> vacuously holds.
        self.assertTrue(
            check_contract_success(B_from("p"), parse("q | !q"), contract)
        )


class TestContractRecovery(unittest.TestCase):
    """K-5 (Recovery):  B  subset of  Cn((B / phi) + phi).

    Recovery is an AGM postulate for *closed* belief sets.  Belief-base
    partial-meet contraction does not in general satisfy Recovery: when a
    formula's content is split across multiple base formulas, contracting by
    a logical consequence may discard atoms that re-adding the consequence
    cannot recover.
    """

    def test_holds_when_vacuous(self):
        # phi not entailed -> contract is a no-op -> recovery trivial.
        self.assertTrue(check_contract_recovery(B_from("p"), parse("q"), contract))

    def test_holds_with_implication(self):
        # B = {p, p -> q}, contract by q.  Partial meet drops (p -> q)
        # because p is more entrenched (atomic).  Re-adding q recovers
        # (p -> q) at the closure level (q alone entails p -> q).
        self.assertTrue(
            check_contract_recovery(B_from("p", "p -> q"), parse("q"), contract)
        )

    def test_documented_failure_disjunction(self):
        # B = {p, q}, contract by (p | q).  Both p and q individually entail
        # (p | q), so the only remainder is the empty base.  Re-adding (p|q)
        # cannot recover the specific atoms p and q -> Recovery FAILS.
        # This is the textbook counterexample for base contraction; we assert
        # False so a future change of construction (e.g. moving to belief-set
        # contraction) flags this as a behavioural change.
        self.assertFalse(
            check_contract_recovery(B_from("p", "q"), parse("p | q"), contract)
        )


class TestContractExtensionality(unittest.TestCase):
    """K-6 (Extensionality):  if phi === psi, then Cn(B / phi) = Cn(B / psi)."""

    def test_double_negation(self):
        self.assertTrue(
            check_contract_extensionality(
                B_from("p", "q"), parse("p"), parse("!!p"), contract
            )
        )

    def test_de_morgan(self):
        # (p & q) === !(!p | !q)
        self.assertTrue(
            check_contract_extensionality(
                B_from("p", "q", "r"),
                parse("p & q"),
                parse("!(!p | !q)"),
                contract,
            )
        )

    def test_not_equivalent_vacuous(self):
        # p and q are not equivalent -> premise fails -> vacuously holds.
        self.assertTrue(
            check_contract_extensionality(
                B_from("r"), parse("p"), parse("q"), contract
            )
        )


class TestContractionEdgeCases(unittest.TestCase):
    """Boundary inputs: empty base, Top/Bot phi, iterated contraction,
    redundant beliefs, and a second Recovery counterexample arising from
    lost implication structure rather than disjunction."""

    _NORMAL = (
        check_contract_inclusion,
        check_contract_vacuity,
        check_contract_success,
    )

    def _assert_normal_postulates(self, B, phi):
        for chk in self._NORMAL:
            self.assertTrue(chk(B, phi, contract),
                            f"{chk.__name__} failed on B={B}, phi={phi}")

    def test_empty_base_any_phi(self):
        # Cn([]) entails only tautologies, so contracting empty base is
        # always a no-op (Vacuity short-circuits).
        self._assert_normal_postulates(B_from(), parse("p"))
        self.assertTrue(check_contract_recovery(B_from(), parse("p"), contract))

    def test_contract_by_top(self):
        # phi = Top is a tautology -> Success premise fails, contract returns
        # B unchanged (tautology branch).  Recovery trivially holds because
        # (B / Top) + Top contains B.
        B = B_from("p", "q")
        self._assert_normal_postulates(B, parse("Top"))
        self.assertTrue(check_contract_recovery(B, parse("Top"), contract))

    def test_contract_by_bot_consistent_base(self):
        # phi = Bot is unsatisfiable (not a tautology).  A consistent B does
        # not entail Bot, so Vacuity fires and the result is B unchanged.
        # Success premise: Bot is not a tautology -> we need Bot not to be
        # entailed by the result, which trivially holds for consistent B.
        B = B_from("p", "q")
        self._assert_normal_postulates(B, parse("Bot"))

    def test_iterated_contraction(self):
        # (B / p) / p — the second contraction is a no-op (p already gone),
        # but Success must still hold against p.
        B = contract(B_from("p", "q"), parse("p"))
        self._assert_normal_postulates(B, parse("p"))

    def test_long_implication_chain(self):
        # B = {p, p->q, q->r}.  Contract by r: must break the chain.
        B = B_from("p", "p -> q", "q -> r")
        self._assert_normal_postulates(B, parse("r"))

    def test_redundant_beliefs(self):
        # B has both p and !!p (equivalent, syntactically distinct).
        # Contracting by p must remove BOTH for Success to hold.
        B = B_from("p", "!!p", "q")
        self._assert_normal_postulates(B, parse("p"))

    def test_recovery_fails_on_lost_implication(self):
        # Second Recovery counterexample, distinct from the disjunction case.
        # B = {p->q, q}.  Both formulas individually entail (p->q):
        #   {p->q}     -- trivially
        #   {q}        -- because q |= !p|q, i.e. q |= (p->q)
        # So the only remainder is {} and B / (p->q) = {}.
        # Re-adding (p->q) gives {p->q}, which does NOT entail q.
        # Therefore Recovery fails: q is in B but not in Cn((B / phi) + phi).
        B = B_from("p -> q", "q")
        self.assertFalse(
            check_contract_recovery(B, parse("p -> q"), contract)
        )

    def test_extensionality_top_disjunction(self):
        # Top === (p | !p).  Both tautologies -> both contractions are
        # no-ops -> trivially equal.
        self.assertTrue(check_contract_extensionality(
            B_from("q"), parse("Top"), parse("p | !p"), contract
        ))

    def test_extensionality_triple_negation(self):
        # !!!p === !p
        self.assertTrue(check_contract_extensionality(
            B_from("p", "q"), parse("!!!p"), parse("!p"), contract
        ))

    def test_extensionality_implication_disjunction(self):
        # (p -> q) === (!p | q)
        self.assertTrue(check_contract_extensionality(
            B_from("p", "p -> q"),
            parse("p -> q"),
            parse("!p | q"),
            contract,
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
