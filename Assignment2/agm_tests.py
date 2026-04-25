"""agm_tests.py
==============
Automated tests for the five core AGM revision postulates.

Postulates tested
-----------------
1. Success       φ ∈ B * φ
                 The new information is always in the revised base.

2. Inclusion     B * φ ⊆ B + φ
                 Revision never adds beliefs beyond what expansion would.

3. Vacuity       If B ⊭ ¬φ  then  B * φ = B + φ
                 If φ does not contradict B, revision equals expansion.

4. Consistency   If φ is consistent then B * φ is consistent.
                 Revision of a consistent formula yields a consistent base.

5. Extensionality  If φ ≡ ψ  then  B * φ = B * ψ  (same entailments)
                 Logically equivalent inputs produce equivalent results.

Run:
    python3 agm_tests.py
"""

from formula     import parse, pretty, neg
from entailment  import entails
from expansion   import expand
from revision    import revise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_belief_base(formula_strings):
    """Build a belief base from a list of formula strings."""
    from belief_base import priority_of
    return [(parse(f), priority_of(parse(f))) for f in formula_strings]


def is_consistent(B):
    """
    B is consistent iff it does not entail both some atom and its negation.
    We use a fresh atom guaranteed not to appear in B.
    """
    fresh = parse("__fresh__")
    return not (entails(B, fresh) and entails(B, neg(fresh)))


def bases_equivalent(B1, B2):
    """
    Two belief bases are equivalent iff they entail exactly the same formulas.
    We approximate this by checking cross-entailment of every formula in each base.
    """
    for phi, _ in B1:
        if not entails(B2, phi):
            return False
    for phi, _ in B2:
        if not entails(B1, phi):
            return False
    return True


def run_test(name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    msg = f"[{status}]  {name}"
    if details:
        msg += f"  —  {details}"
    print(msg)
    return passed


# ---------------------------------------------------------------------------
# Individual postulate checks
# ---------------------------------------------------------------------------

def test_success(B, phi):
    """Success: φ must be in (entailed by) the revised base."""
    revised = revise(B[:], phi)
    ok = entails(revised, phi)
    return run_test("Success", ok,
                    f"B*φ ⊨ {pretty(phi)}  →  {ok}")


def test_inclusion(B, phi):
    """Inclusion: every formula in B*φ must be entailed by B+φ."""
    revised  = revise(B[:], phi)
    expanded = expand(B[:], phi)
    ok = all(entails(expanded, f) for f, _ in revised)
    return run_test("Inclusion", ok,
                    "every belief in B*φ is entailed by B+φ")


def test_vacuity(B, phi):
    """Vacuity: if B ⊭ ¬φ then B*φ and B+φ must be equivalent."""
    if entails(B, neg(phi)):
        return run_test("Vacuity", True, "N/A (B already entails ¬φ)")
    revised  = revise(B[:], phi)
    expanded = expand(B[:], phi)
    ok = bases_equivalent(revised, expanded)
    return run_test("Vacuity", ok,
                    "B ⊭ ¬φ  ⟹  B*φ ≡ B+φ")


def test_consistency(B, phi):
    """Consistency: if φ is consistent, B*φ must be consistent."""
    if not is_consistent([(phi, 0)]):
        return run_test("Consistency", True, "N/A (φ is itself inconsistent)")
    revised = revise(B[:], phi)
    ok = is_consistent(revised)
    return run_test("Consistency", ok,
                    "B*φ is consistent")


def test_extensionality(B, phi, psi):
    """Extensionality: if φ ≡ ψ then B*φ ≡ B*ψ."""
    if not (entails([(phi, 0)], psi) and entails([(psi, 0)], phi)):
        return run_test("Extensionality", True,
                        "N/A (φ and ψ are not logically equivalent)")
    rev_phi = revise(B[:], phi)
    rev_psi = revise(B[:], psi)
    ok = bases_equivalent(rev_phi, rev_psi)
    return run_test("Extensionality", ok,
                    f"φ≡ψ  ⟹  B*{pretty(phi)} ≡ B*{pretty(psi)}")


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------

def run_suite(label, kb_strings, phi_str, psi_str=None):
    """Run all five postulates for one (B, φ) scenario."""
    print(f"\n{'='*55}")
    print(f"Scenario: {label}")
    print(f"  B   = {{ {', '.join(kb_strings)} }}")
    print(f"  φ   = {phi_str}")
    if psi_str:
        print(f"  ψ   = {psi_str}  (should be ≡ φ)")
    print(f"{'='*55}")

    B   = make_belief_base(kb_strings)
    phi = parse(phi_str)
    psi = parse(psi_str) if psi_str else neg(neg(phi))   # default ψ = ¬¬φ ≡ φ

    results = [
        test_success(B, phi),
        test_inclusion(B, phi),
        test_vacuity(B, phi),
        test_consistency(B, phi),
        test_extensionality(B, phi, psi),
    ]
    total  = len(results)
    passed = sum(results)
    print(f"\n  {passed}/{total} postulates passed.")
    return passed, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_passed = all_total = 0

    # --- Scenario 1: basic modus ponens KB, revise with consistent formula ---
    p, t = run_suite(
        label      = "Modus ponens KB, consistent revision",
        kb_strings = ["p", "p -> q"],
        phi_str    = "r",
        psi_str    = "!!r",          # !!r ≡ r
    )
    all_passed += p; all_total += t

    # --- Scenario 2: revise with formula that contradicts KB ---
    p, t = run_suite(
        label      = "KB entails q, revise with !q",
        kb_strings = ["p", "p -> q"],
        phi_str    = "!q",
        psi_str    = "!(p & !p) -> !q",   # tautology -> !q  ≡  !q
    )
    all_passed += p; all_total += t

    # --- Scenario 3: empty KB ---
    p, t = run_suite(
        label      = "Empty KB",
        kb_strings = [],
        phi_str    = "p -> q",
        psi_str    = "!p | q",       # !p | q  ≡  p -> q
    )
    all_passed += p; all_total += t

    # --- Scenario 4: KB with compound formulas ---
    p, t = run_suite(
        label      = "Compound KB, biconditional revision",
        kb_strings = ["p | q", "p -> r", "q -> r"],
        phi_str    = "!r",
    )
    all_passed += p; all_total += t

    print(f"\n{'='*55}")
    print(f"TOTAL: {all_passed}/{all_total} postulates passed across all scenarios.")
    print(f"{'='*55}")
