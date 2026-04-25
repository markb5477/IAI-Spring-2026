"""entailment.py
================
Logical entailment via CNF conversion + resolution refutation.

    entails(B, phi)  →  True  iff  B ⊨ phi

Algorithm
---------
1. Negate the query:          {¬phi}
2. Combine with the KB:       B ∪ {¬phi}
3. Convert every formula to CNF and extract clauses (frozensets of literals).
4. Run propositional resolution until either:
     - the empty clause is derived  →  contradiction found  →  B ⊨ phi
     - no new clauses can be added  →  B ⊭ phi

A *literal* is a plain string: "p" (positive) or "!p" (negative).
A *clause* is a frozenset of literals (disjunction).

CNF pipeline (all operating on tuple-AST from formula.py):
    eliminate_iff  →  eliminate_impl  →  push_not  →  distribute_or
then flatten the resulting And-tree into a list of clauses.

No external libraries are used.
"""

from formula import parse, neg


# ---------------------------------------------------------------------------
# 1.  CNF conversion  (tuple AST  →  tuple AST in CNF)
# ---------------------------------------------------------------------------

def _elim_iff(f):
    """Replace every <-> with two implications."""
    k = f[0]
    if k == "var":
        return f
    if k == "not":
        return ("not", _elim_iff(f[1]))
    if k == "iff":
        # A <-> B  ≡  (A -> B) & (B -> A)
        a, b = _elim_iff(f[1]), _elim_iff(f[2])
        return ("and", ("impl", a, b), ("impl", b, a))
    # and, or, impl
    return (k, _elim_iff(f[1]), _elim_iff(f[2]))


def _elim_impl(f):
    """Replace every -> with  ¬A ∨ B."""
    k = f[0]
    if k == "var":
        return f
    if k == "not":
        return ("not", _elim_impl(f[1]))
    if k == "impl":
        # A -> B  ≡  ¬A ∨ B
        a, b = _elim_impl(f[1]), _elim_impl(f[2])
        return ("or", ("not", a), b)
    # and, or
    return (k, _elim_impl(f[1]), _elim_impl(f[2]))


def _push_not(f):
    """Push negations inward using De Morgan; eliminate double negations."""
    k = f[0]
    if k == "var":
        return f
    if k == "not":
        inner = f[1]
        ik = inner[0]
        if ik == "var":
            return f                                    # ¬p  — already a literal
        if ik == "not":
            return _push_not(inner[1])                 # ¬¬A  →  A
        if ik == "and":
            # ¬(A ∧ B)  →  ¬A ∨ ¬B
            return _push_not(("or",  ("not", inner[1]), ("not", inner[2])))
        if ik == "or":
            # ¬(A ∨ B)  →  ¬A ∧ ¬B
            return _push_not(("and", ("not", inner[1]), ("not", inner[2])))
        raise ValueError(f"_push_not: unexpected node under not: {ik!r}")
    # and, or
    return (k, _push_not(f[1]), _push_not(f[2]))


def _distribute(f):
    """Distribute ∨ over ∧ to obtain CNF."""
    k = f[0]
    if k in ("var", "not"):          # literal — already in CNF
        return f
    if k == "and":
        return ("and", _distribute(f[1]), _distribute(f[2]))
    if k == "or":
        left  = _distribute(f[1])
        right = _distribute(f[2])
        # (A ∧ B) ∨ C  →  (A ∨ C) ∧ (B ∨ C)
        if left[0] == "and":
            return _distribute(("and",
                                 ("or", left[1], right),
                                 ("or", left[2], right)))
        # A ∨ (B ∧ C)  →  (A ∨ B) ∧ (A ∨ C)
        if right[0] == "and":
            return _distribute(("and",
                                 ("or", left, right[1]),
                                 ("or", left, right[2])))
        return ("or", left, right)
    raise ValueError(f"_distribute: unexpected node {k!r}")


def to_cnf(f):
    """Full CNF pipeline: returns a tuple AST whose top-level is a conjunction
    of disjunctions of literals."""
    f = _elim_iff(f)
    f = _elim_impl(f)
    f = _push_not(f)
    f = _distribute(f)
    return f


# ---------------------------------------------------------------------------
# 2.  Extract clauses from a CNF AST
# ---------------------------------------------------------------------------

def _literal_str(f):
    """Convert a CNF literal node to a string: 'p' or '!p'."""
    if f[0] == "var":
        return f[1]
    if f[0] == "not" and f[1][0] == "var":
        return "!" + f[1][1]
    raise ValueError(f"Not a literal: {f}")


def _collect_or(f):
    """Flatten an Or-tree of literals into a set of literal strings."""
    if f[0] == "or":
        return _collect_or(f[1]) | _collect_or(f[2])
    return {_literal_str(f)}


def _collect_and(f):
    """Flatten an And-tree of clauses into a list of frozensets."""
    if f[0] == "and":
        return _collect_and(f[1]) + _collect_and(f[2])
    # single clause (or-tree / literal)
    return [frozenset(_collect_or(f))]


def formula_to_clauses(f):
    """
    Convert a formula (tuple AST) to a list of clauses.
    Each clause is a frozenset of literal strings ('p', '!p').
    """
    cnf = to_cnf(f)
    return _collect_and(cnf)


# ---------------------------------------------------------------------------
# 3.  Resolution
# ---------------------------------------------------------------------------

def _resolve(ci, cj):
    """
    Try to resolve two clauses.
    Returns a frozenset (the resolvent) if exactly one complementary pair
    exists, otherwise returns None.
    """
    for lit in ci:
        complement = lit[1:] if lit.startswith("!") else "!" + lit
        if complement in cj:
            resolvent = (ci - {lit}) | (cj - {complement})
            return frozenset(resolvent)
    return None


def _is_tautology(clause):
    """A clause is a tautology if it contains both p and !p for some p."""
    for lit in clause:
        complement = lit[1:] if lit.startswith("!") else "!" + lit
        if complement in clause:
            return True
    return False


def _resolution(clauses):
    """
    Run propositional resolution on a list of clauses (frozensets).
    Returns True iff the empty clause is derivable (i.e. the set is UNSAT).
    """
    clause_set = set()
    for c in clauses:
        if not _is_tautology(c):
            clause_set.add(c)

    # Check for empty clause immediately
    if frozenset() in clause_set:
        return True

    seen_pairs = set()

    while True:
        clause_list = list(clause_set)
        new_clauses  = set()

        for i in range(len(clause_list)):
            for j in range(i + 1, len(clause_list)):
                ci, cj = clause_list[i], clause_list[j]
                pair = (ci, cj) if ci < cj else (cj, ci)  # canonical order
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                resolvent = _resolve(ci, cj)
                if resolvent is None:
                    continue
                if not resolvent:                   # empty clause → UNSAT
                    return True
                if _is_tautology(resolvent):
                    continue
                if resolvent not in clause_set:
                    new_clauses.add(resolvent)

        if not new_clauses:
            return False                            # saturated without empty clause → SAT

        clause_set |= new_clauses


# ---------------------------------------------------------------------------
# 4.  Public entailment interface
# ---------------------------------------------------------------------------

def entails(B, phi):
    """
    Return True iff the belief base B entails phi.

    Parameters
    ----------
    B   : list of tuple-AST formulas   (as stored in the belief base)
    phi : tuple-AST formula            (the query)

    Strategy: B ⊨ phi  iff  B ∪ {¬phi} is unsatisfiable.
    """
    if not B:
        # Empty belief base entails nothing (except tautologies).
        clauses = formula_to_clauses(neg(phi))
        return _resolution(clauses)

    all_clauses = []

    # Clauses from every belief in B  (B stores (formula, priority) pairs)
    for entry in B:
        formula = entry[0] if isinstance(entry, tuple) and len(entry) == 2 \
                            and not isinstance(entry[0], str) else entry
        # Defensive: handle both (ast, priority) pairs and raw AST tuples
        if isinstance(formula, tuple) and formula[0] in \
                ("var", "not", "and", "or", "impl", "iff"):
            all_clauses.extend(formula_to_clauses(formula))
        else:
            # formula is itself the AST (raw tuple from formula.py)
            all_clauses.extend(formula_to_clauses(entry))

    # Clauses from ¬phi
    all_clauses.extend(formula_to_clauses(neg(phi)))

    return _resolution(all_clauses)


# ---------------------------------------------------------------------------
# 5.  Smoke-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from formula import parse

    def check(kb_strs, query_str, expected):
        B   = [(parse(f), 0) for f in kb_strs]   # priority 0 — irrelevant here
        phi = parse(query_str)
        result = entails(B, phi)
        status = "OK" if result == expected else "FAIL"
        print(f"[{status}]  {kb_strs} ⊨ {query_str!r}  →  {result}  (expected {expected})")

    # Basic modus ponens
    check(["p", "p -> q"], "q", True)

    # Does not entail
    check(["p"], "q", False)

    # Tautology — entailed by empty KB
    check([], "p | !p", True)

    # Contradiction in KB entails everything
    check(["p", "!p"], "q", True)

    # Biconditional
    check(["p <-> q", "p"], "q", True)
    check(["p <-> q", "!p"], "q", False)

    # Chained implications
    check(["p -> q", "q -> r", "p"], "r", True)

    # De Morgan
    check(["!(p & q)"], "!p | !q", True)

    # Negation
    check(["p"], "!p", False)
    check(["!p"], "p", False)
