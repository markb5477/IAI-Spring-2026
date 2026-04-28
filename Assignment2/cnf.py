"""CNF conversion pipeline and clause utilities.

Type aliases:
    Clause = frozenset[Literal]    # disjunction of literals
    CNF    = set[Clause]           # conjunction of clauses

Pipeline (kept as separate functions so the report can show each step on the
slide example):

    to_cnf  =  distribute_or_over_and  ∘  push_negations
                                       ∘  eliminate_implies
                                       ∘  eliminate_iff
"""

from __future__ import annotations

from formula import (
    Formula, Top, Bot, Var, Not, And, Or, Implies, Iff,
    Literal, TOP, BOT,
)


Clause = frozenset[Literal]
CNF = set[Clause]


# ---------------------------------------------------------------------------
# Literal & clause helpers
# ---------------------------------------------------------------------------

def negate_literal(lit: Literal) -> Literal:
    """Logical negation of a literal, with double-negation collapse."""
    match lit:
        case Var():
            return Not(lit)
        case Not(f=Var()) as n:
            assert isinstance(n.f, Var)
            return n.f
    raise TypeError(f"not a well-formed literal: {lit!r}")


def is_tautology(clause: Clause) -> bool:
    """True iff the clause contains some literal alongside its negation."""
    for lit in clause:
        if negate_literal(lit) in clause:
            return True
    return False


# ---------------------------------------------------------------------------
# CNF pipeline
# ---------------------------------------------------------------------------

def eliminate_iff(phi: Formula) -> Formula:
    match phi:
        case Top() | Bot() | Var():
            return phi
        case Not(f=g):
            return Not(eliminate_iff(g))
        case Iff(left=a, right=b):
            ea, eb = eliminate_iff(a), eliminate_iff(b)
            return And(Implies(ea, eb), Implies(eb, ea))
        case And(left=a, right=b):
            return And(eliminate_iff(a), eliminate_iff(b))
        case Or(left=a, right=b):
            return Or(eliminate_iff(a), eliminate_iff(b))
        case Implies(left=a, right=b):
            return Implies(eliminate_iff(a), eliminate_iff(b))
    raise TypeError(f"unknown formula node: {phi!r}")


def eliminate_implies(phi: Formula) -> Formula:
    match phi:
        case Top() | Bot() | Var():
            return phi
        case Not(f=g):
            return Not(eliminate_implies(g))
        case Implies(left=a, right=b):
            return Or(Not(eliminate_implies(a)), eliminate_implies(b))
        case And(left=a, right=b):
            return And(eliminate_implies(a), eliminate_implies(b))
        case Or(left=a, right=b):
            return Or(eliminate_implies(a), eliminate_implies(b))
        case Iff():
            # Defensive dead branch: unreachable in normal pipeline flow
            # (eliminate_iff runs first), but guards against direct callers
            # who skip the iff-elimination step.
            return eliminate_implies(eliminate_iff(phi))
    raise TypeError(f"unknown formula node: {phi!r}")


def push_negations(phi: Formula) -> Formula:
    """Push negations down to atoms (NNF).  Assumes iff/impl already eliminated."""
    match phi:
        case Top() | Bot() | Var():
            return phi
        case Not(f=Top()):
            return BOT
        case Not(f=Bot()):
            return TOP
        case Not(f=Var()):
            return phi
        case Not(f=Not(f=g)):
            return push_negations(g)
        case Not(f=And(left=a, right=b)):
            return Or(push_negations(Not(a)), push_negations(Not(b)))
        case Not(f=Or(left=a, right=b)):
            return And(push_negations(Not(a)), push_negations(Not(b)))
        case And(left=a, right=b):
            return And(push_negations(a), push_negations(b))
        case Or(left=a, right=b):
            return Or(push_negations(a), push_negations(b))
    raise TypeError(f"unexpected node in push_negations: {phi!r}")


def to_nnf(phi: Formula) -> Formula:
    return push_negations(eliminate_implies(eliminate_iff(phi)))


def distribute_or_over_and(phi: Formula) -> Formula:
    """Distribute | over &.  Input must already be in NNF."""
    match phi:
        case Top() | Bot() | Var() | Not():
            return phi
        case And(left=a, right=b):
            return And(distribute_or_over_and(a), distribute_or_over_and(b))
        case Or(left=a, right=b):
            da = distribute_or_over_and(a)
            db = distribute_or_over_and(b)
            match da:
                case And(left=x, right=y):
                    return And(distribute_or_over_and(Or(x, db)),
                               distribute_or_over_and(Or(y, db)))
            match db:
                case And(left=x, right=y):
                    return And(distribute_or_over_and(Or(da, x)),
                               distribute_or_over_and(Or(da, y)))
            return Or(da, db)
    raise TypeError(f"unexpected node in distribute_or_over_and: {phi!r}")


# ---------------------------------------------------------------------------
# Clause extraction
# ---------------------------------------------------------------------------

def _collect_disjuncts(phi: Formula, acc: list[Formula]) -> None:
    match phi:
        case Or(left=a, right=b):
            _collect_disjuncts(a, acc)
            _collect_disjuncts(b, acc)
        case _:
            acc.append(phi)


def _collect_conjuncts(phi: Formula, acc: list[Formula]) -> None:
    match phi:
        case And(left=a, right=b):
            _collect_conjuncts(a, acc)
            _collect_conjuncts(b, acc)
        case _:
            acc.append(phi)


def _as_literal(node: Formula) -> Literal:
    """Validate that a leaf node is a well-formed literal."""
    match node:
        case Var():
            return node
        case Not(f=Var()):
            return node
    raise TypeError(f"non-literal in clause: {node!r}")


def to_cnf(phi: Formula) -> CNF:
    """Convert phi to a set of clauses.

    Top -> empty CNF (trivially true).
    Bot -> {frozenset()} (single empty clause = unsatisfiable).
    Tautological clauses are dropped.
    """
    g = distribute_or_over_and(to_nnf(phi))
    conjuncts: list[Formula] = []
    _collect_conjuncts(g, conjuncts)
    clauses: CNF = set()
    for c in conjuncts:
        match c:
            case Top():
                continue
            case Bot():
                return {frozenset()}
        disj: list[Formula] = []
        _collect_disjuncts(c, disj)
        if any(isinstance(lit, Top) for lit in disj):
            continue
        lits: set[Literal] = set()
        for lit in disj:
            if isinstance(lit, Bot):
                continue
            lits.add(_as_literal(lit))
        clause: Clause = frozenset(lits)
        if is_tautology(clause):
            continue
        clauses.add(clause)
    return clauses


