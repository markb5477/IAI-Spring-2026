"""Syntactic complexity ranking for propositional formulas.

Penalty weights, summed recursively:
    atom / Top / Bot   : 0
    !                  : 1
    & , |              : 2
    -> , <->           : 3

``rank(phi)`` returns a non-negative int.  Lower = simpler = more entrenched.
An atom has rank 0; a negated atom has rank 1; anything built from binary
connectives has rank >= 2, so literals always beat compound formulas.

Callers using this as a belief-base priority should negate the result so that
argmax-by-priority drops the most complex formula first (see ``belief_base``).
"""

from __future__ import annotations

from formula import Formula, Top, Bot, Var, Not, And, Or, Implies, Iff


def rank(phi: Formula) -> int:
    match phi:
        case Top() | Bot() | Var():
            return 0
        case Not(f=g):
            return 1 + rank(g)
        case And(left=a, right=b) | Or(left=a, right=b):
            return 2 + rank(a) + rank(b)
        case Implies(left=a, right=b) | Iff(left=a, right=b):
            return 3 + rank(a) + rank(b)
    raise TypeError(f"unknown formula node: {phi!r}")
