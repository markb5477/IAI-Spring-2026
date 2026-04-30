"""Syntactic complexity ranking for propositional formulas.

Penalty weights, summed recursively:
    atom         : 0
    !            : 1
    &            : 2
    |            : 3
    ->           : 4
    <->          : 5

rank(phi) returns a non-negative int.  Lower = simpler = more entrenched.
An atom has rank 0; a negated atom has rank 1; anything built from binary
connectives has rank >= 2, so literals always beat compound formulas.

Callers that use this as a belief-base priority should negate the result
(so the existing argmin-by-priority logic in contraction drops the most
complex formula first).

Tiebreaking: rank is deterministic. When several formulas share a rank,
resolve the tie at the call site with random.choice.
"""

_W = {"not": 1, "and": 2, "or": 3, "impl": 4, "iff": 5}


def rank(phi):
    k = phi[0]
    if k == "var":
        return 0
    if k == "not":
        return _W["not"] + rank(phi[1])
    return _W[k] + rank(phi[1]) + rank(phi[2])
