#!/usr/bin/env python3
"""Entry point for the belief revision engine.

Run:
    python3 belief_revision.py

Module layout:
    formula.py      Frozen-dataclass AST + parser, pretty, neg, Top/Bot
    cnf.py          CNF pipeline + Clause/CNF type aliases
    resolution.py   entails(premises: Iterable[Formula], phi) via resolution
    belief_base.py  Belief dataclass, BeliefBase alias, priority_of, formulas_of
    expansion.py    expand(B, phi)
    contraction.py  contract(B, phi) via partial meet over remainders
    revision.py     revise(B, phi) via the Levi identity
    postulates.py   AGM postulate checkers
    shell.py        interactive CLI
"""

from shell import main

if __name__ == "__main__":
    main()
