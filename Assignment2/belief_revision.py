#!/usr/bin/env python3
"""Entry point for the belief revision engine.

Run:
    python3 belief_revision.py

Module layout:
    formula.py      AST, tokenizer, parser, pretty-printer, neg
    belief_base.py  the (formula, priority) list and helpers
    entailment.py   entails(B, phi)     - stage 2, the only primitive
    expansion.py    expand(B, phi)      - stage 4
    contraction.py  contract(B, phi)    - stage 3
    revision.py     revise(B, phi)      - Levi identity
    shell.py        interactive CLI
"""

from shell import main

if __name__ == "__main__":
    main()
