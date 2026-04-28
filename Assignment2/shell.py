"""Interactive CLI shell for the belief revision engine."""

from __future__ import annotations

from formula import parse, pretty
from belief_base import BeliefBase, formulas_of
from resolution import entails
from expansion import expand
from contraction import contract
from revision import revise

HELP = """\
Commands:
  + <formula>   | expand    B + phi
  - <formula>   | contract  B / phi
  * <formula>   | revise    B * phi
  ? <formula>   | entails   does B entail phi?
  show                      list belief base (sorted by priority, highest first)
  clear                     empty the belief base
  help                      this message
  quit | exit               leave

Formula syntax: !A   A&B   A|B   A->B   A<->B   (A)   Top|T   Bot|F
"""


def _show(B: BeliefBase) -> None:
    if not B:
        print("  (empty)")
        return
    for b in sorted(B, key=lambda x: -x.priority):
        print(f"  [{b.priority:>2}]  {pretty(b.formula)}")


def run_line(B: BeliefBase, line: str) -> BeliefBase:
    """Execute one shell line.  Returns the (possibly new) belief base."""
    line = line.strip()
    if not line:
        return B
    head, _, rest = line.partition(" ")
    rest = rest.strip()
    try:
        if head in ("+", "expand"):     return expand(B,   parse(rest))
        if head in ("-", "contract"):   return contract(B, parse(rest))
        if head in ("*", "revise"):     return revise(B,   parse(rest))
        if head in ("?", "entails"):
            if not rest:
                print("usage: ? <formula>")
            else:
                print(entails(formulas_of(B), parse(rest)))
            return B
        if head in ("show", "list", "B"):  _show(B);    return B
        if head == "clear":                              return []
        if head == "help":                 print(HELP); return B
        if head in ("quit", "exit"):       raise SystemExit
        print(f"unknown command: {head!r}. type 'help'.")
    except ValueError as e:
        print(f"parse error: {e}")
    return B


def main() -> None:
    B: BeliefBase = []
    print("Belief Revision Engine.  Type 'help' for commands, 'quit' to exit.")
    while True:
        try:
            line = input("B> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        try:
            B = run_line(B, line)
        except SystemExit:
            break
