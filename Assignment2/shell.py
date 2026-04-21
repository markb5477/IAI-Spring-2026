"""Interactive CLI shell for the belief revision engine."""

from formula import parse, pretty
from entailment import entails
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

Formula syntax: !A   A&B   A|B   A->B   A<->B   (A)
"""


def _show(B):
    if not B:
        print("  (empty)"); return
    for phi, p in sorted(B, key=lambda x: -x[1]):
        print(f"  [{p:>2}]  {pretty(phi)}")


def run_line(B, line):
    """Execute one shell line. Returns the (possibly new) belief base."""
    line = line.strip()
    if not line: return B
    head, _, rest = line.partition(" ")
    rest = rest.strip()
    try:
        if head in ("+", "expand"):     return expand(B,   parse(rest))
        if head in ("-", "contract"):   return contract(B, parse(rest))
        if head in ("*", "revise"):     return revise(B,   parse(rest))
        if head in ("?", "entails"):
            if not rest: print("usage: ? <formula>")
            else:        print(entails(B, parse(rest)))
            return B
        if head in ("show", "list", "B"):  _show(B);     return B
        if head == "clear":                               return []
        if head == "help":                 print(HELP);  return B
        if head in ("quit", "exit"):       raise SystemExit
        print(f"unknown command: {head!r}. type 'help'.")
    except NotImplementedError as e:
        print(f"not implemented yet: {e}")
    except ValueError as e:
        print(f"parse error: {e}")
    return B


def main():
    B = []
    print("Belief Revision Engine.  Type 'help' for commands, 'quit' to exit.")
    while True:
        try:
            line = input("B> ")
        except (EOFError, KeyboardInterrupt):
            print(); break
        try:
            B = run_line(B, line)
        except SystemExit:
            break
