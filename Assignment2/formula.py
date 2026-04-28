"""Propositional formula AST, tokenizer, parser, and pretty-printer.

The AST is a closed family of frozen dataclasses, all derived from the abstract
base ``Formula``.  Frozen dataclasses give us:

  - structural equality and hashing (so AST nodes can live in sets / frozensets),
  - immutability (no accidental in-place mutation during CNF conversion),

Concrete node types:

    Top()                              # logical truth
    Bot()                              # logical falsehood
    Var(name: str)                     # propositional atom
    Not(f: Formula)                    # negation
    And(left: Formula, right: Formula)
    Or(left: Formula, right: Formula)
    Implies(left: Formula, right: Formula)
    Iff(left: Formula, right: Formula)

A *literal* is either a ``Var`` or a ``Not`` whose inner formula is a ``Var``;
the ``Literal`` alias narrows that intent for the CNF layer.

Concrete syntax (keyboard-only):
    !A / ~A    A & B    A | B    A -> B    A <-> B    (A)    Top|T    Bot|F
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union


class Formula:
    """Abstract base for AST nodes.  Do not instantiate directly."""
    __slots__ = ()


@dataclass(frozen=True, slots=True)
class Top(Formula):
    pass


@dataclass(frozen=True, slots=True)
class Bot(Formula):
    pass


@dataclass(frozen=True, slots=True)
class Var(Formula):
    name: str


@dataclass(frozen=True, slots=True)
class Not(Formula):
    f: Formula


@dataclass(frozen=True, slots=True)
class And(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True, slots=True)
class Or(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True, slots=True)
class Implies(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True, slots=True)
class Iff(Formula):
    left: Formula
    right: Formula


Literal = Union[Var, Not]   # Not's inner must be a Var; enforced at construction sites

TOP: Top = Top()
BOT: Bot = Bot()


# ---------------------------------------------------------------------------
# Tokenizer / parser
# ---------------------------------------------------------------------------

_TOK = re.compile(r"\s*(<->|->|[!~&|()]|[A-Za-z_][A-Za-z0-9_]*)")
_CONST: dict[str, Formula] = {"Top": TOP, "T": TOP, "Bot": BOT, "F": BOT}


def tokenize(s: str) -> list[str]:
    toks: list[str] = []
    i = 0
    while i < len(s):
        m = _TOK.match(s, i)
        if not m:
            if s[i:].strip() == "":
                break
            raise ValueError(f"unexpected character {s[i]!r} at {i}")
        t = m.group(1)
        toks.append("!" if t == "~" else t)
        i = m.end()
    return toks


class _Parser:
    def __init__(self, toks: list[str]) -> None:
        self.t: list[str] = toks
        self.i: int = 0

    def _peek(self) -> str | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def _eat(self, x: str | None = None) -> str:
        tok = self._peek()
        if tok is None:
            raise ValueError("unexpected end of input")
        if x is not None and tok != x:
            raise ValueError(f"expected {x!r}, got {tok!r}")
        self.i += 1
        return tok

    def parse(self) -> Formula:
        f = self._iff()
        if self._peek() is not None:
            raise ValueError(f"unexpected token {self._peek()!r}")
        return f

    def _iff(self) -> Formula:
        f = self._impl()
        while self._peek() == "<->":
            self._eat()
            f = Iff(f, self._impl())
        return f

    def _impl(self) -> Formula:
        f = self._or()
        if self._peek() == "->":
            self._eat()
            return Implies(f, self._impl())
        return f

    def _or(self) -> Formula:
        f = self._and()
        while self._peek() == "|":
            self._eat()
            f = Or(f, self._and())
        return f

    def _and(self) -> Formula:
        f = self._not()
        while self._peek() == "&":
            self._eat()
            f = And(f, self._not())
        return f

    def _not(self) -> Formula:
        if self._peek() == "!":
            self._eat()
            return Not(self._not())
        return self._atom()

    def _atom(self) -> Formula:
        tok = self._eat()
        if tok == "(":
            f = self._iff()
            self._eat(")")
            return f
        if tok in _CONST:
            return _CONST[tok]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
            return Var(tok)
        raise ValueError(f"unexpected token {tok!r}")


def parse(s: str) -> Formula:
    return _Parser(tokenize(s)).parse()


# ---------------------------------------------------------------------------
# Pretty printer / helpers
# ---------------------------------------------------------------------------

def pretty(phi: Formula) -> str:
    match phi:
        case Top():           return "Top"
        case Bot():           return "Bot"
        case Var(name=n):     return n
        case Not(f=g):        return "!" + pretty(g)
        case And(left=a, right=b):     return f"({pretty(a)} & {pretty(b)})"
        case Or(left=a, right=b):      return f"({pretty(a)} | {pretty(b)})"
        case Implies(left=a, right=b): return f"({pretty(a)} -> {pretty(b)})"
        case Iff(left=a, right=b):     return f"({pretty(a)} <-> {pretty(b)})"
    raise TypeError(f"unknown formula node: {phi!r}")


def neg(phi: Formula) -> Formula:
    return Not(phi)
