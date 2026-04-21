"""Propositional formula AST, tokenizer, parser, and pretty-printer.

AST nodes are plain tuples:
    ('var',  name)
    ('not',  F)
    ('and',  F, G)   ('or', F, G)
    ('impl', F, G)   ('iff', F, G)

Syntax (keyboard-only input):
    !A / ~A    A & B    A | B    A -> B    A <-> B    (A)

Precedence, highest to lowest:  !   &   |   ->   <->
Implication is right-associative; everything else is left-associative.
"""

import re

_TOK = re.compile(r"\s*(<->|->|[!~&|()]|[A-Za-z_][A-Za-z0-9_]*)")


def tokenize(s):
    toks, i = [], 0
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
    def __init__(self, toks): self.t, self.i = toks, 0
    def _peek(self): return self.t[self.i] if self.i < len(self.t) else None
    def _eat(self, x=None):
        tok = self._peek()
        if tok is None: raise ValueError("unexpected end of input")
        if x is not None and tok != x: raise ValueError(f"expected {x!r}, got {tok!r}")
        self.i += 1; return tok
    def parse(self):
        f = self._iff()
        if self._peek() is not None: raise ValueError(f"unexpected token {self._peek()!r}")
        return f
    def _iff(self):
        f = self._impl()
        while self._peek() == "<->": self._eat(); f = ("iff", f, self._impl())
        return f
    def _impl(self):
        f = self._or()
        if self._peek() == "->": self._eat(); return ("impl", f, self._impl())
        return f
    def _or(self):
        f = self._and()
        while self._peek() == "|": self._eat(); f = ("or", f, self._and())
        return f
    def _and(self):
        f = self._not()
        while self._peek() == "&": self._eat(); f = ("and", f, self._not())
        return f
    def _not(self):
        if self._peek() == "!": self._eat(); return ("not", self._not())
        return self._atom()
    def _atom(self):
        tok = self._eat()
        if tok == "(":
            f = self._iff(); self._eat(")"); return f
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok): return ("var", tok)
        raise ValueError(f"unexpected token {tok!r}")


def parse(s):
    return _Parser(tokenize(s)).parse()


def pretty(f):
    k = f[0]
    if k == "var":  return f[1]
    if k == "not":  return "!" + pretty(f[1])
    op = {"and": "&", "or": "|", "impl": "->", "iff": "<->"}[k]
    return f"({pretty(f[1])} {op} {pretty(f[2])})"


def neg(f):
    return ("not", f)
