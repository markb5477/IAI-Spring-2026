# Belief Revision Engine

Course assignment for 02180 Introduction to AI (Spring 2026).
A propositional belief revision engine in the Hansson belief-base style,
with resolution-based entailment, priority-based partial meet contraction,
expansion, AGM revision via the Levi identity, and an executable AGM
postulate harness.

## Requirements

- Python 3.10 or later (the code uses `match` statements and
  `dataclass(frozen=True, slots=True)`)
- No third-party dependencies

## Running the interactive shell

From this directory:

```
python3 belief_revision.py
```

(`python3 shell.py` also works.) You will see a `B>` prompt. Type
`help` for the command list.

### Shell commands

| Command           | Meaning                                   |
|-------------------|-------------------------------------------|
| `+ <formula>`     | expand:    `B + phi`                      |
| `- <formula>`     | contract:  `B / phi`                      |
| `* <formula>`     | revise:    `B * phi`                      |
| `? <formula>`     | entails:   does `B` entail `phi`?         |
| `show`            | list the current belief base by priority  |
| `clear`           | empty the belief base                     |
| `help`            | show the help message                     |
| `quit` / `exit`   | leave the shell                           |

### Formula syntax

| Connective     | Symbol         |
|----------------|----------------|
| negation       | `!A` or `~A`   |
| conjunction    | `A & B`        |
| disjunction    | `A \| B`       |
| implication    | `A -> B`       |
| biconditional  | `A <-> B`      |
| truth          | `Top` or `T`   |
| falsehood      | `Bot` or `F`   |
| grouping       | `( ... )`      |

Atoms are identifiers matching `[A-Za-z_][A-Za-z0-9_]*`.

### Example session

```
B> + p
B> + p -> q
B> show
  [ 0]  p
  [-3]  (p -> q)
B> ? q
True
B> * !q
B> show
  [ 0]  p
  [-1]  !q
B> ? q
False
```

The numbers in brackets are entrenchment priorities (higher = more
entrenched, derived from `-rank(phi)` over syntactic complexity).

## Running the AGM postulate checks

The five revision postulates and five basic contraction postulates are
implemented as predicates in `postulates.py`. A minimal session:

```
python3 -c "
from formula import parse
from belief_base import make
from revision import revise
from contraction import contract
from postulates import (
    check_success, check_inclusion, check_vacuity,
    check_consistency, check_extensionality,
    check_contract_inclusion, check_contract_vacuity,
    check_contract_success, check_contract_recovery,
    check_contract_extensionality,
)

B = [make(parse('p')), make(parse('p->q'))]

print('Revision postulates:')
print('  Success:       ', check_success(B, parse('!q'), revise))
print('  Inclusion:     ', check_inclusion(B, parse('!q'), revise))
print('  Vacuity:       ', check_vacuity(B, parse('r'), revise))
print('  Consistency:   ', check_consistency(B, parse('!q'), revise))
print('  Extensionality:', check_extensionality(B, parse('!q'), parse('!(!(!q))'), revise))

print('Contraction postulates:')
print('  Inclusion:     ', check_contract_inclusion(B, parse('q'), contract))
print('  Vacuity:       ', check_contract_vacuity(B, parse('r'), contract))
print('  Success:       ', check_contract_success(B, parse('q'), contract))
print('  Recovery:      ', check_contract_recovery(B, parse('q'), contract))
print('  Extensionality:', check_contract_extensionality(B, parse('q'), parse('!(!q)'), contract))
"
```

Note: Recovery may return `False` on some inputs (e.g.
`B = {p & q}` contracted by `p`). This is a known property of
belief-base partial meet contraction, not a bug; see the report.

## Module overview

| File                | Responsibility                                          |
|---------------------|---------------------------------------------------------|
| `formula.py`        | AST, tokenizer, parser, pretty-printer                  |
| `cnf.py`            | NNF and CNF conversion, clause / literal helpers        |
| `resolution.py`     | Resolution-based entailment                             |
| `ranking.py`        | Syntactic complexity score for formulas                 |
| `belief_base.py`    | `Belief` and `BeliefBase` types, priority assignment    |
| `expansion.py`      | AGM expansion `B + phi`                                 |
| `contraction.py`    | AGM partial meet contraction `B / phi`                  |
| `revision.py`       | AGM revision `B * phi` via the Levi identity            |
| `postulates.py`     | AGM postulate checkers (revision and contraction)       |
| `shell.py`          | Interactive CLI                                         |
| `belief_revision.py`| Top-level entry point (imports and runs `shell.main`)   |
