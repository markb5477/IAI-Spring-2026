"""Logical entailment primitive.

This is the only piece of actual propositional reasoning in the engine -
expansion/contraction/revision are all expressed in terms of calls to entails().

TODO (stage 2):  CNF conversion + resolution refutation, implemented by hand
                 (no external SAT libraries).  Must be read-only: do NOT mutate B.
"""


def entails(B, phi):
    # Convert  B together with {!phi}  to CNF, run resolution; return True iff
    # the empty clause (false) is derivable.
    raise NotImplementedError("entails: implement CNF + resolution here")
