"""Function-encoder training for Mode F in Phase 3.

This is the only package under ident/ that may import Torch (project invariant 2).
The Phase 1 and 2 identification paths do not import this
package; the deployed artifact is a frozen tabulation wrapped by
ident.features.function_encoder.FunctionEncoderDict, which is NumPy-only.

Training learns a basis spanning a corpus of mu(I) laws by closed-form projection
(docs/FUNCTION_ENCODER.md Section 2). It does not differentiate the simulator or
perform rollout matching (invariant 1). The basis is trained offline on mu(I)
functions and then used in the same linear weak-form
solve as Modes C and P.
"""
