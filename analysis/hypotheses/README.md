# Preserved hypotheses

`custom-mask-prototype.patch` is an untested model-level experiment against the
v12 attention route. It replaces the ordinary one-query causal specialization
with an all-true custom mask. It was removed from active development after
3,840 fixed-Q/K/V query-row fixtures showed bitwise equality between the causal,
one-query custom-mask and multi-query verification paths. These fixtures do not
prove equivalence for every model activation, but provide no positive evidence
for adding this complexity. No GPU model image or public gate used the prototype.

Raw control: `results/modal/modal-20261004-attention-equivalence/`.
