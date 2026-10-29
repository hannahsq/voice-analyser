"""Invariance and equivariance probe suite (roadmap item 7, D-050).

A probe renders each source item under a family of paired conditions. Each condition carries the change it imposes,
as a log2 factor per output (0 where the output should not move). Every model is scored on the same cached renders
with the same measures (`measures`): per-item slopes against the imposed change, a within-item shuffle null, and the
change in the model's own sigma.
"""
