# Matcher v3 structure-factor validation

This versioned directory repairs and validates the reciprocal-space channel without
modifying the failed matcher-v2 outputs.

Status: the structure-factor regression suite passes 11/11 and the approved
Nicole CsCl-only ranking diagnostic passes 23/23 for RDF-only, S(k)-only, and the
anchor-scaled combined ranking. This is a validated reciprocal-space component,
not yet a production matcher: no acceptance threshold has been fitted and the
known-broken Th3P4 domain extractor has not been repaired.

The channel deliberately separates:

- raw, particle-count-audited Bragg coherence used as an order/evidence gate;
- positive-area-normalized curve shape used for peak-location and relative-intensity matching.

No GAT labels, candidate answers, old RDF thresholds, or matcher-v1 thresholds are used.

See `structure_factor_validation_report.md` for results and limitations.
