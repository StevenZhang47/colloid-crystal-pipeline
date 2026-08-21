# Pre-scoring baseline checkpoint

Status: **complete; sentinel candidates and external payloads remain closed**  
Completed GAT regeneration job: `15774378`

## Frozen assets

| Arm | Sentinel coverage | Development status | Freeze SHA-256 |
|---|---|---|---|
| SOAP OOD + retrieval | H-TH3P4, H-CU3AU | frozen | `e78366f129ee1de54d5e1ee6d9477012167cccf0ea5608c78e28c69a6cbb44a6` |
| EMD RDF/S(k) retrieval | H-TH3P4, H-CU3AU | frozen | `1344864ec35e142847a4b479013b18cedab9c14f044fffa3b539906fbd0d3676` |
| corrected supervised GAT | H-TH3P4 only | frozen | `aec830af085398fe7d99a5ac1808b2a042f85e5ba2e2f1d0fa3030fcf4eb1c1f` |
| SSL seed 17/29 | H-TH3P4, H-CU3AU | four encoders frozen | see per-run completion records |

The corrected GAT arm is unavailable for H-CU3AU under the frozen supervised-baseline policy. It is not extrapolated or retrained.

## Development-only retrieval diagnostics

| Method/fold | Supported runs | Top-1 | Known accepted | Wrong accepted after correct-prototype removal |
|---|---:|---:|---:|---:|
| SOAP H-TH3P4 | 11 | 11/11 | 11/11 | 2/11 |
| SOAP H-CU3AU | 10 | 10/10 | 10/10 | 2/10 |
| EMD H-TH3P4 | 13 | 9/13 | 13/13 | 13/13 |
| EMD H-CU3AU | 12 | 8/12 | 12/12 | 12/12 |
| GAT H-TH3P4 seed 42 | 13 | 4/13 | 13/13 | 11/13 |
| GAT H-TH3P4 seed 43 | 13 | 2/13 | 13/13 | 13/13 |
| GAT H-TH3P4 seed 44 | 13 | 5/13 | 13/13 | 13/13 |

These are development diagnostics, not sentinel results. No method is retuned from them. `known accepted` is calibrated by construction and is distinct from correct prototype identity.

## GAT correction and validation

- Job `15726586` failed before any fit because a historical cached graph contained 1,023 nodes for a 1,024-particle frame.
- The historical builder mapping was reconstructed exactly; original particle 862 was a deliberately omitted one-particle non-dominant component.
- Development GAT graphs are now rebuilt directly on the frozen common physical-component mask.
- Final audit: 134/134 regions have exact common-domain node counts and exact species order.
- Job `15774378` completed with a zero-byte error log in 48.60 seconds.
- The correct-prototype-removal diagnostic now requires both frozen distance and margin gates, identically to SOAP and EMD.
- Full correction provenance is in `data/GAT_PREPARATION_IMPLEMENTATION_FIX.json`.

Two pre-score GAT preparations using identical corrected checkpoints showed small float32/GPU drift in fitted OOD thresholds (maximum relative difference below `5e-5`). The final job-15774378 assets above are the frozen assets. Sentinel scoring must record numerical repeat/decision stability; no threshold may be adjusted from a sentinel result.

## Firewall state

- `closed_sentinel_candidate_score_opened = false`
- `external_candidate_payload_accessed = false`
- No `SSL_SENTINEL_COMPLETE.json` exists.
- Steven examples, Phase-5 external candidates, untouched tests, fused examples, RDF/S(k) external panels, and remaining H-CSCL/H-NI3SN folds remain closed.

## Next authorized checkpoint

With explicit approval, score only H-TH3P4 and H-CU3AU for OOD, canonical-prototype retrieval, and correct-prototype removal; then create `SENTINEL_CHECKPOINT.md` and stop before H-CSCL or H-NI3SN.
