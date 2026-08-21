# Sentinel checkpoint: H-TH3P4 and H-CU3AU

Status: **complete; stop before H-CSCL, H-NI3SN, SSL seed 43, incremental updates, or external candidates**

## Compact comparison

| Result | SOAP | corrected GAT | SSL-17 | SSL-29 |
|---|---:|---:|---:|---:|
| OOD Th3P4: TPR / known FPR | 100.0% / 9.1% | 100% / 0–9.1% (seeds 42–44) | 100.0% / 0.0% | 100.0% / 0.0% |
| OOD Th3P4 candidate rho | 5.281 | 7.339–18.809 | 2.105 | 1.695 |
| OOD Cu3Au: TPR / known FPR | 100.0% / 21.4% | N/A | 100.0% / 0.0% | 100.0% / 14.3% |
| OOD Cu3Au candidate rho | 14.603 | N/A | 2.890 | 2.140 |
| Th3P4 retrieval | rank 1, accepted | ranks 13, 11, 6; 0/3 correct | rank 1, accepted | rank 1, accepted |
| Cu3Au retrieval | rank 1, accepted | N/A | rank 1, accepted | rank 1, accepted |
| D_wrong/D_correct, Th3P4 | 3.110 | 0.285–0.697 | 2.085 | 5.668 |
| D_wrong/D_correct, Cu3Au | 2.131 | N/A | 2.248 | 4.456 |
| Confident wrong after prototype removal | 0/2 | 2/3 | 1/2 | 1/2 |
| Sentinel scoring runtime | 172.8 CPU s | 1.37 GPU s | included in 3.40 GPU s total | included in 3.40 GPU s total |
| Representation training cost in this study | none | historical checkpoint; none | 160.7 GPU min across all 4 SSL runs | same shared total |

EMD retrieved both sentinels at rank 1 but confidently accepted the wrong prototype after removal in 2/2 cases. Its sentinel runtime was 137.8 CPU seconds.

## Interpretation

### OOD

SSL survives the preregistered OOD criterion. Both seeds rejected every Th3P4 held-out group and the Cu3Au seed-1 candidate. SSL-17 had zero known-group false positives on both folds. SSL-29 had zero on H-TH3P4 and 2/14 known-group false positives on H-CU3AU (14.3%). SOAP also rejected both unseen families but produced 1/11 and 3/14 known-group false positives. Corrected GAT rejected all Th3P4 held-out groups for all three seeds, but it has no frozen Cu3Au model.

### Retrieval and removal

SOAP and both SSL seeds retrieved both correct prototypes at rank 1 and accepted them. SOAP safely abstained in both correct-prototype-removal cases. Each SSL seed confidently accepted one wrong prototype, on different families: SSL-17 accepted BiF3 for Th3P4; SSL-29 accepted Al3Ti for Cu3Au. SSL-29 improved median separation by more than 25%, whereas SSL-17 did not. Therefore the two-seed retrieval stop condition is not met; the result is mixed and seed-dependent rather than a decisive SSL retrieval win.

Corrected supervised GAT demonstrates the distinction between novelty and identity: it detects unseen Th3P4 extremely strongly, but ranks the correct prototype only 13th, 11th, and 6th for seeds 42–44. Two seeds confidently accept a wrong identity. The supervised embedding is not a universal retrieval space.

## Preregistered decisions

1. **Did SSL survive the OOD criterion?** Yes. `OOD branch stop = FALSE`.
2. **Did SSL survive the retrieval criterion?** Mixed but not stopped. `Retrieval branch stop = FALSE` because seed 29 exceeds the 25% median-separation criterion while seed 17 does not; removal safety is worse than SOAP for both seeds.
3. **Did SSL materially improve anything over SOAP?** Yes for known-group preservation during OOD scoring, especially SSL-17. No robust improvement in safe prototype removal.
4. **Any catastrophic family failure?** No OOD or top-1 identity catastrophe. There are seed-dependent confident-wrong removal failures that preclude calling SSL a safe universal matcher.
5. **Was preprocessing fair?** Yes within the frozen representation definitions. Every method received the same extracted physical domain and domain-local scale/context. SOAP used its frozen M3 reliable-center pooling inside that domain; GAT/SSL/EMD used their frozen full-domain representations. No method saw the full GSD as a shared global SOAP context.
6. **Should remaining folds run?** The frozen rules say the SSL OOD branch survives and mixed retrieval results may proceed. However, conditional SSL seed 43 is permitted first because seeds 17 and 29 differ by 14.3 percentage points in H-CU3AU known FPR and reach different prototype-removal decisions. No additional job is launched without approval.

`Entire SSL encoder stop = FALSE`. A tie does not displace SOAP. At this checkpoint, the strongest OOD result is SSL (especially seed 17), while SOAP remains the safest retrieval/abstention representation.

## Firewall and provenance

- Opened internal sentinels only: Th3P4 seed 1 and Cu3Au seed 1, plus frozen v3-A internal validation.
- External candidates, Steven examples, untouched tests, fused examples, candidate-novel payloads, RDF/S(k) external panels, H-CSCL, and H-NI3SN remain closed.
- SSL output hash: `87041b63f2565d207b996541900b6bbf9571816867a92827badf471664bc0f74`
- fixed SOAP/EMD output hash: `6eff02e4e5c52c804f9a07f385e477258824d588b5b2093484f6f2d057a14540`
- corrected GAT output hash: `02e48b665278ea2b69b56eb08fcd220b9398ef5d721dea7dd32f16d0852cb8ba`
