# Acceptance data v2: interim training-only diagnostics

Snapshot: 2026-09-25T12:58:37.464281+00:00

945 source trajectories; 2943 unique valid pairs; 2064 changed pairs. Collection is still running.

These are descriptive label counts, not a selection of epsilon. Feedback is the existing sentence proxy divided by 100.

| ε (0–1) | Metric points | Better | Tie | Worse | Better (changed only) | Tie (changed only) | Worse (changed only) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 779 | 1038 | 1126 | 779 | 159 | 1126 |
| 0.005 | 0.5 | 586 | 1483 | 874 | 586 | 604 | 874 |
| 0.01 | 1.0 | 484 | 1682 | 777 | 484 | 803 | 777 |
| 0.02 | 2.0 | 355 | 1929 | 659 | 355 | 1050 | 659 |
| 0.03 | 3.0 | 288 | 2091 | 564 | 288 | 1212 | 564 |
| 0.05 | 5.0 | 210 | 2318 | 415 | 210 | 1439 | 415 |

Randomly chosen second-state feedback signs (computed afterward): {"negative": 363, "positive": 248, "zero": 333}

The raw trajectories retain all candidates and flags. Filter `valid == true` for acceptance training; do not use reference feedback as verifier input. Keep exact identities visible when reporting distributions.

Manual review is still needed to distinguish lexical score movement from semantic improvement; increasing epsilon alone does not guarantee trustworthy labels.
