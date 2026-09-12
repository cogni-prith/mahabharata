# Phase 6 entry gate

Checked before training, as required. No model was trained.

| condition | required | measured | result |
|---|---|---|---|
| distinct asserted triples | >= 10,000 | 635 | **FAIL** |
| entities of degree >= 5 | >= 500 | 61 | **FAIL** |
| largest relation share | <= 40% | 40.8% (speaksTo) | **FAIL** |

**The gate does not pass. Nothing was trained.**

Proposed triples, held separately and not eligible: 26.
Entities appearing in the asserted graph: 488.

## Degree distribution

| degree | entities |
|---|---|
| 1 | 286 |
| 2 | 74 |
| 3 | 46 |
| 4 | 21 |
| 5 | 16 |
| 6 | 8 |
| 7 | 6 |
| 8 | 8 |
| 9 | 1 |
| 10 | 3 |
| 11 | 4 |
| 12 | 2 |
| 13 | 1 |
| 14 | 1 |
| 15 | 1 |
| 16 | 2 |
| 17 | 2 |
| 19 | 1 |
| 20+ | 5 |

## Relation shares

| relation | triples | share |
|---|---|---|
| speaksTo | 259 | 40.8% |
| hasBiologicalParent | 237 | 37.3% |
| slainBy | 71 | 11.2% |
| rules | 37 | 5.8% |
| hasSpouse | 21 | 3.3% |
| cursedBy | 7 | 1.1% |
| instructedBy | 3 | 0.5% |
