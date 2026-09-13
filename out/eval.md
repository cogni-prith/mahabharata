# Phase 6 entry gate

Checked before training, as required. No model was trained.

| condition | required | measured | result |
|---|---|---|---|
| distinct asserted triples | >= 10,000 | 1949 | **FAIL** |
| entities of degree >= 5 | >= 500 | 182 | **FAIL** |
| largest relation share | <= 40% | 39.9% (speaksTo) | PASS |

**The gate does not pass. Nothing was trained.**

Proposed triples, held separately and not eligible: 26.
Entities appearing in the asserted graph: 724.

## Degree distribution

| degree | entities |
|---|---|
| 1 | 322 |
| 2 | 110 |
| 3 | 63 |
| 4 | 47 |
| 5 | 35 |
| 6 | 29 |
| 7 | 12 |
| 8 | 9 |
| 9 | 10 |
| 10 | 13 |
| 11 | 9 |
| 12 | 3 |
| 13 | 6 |
| 14 | 2 |
| 15 | 1 |
| 16 | 2 |
| 17 | 2 |
| 18 | 3 |
| 19 | 5 |
| 20+ | 41 |

## Relation shares

| relation | triples | share |
|---|---|---|
| speaksTo | 777 | 39.9% |
| foughtWith | 520 | 26.7% |
| hasBiologicalParent | 372 | 19.1% |
| slainBy | 109 | 5.6% |
| hasSpouse | 57 | 2.9% |
| rules | 39 | 2.0% |
| defeatedBy | 38 | 1.9% |
| cursedBy | 20 | 1.0% |
| instructedBy | 17 | 0.9% |
