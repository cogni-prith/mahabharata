# Phase 6 entry gate

Checked before training, as required. No model was trained.

| condition | required | measured | result |
|---|---|---|---|
| distinct asserted triples | >= 10,000 | 861 | **FAIL** |
| entities of degree >= 5 | >= 500 | 82 | **FAIL** |
| largest relation share | <= 40% | 42.6% (hasBiologicalParent) | **FAIL** |

**The gate does not pass. Nothing was trained.**

Proposed triples, held separately and not eligible: 26.
Entities appearing in the asserted graph: 585.

## Degree distribution

| degree | entities |
|---|---|
| 1 | 311 |
| 2 | 110 |
| 3 | 50 |
| 4 | 32 |
| 5 | 15 |
| 6 | 15 |
| 7 | 9 |
| 8 | 9 |
| 9 | 6 |
| 10 | 4 |
| 11 | 3 |
| 12 | 2 |
| 14 | 2 |
| 15 | 1 |
| 16 | 3 |
| 19 | 1 |
| 20+ | 12 |

## Relation shares

| relation | triples | share |
|---|---|---|
| hasBiologicalParent | 367 | 42.6% |
| speaksTo | 259 | 30.1% |
| slainBy | 107 | 12.4% |
| hasSpouse | 52 | 6.0% |
| rules | 39 | 4.5% |
| cursedBy | 20 | 2.3% |
| instructedBy | 17 | 2.0% |
