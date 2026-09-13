# Phase 6 entry gate

Checked before training, as required. No model was trained.

| condition | required | measured | result |
|---|---|---|---|
| distinct asserted triples | >= 10,000 | 1845 | **FAIL** |
| entities of degree >= 5 | >= 500 | 169 | **FAIL** |
| largest relation share | <= 40% | 42.6% (speaksTo) | **FAIL** |

**The gate does not pass. Nothing was trained.**

Proposed triples, held separately and not eligible: 26.
Entities appearing in the asserted graph: 688.

## Degree distribution

| degree | entities |
|---|---|
| 1 | 300 |
| 2 | 112 |
| 3 | 65 |
| 4 | 42 |
| 5 | 31 |
| 6 | 26 |
| 7 | 11 |
| 8 | 11 |
| 9 | 8 |
| 10 | 9 |
| 11 | 9 |
| 12 | 4 |
| 13 | 6 |
| 14 | 3 |
| 15 | 2 |
| 17 | 1 |
| 18 | 4 |
| 19 | 4 |
| 20+ | 40 |

## Relation shares

| relation | triples | share |
|---|---|---|
| speaksTo | 786 | 42.6% |
| foughtWith | 509 | 27.6% |
| hasBiologicalParent | 289 | 15.7% |
| slainBy | 109 | 5.9% |
| hasSpouse | 40 | 2.2% |
| rules | 38 | 2.1% |
| defeatedBy | 38 | 2.1% |
| cursedBy | 20 | 1.1% |
| instructedBy | 16 | 0.9% |
