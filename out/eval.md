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

---

# Phase 6 training run

**The entry gate above does not pass and this model was trained anyway,
on instruction. Every number in this section describes how a model fits
1,845 triples. None of them is a result about the Mahabharata.** The plan
puts the floor for meaningful vectors at roughly 5,000 triples.

| setting | value |
|---|---|
| model | RotatE |
| embedding dim | 200 complex, stored as 400 float32 |
| random seed | 20260917 |
| epochs | 400 |
| optimiser | Adam, lr 1e-3, batch 512 |
| negative sampling | basic, uniform corruption of head or tail |
| negatives per positive | 32 |
| split | 80/10/10 by triple, seed 20260917 |
| train / valid / test | 1408 / 176 / 177 |
| entities / relations | 688 / 9 |

## Leakage checks

- Symmetric: `foughtWith` pairs ordered canonically and deduplicated before the split. 84 of 1845 distinct triples removed as the same fact written both ways.
- Inverse: no inverse relation is materialised in this graph and `create_inverse_triples` is off. Predicate pairs sharing an entity pair in opposite order, measured rather than assumed:

| predicate a | predicate b | shared reversed pairs |
|---|---|---|
| speaksTo | speaksTo | 224 |
| foughtWith | speaksTo | 32 |
| hasBiologicalParent | speaksTo | 25 |
| foughtWith | slainBy | 21 |
| defeatedBy | foughtWith | 8 |
| foughtWith | hasBiologicalParent | 8 |
| hasBiologicalParent | hasBiologicalParent | 6 |
| hasSpouse | speaksTo | 6 |
| instructedBy | speaksTo | 5 |
| slainBy | speaksTo | 5 |
| cursedBy | speaksTo | 4 |
| defeatedBy | speaksTo | 4 |

## Filtered ranking, test set

| metric | head | tail | both |
|---|---|---|---|
| MRR | 0.1488 | 0.1384 | 0.1436 |
| Hits@1 | 0.0452 | 0.0621 | 0.0537 |
| Hits@3 | 0.1751 | 0.1469 | 0.1610 |
| Hits@10 | 0.3277 | 0.3164 | 0.3220 |

Filtered ranking, realistic rank definition, as PyKEEN reports it.
PyKEEN 1.11.1.

