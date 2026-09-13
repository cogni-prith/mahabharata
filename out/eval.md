# Phase 6 entry gate

Checked before training, as required. No model was trained.

| condition | required | measured | result |
|---|---|---|---|
| distinct asserted triples | >= 10,000 | 1771 | **FAIL** |
| entities of degree >= 5 | >= 500 | 162 | **FAIL** |
| largest relation share | <= 40% | 42.2% (speaksTo) | **FAIL** |

**The gate does not pass. Nothing was trained.**

Proposed triples, held separately and not eligible: 26.
Entities appearing in the asserted graph: 683.

## Degree distribution

| degree | entities |
|---|---|
| 1 | 302 |
| 2 | 116 |
| 3 | 60 |
| 4 | 43 |
| 5 | 30 |
| 6 | 24 |
| 7 | 12 |
| 8 | 10 |
| 9 | 6 |
| 10 | 10 |
| 11 | 9 |
| 12 | 5 |
| 13 | 2 |
| 14 | 5 |
| 16 | 1 |
| 17 | 1 |
| 18 | 6 |
| 19 | 5 |
| 20+ | 36 |

## Relation shares

| relation | triples | share |
|---|---|---|
| speaksTo | 748 | 42.2% |
| foughtWith | 483 | 27.3% |
| hasBiologicalParent | 286 | 16.1% |
| slainBy | 104 | 5.9% |
| hasSpouse | 40 | 2.3% |
| rules | 38 | 2.1% |
| defeatedBy | 37 | 2.1% |
| cursedBy | 19 | 1.1% |
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
| train / valid / test | 1355 / 169 / 170 |
| entities / relations | 683 / 9 |

## Leakage checks

- Symmetric: `foughtWith` pairs ordered canonically and deduplicated before the split. 77 of 1771 distinct triples removed as the same fact written both ways.
- Inverse: no inverse relation is materialised in this graph and `create_inverse_triples` is off. Predicate pairs sharing an entity pair in opposite order, measured rather than assumed:

| predicate a | predicate b | shared reversed pairs |
|---|---|---|
| speaksTo | speaksTo | 208 |
| foughtWith | speaksTo | 28 |
| hasBiologicalParent | speaksTo | 20 |
| foughtWith | slainBy | 19 |
| foughtWith | hasBiologicalParent | 8 |
| defeatedBy | foughtWith | 7 |
| hasBiologicalParent | hasBiologicalParent | 6 |
| hasSpouse | speaksTo | 6 |
| instructedBy | speaksTo | 5 |
| slainBy | speaksTo | 5 |
| defeatedBy | speaksTo | 4 |
| hasSpouse | hasSpouse | 4 |

## Filtered ranking, test set

| metric | head | tail | both |
|---|---|---|---|
| MRR | 0.1622 | 0.1283 | 0.1453 |
| Hits@1 | 0.0647 | 0.0529 | 0.0588 |
| Hits@3 | 0.1529 | 0.1118 | 0.1324 |
| Hits@10 | 0.3765 | 0.2941 | 0.3353 |

Filtered ranking, realistic rank definition, as PyKEEN reports it.
PyKEEN 1.11.1.

