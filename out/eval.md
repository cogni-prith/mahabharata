# Phase 6 entry gate

Checked before training, as required. No model was trained.

| condition | required | measured | result |
|---|---|---|---|
| distinct asserted triples | >= 10,000 | 1100 | **FAIL** |
| entities of degree >= 5 | >= 500 | 104 | **FAIL** |
| largest relation share | <= 40% | 55.7% (speaksTo) | **FAIL** |

**The gate does not pass. Nothing was trained.**

Proposed triples, held separately and not eligible: 26.
Entities appearing in the asserted graph: 570.

## Degree distribution

| degree | entities |
|---|---|
| 1 | 292 |
| 2 | 104 |
| 3 | 40 |
| 4 | 30 |
| 5 | 21 |
| 6 | 16 |
| 7 | 10 |
| 8 | 8 |
| 9 | 6 |
| 10 | 2 |
| 11 | 7 |
| 12 | 4 |
| 13 | 3 |
| 14 | 1 |
| 15 | 3 |
| 16 | 1 |
| 17 | 1 |
| 18 | 3 |
| 19 | 1 |
| 20+ | 17 |

## Relation shares

| relation | triples | share |
|---|---|---|
| speaksTo | 613 | 55.7% |
| bornTo | 172 | 15.6% |
| foughtWith | 171 | 15.5% |
| slainBy | 40 | 3.6% |
| rules | 32 | 2.9% |
| hasSpouse | 21 | 1.9% |
| cursedBy | 17 | 1.5% |
| instructedBy | 14 | 1.3% |
| defeatedBy | 11 | 1.0% |
| hasDivineProgenitor | 9 | 0.8% |

---

# Phase 6 training run

**The entry gate above does not pass and this model was trained anyway,
on instruction. Every number in this section describes how a model fits
1081 triples. None of them is a result about the Mahabharata.** The plan
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
| train / valid / test | 864 / 108 / 109 |
| entities / relations | 570 / 10 |
| device | cpu |
| determinism | `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; two runs at this seed are bit-identical |

## Leakage checks

- Symmetric: `foughtWith` pairs ordered canonically and deduplicated before the split. 19 of 1100 distinct triples removed as the same fact written both ways.
- Inverse: no inverse relation is materialised in this graph and `create_inverse_triples` is off. Predicate pairs sharing an entity pair in opposite order, measured rather than assumed:

| predicate a | predicate b | shared reversed pairs |
|---|---|---|
| speaksTo | speaksTo | 200 |
| foughtWith | speaksTo | 16 |
| foughtWith | slainBy | 12 |
| bornTo | speaksTo | 10 |
| instructedBy | speaksTo | 5 |
| bornTo | bornTo | 4 |
| defeatedBy | foughtWith | 4 |
| slainBy | speaksTo | 4 |
| cursedBy | speaksTo | 3 |
| slainBy | slainBy | 2 |
| bornTo | foughtWith | 1 |
| bornTo | slainBy | 1 |

## Filtered ranking, test set

| metric | head | tail | both |
|---|---|---|---|
| MRR | 0.1906 | 0.1894 | 0.1900 |
| Hits@1 | 0.0642 | 0.0826 | 0.0734 |
| Hits@3 | 0.2844 | 0.2385 | 0.2615 |
| Hits@10 | 0.4037 | 0.4037 | 0.4037 |

Filtered ranking, realistic rank definition, as PyKEEN reports it.
PyKEEN 1.11.1.

