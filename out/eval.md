# Phase 6 entry gate

Checked before training, as required. No model was trained.

| condition | required | measured | result |
|---|---|---|---|
| distinct asserted triples | >= 10,000 | 1444 | **FAIL** |
| entities of degree >= 5 | >= 500 | 137 | **FAIL** |
| largest relation share | <= 40% | 50.6% (speaksTo) | **FAIL** |

**The gate does not pass. Nothing was trained.**

Proposed triples, held separately and not eligible: 26.
Entities appearing in the asserted graph: 561.

## Degree distribution

| degree | entities |
|---|---|
| 1 | 256 |
| 2 | 93 |
| 3 | 45 |
| 4 | 30 |
| 5 | 22 |
| 6 | 23 |
| 7 | 12 |
| 8 | 12 |
| 9 | 7 |
| 10 | 8 |
| 11 | 3 |
| 12 | 3 |
| 13 | 4 |
| 14 | 2 |
| 15 | 3 |
| 16 | 3 |
| 17 | 1 |
| 18 | 3 |
| 19 | 3 |
| 20+ | 28 |

## Relation shares

| relation | triples | share |
|---|---|---|
| speaksTo | 730 | 50.6% |
| foughtWith | 473 | 32.8% |
| slainBy | 104 | 7.2% |
| rules | 37 | 2.6% |
| defeatedBy | 37 | 2.6% |
| hasSpouse | 21 | 1.5% |
| cursedBy | 18 | 1.2% |
| instructedBy | 13 | 0.9% |
| hasDivineProgenitor | 11 | 0.8% |

---

# Phase 6 training run

**The entry gate above does not pass and this model was trained anyway,
on instruction. Every number in this section describes how a model fits
1369 triples. None of them is a result about the Mahabharata.** The plan
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
| train / valid / test | 1095 / 137 / 137 |
| entities / relations | 561 / 9 |
| device | cuda:0 |
| determinism | `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; two runs at this seed are bit-identical |

## Leakage checks

- Symmetric: `foughtWith` pairs ordered canonically and deduplicated before the split. 75 of 1444 distinct triples removed as the same fact written both ways.
- Inverse: no inverse relation is materialised in this graph and `create_inverse_triples` is off. Predicate pairs sharing an entity pair in opposite order, measured rather than assumed:

| predicate a | predicate b | shared reversed pairs |
|---|---|---|
| speaksTo | speaksTo | 198 |
| foughtWith | speaksTo | 28 |
| foughtWith | slainBy | 19 |
| defeatedBy | foughtWith | 7 |
| slainBy | speaksTo | 5 |
| defeatedBy | speaksTo | 4 |
| instructedBy | speaksTo | 4 |
| cursedBy | speaksTo | 3 |
| defeatedBy | slainBy | 2 |
| slainBy | slainBy | 2 |
| cursedBy | foughtWith | 1 |
| foughtWith | instructedBy | 1 |

## Filtered ranking, test set

| metric | head | tail | both |
|---|---|---|---|
| MRR | 0.1953 | 0.1546 | 0.1750 |
| Hits@1 | 0.0876 | 0.0730 | 0.0803 |
| Hits@3 | 0.2263 | 0.1533 | 0.1898 |
| Hits@10 | 0.4161 | 0.2993 | 0.3577 |

Filtered ranking, realistic rank definition, as PyKEEN reports it.
PyKEEN 1.11.1.

