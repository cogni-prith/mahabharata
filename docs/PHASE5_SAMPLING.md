# Phase 5. The blind calibration set: sampling procedure

Written before the sample was drawn, which is the point of writing it down. The
sampler `src/phase5_blind_sample.py` implements exactly what is below and
nothing else. If the procedure changes, it changes here first and the sample is
redrawn with a new seed recorded.

## Why this set exists

The Phase 5 gate is a groundedness rate per predicate. It cannot be computed
against the existing human files: the human pass changed 1.25% of tokens
against the LLM output it was edited from, so an agreement number between them
measures a model against a lightly edited copy of itself. Any number computed
that way is labelled silver-vs-silver and is not this.

This set is annotated from scratch, blind to the LLM output, by a human. It is
the only set any reported agreement or groundedness number may be computed
against.

## Population

Sentences from `mahabharata_ner_by_sentence_master (2).csv`, all 18 parvas.
A sentence is eligible when all of the following hold:

1. It resolves to a chapter, meaning `adhyaya_ordinal` is not null. A sentence
   with no chapter anchor cannot be cited at `(parvan, adhyaya)` and is not
   useful as calibration for a citation-anchored graph.
2. It is at least 60 characters long. Shorter sentences in this translation are
   vocatives and interjections, not propositions.
3. It carries at least two distinct PERSON surfaces in the annotation layer.

Condition 3 uses the annotation, which the plan permits: the annotation is
allowed as a retrieval index and forbidden as evidence. The set is blind in the
sense that matters, which is that the annotator never sees a label, a span, an
extracted triple or a proposal.

## Strata

Two strata, because precision and recall need different sentences and drawing
only from one gives a number that looks like an evaluation and is not.

- **Stratum A, 200 sentences.** The pipeline produced at least one triple from
  this sentence, asserted in Phase 4 or proposed in Phase 5. Reading these gives
  the groundedness rate: of what the pipeline claims, how much the prose
  supports.
- **Stratum B, 200 sentences.** The pipeline produced nothing from this
  sentence, although it meets conditions 1 to 3 and therefore looks extractable.
  Reading these gives the miss rate: of what the prose states, how much the
  pipeline never saw. Without stratum B the groundedness number describes only
  the sentences the extractor happened to like.

400 sentences total, inside the 300 to 500 the plan asks for.

## Allocation within a stratum

Proportional to each parva's share of that stratum's eligible population,
rounded down, then a floor of 5 sentences per parva so no parva is unrepresented,
then the remainder given to the largest parvas by eligible count. Drawn without
replacement with `random.Random(SEED)`.

**SEED = 20260913.** Recorded here, in the sampler, and in the key file.

## What the annotator sees

`build/blind_calibration_sheet.csv`, one row per sentence, in an order shuffled
across both strata so the stratum is not inferable from position. Columns:

| column | meaning |
|---|---|
| `id` | opaque row id, matches the key file |
| `parva`, `adhyaya` | the citation, so the passage can be looked up |
| `sentence` | the text, verbatim, punctuation and nested quotes intact |
| `triples` | to fill in: one per line, `Subject \| predicate \| Object` |
| `none` | to fill in: `y` if the sentence states no relation in the inventory |
| `unclear` | to fill in: `y` if the referent or the relation is genuinely undecidable |
| `notes` | to fill in: free text, especially for anything contested |

No label, no span, no extracted triple and no stratum appears in the sheet.
`build/blind_calibration_key.csv` holds the id, the stratum and the pipeline's
own output for each sentence. **Do not open the key file before annotating.**

## The predicate inventory to annotate with

Fourteen, the recorded set. Nothing outside this list; if the sentence states
something the list cannot express, write it in `notes` rather than inventing a
predicate.

| predicate | direction and meaning |
|---|---|
| `hasBiologicalParent` | `Child \| hasBiologicalParent \| Parent` |
| `hasLegalParent` | the socially recognised parent where the text separates them |
| `hasFosterParent` | the raising parent, Radha and Adhiratha for Karna |
| `hasDivineProgenitor` | the god who fathered, Surya for Karna |
| `hasSpouse` | either order, the text rarely marks one |
| `slainBy` | `Victim \| slainBy \| Killer`, only where the sentence says the killing happened |
| `defeatedBy` | `Loser \| defeatedBy \| Winner`, defeat short of death |
| `foughtWith` | symmetric, a fight took place |
| `speaksTo` | `Speaker \| speaksTo \| Addressee` |
| `cursedBy` | `Cursed \| cursedBy \| Curser` |
| `instructedBy` | `Student \| instructedBy \| Teacher` |
| `rules` | `Ruler \| rules \| Place` |
| `isAmshaOf` | a portion or incarnation of a divinity |
| `rebornAs` | a rebirth, not an identity |

## Rules for the annotator

These mirror the project's hard rules and exist so the calibration set is not
contaminated by the same faults it is meant to measure.

1. **Annotate what this sentence states**, not what the epic elsewhere
   establishes. "Arjuna wielded Gandiva" is not evidence that Kunti is his
   mother even though you know she is.
2. **A future, conditional, hypothetical or questioned event is not an
   assertion.** "I will kill Jayadratha tomorrow" states no slaying.
3. **A simile is not an event.** "Slay him as Vritra was destroyed by Indra"
   states nothing about Vritra in this sentence.
4. **Quoted speech still counts**, and the speaker is not the narrator. Record
   the relation the speaker asserts, and note in `notes` if you think the
   speaker is unreliable.
5. **Where the referent is a description rather than a name**, for example
   "Duhshasana's son", write the description in the slot and put `y` in
   `unclear`. Do not resolve it to a name from memory.
6. **Where the epic itself does not settle it**, for example who slew
   Abhimanyu, leave it out and say so in `notes`. A gap is a result.
7. Use the surface name as written in this translation, `Droupadi` and not
   `Draupadi`.

## After the sheet comes back

`src/phase5_calibrate.py` does not exist yet and is deliberately not written
before there are annotations for it to read. It will compute, per predicate:
groundedness on stratum A, the miss rate on stratum B, and the agreement between
the annotator and the pipeline with the strata kept separate and both reported.
That number is the Phase 5 gate, and it is the only independent evaluation this
project can produce.
