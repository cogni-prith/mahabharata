# Decisions

The audit trail for the methods section. Every decision is dated and carries its
reasoning. Do not change a decision in code without amending it here.

Decisions D1 and D2 were taken on 2026-09-12 before the change of plan. D3
onward record the change of plan itself, also 2026-09-12.

---

## D1. Scope is all 18 parvas (2026-09-12)

Decided by the user. The alternative was the 6 parvas that are fully
chapter-addressable and have a genuine human annotation pass.

**Harivamsha is excluded.** It appears in the master CSVs (5,660 sentences) and
as a PDF, but has no token-level annotation in either folder. "All 18 parvas"
means the 18 parvas of the epic; Harivamsha is the khila. Reopen if wanted.

Consequences, each traceable to a measurement in `docs/DATASET.md`, are recorded
under D4.

## D2. Substrates join on normalized sentence text, not on index (2026-09-12)

**This supersedes an earlier recommendation of mine, and it also supersedes
section 3.1 of the change-of-plan document, which adopted that recommendation.**
The index join does not work. Measurement is what showed it.

I had proposed joining the master CSVs to the BIO CSVs on
`sent_id + 1 == sentence_no`, citing 99.4 to 99.8% verified on five parvas. The
change-of-plan document adopted that proposal with the condition that the join be
verified empirically before anything was built on it. It was verified, across all
18 parvas, and it fails.

The index join scores roughly 96% on fourteen parvas and **exactly 0% on Karna,
Shalya, Mahaprasthanika and Svargarohana.** The cause is that the master CSV is a
sparse subset that retained its original indices while the BIO files segmented
sentences independently. Karna has 3,843 master sentences with `sent_id` values
running to 7,368, against 7,350 BIO sentences. No fixed offset repairs it; a
brute-force search over offsets -5 to 60 found none. My original five-parva
sample happened to exclude all four failures.

The failure mode is the dangerous one. The four parvas do not arrive as errors.
They arrive as absent rows, which is exactly the silent drop the conventions
warn about.

**The key is the sentence text, normalized** as `preprocess_pdf.py` normalizes
it: possessives removed, all punctuation replaced with whitespace, digit-only
tokens dropped, whitespace collapsed, lowercased. Measured coverage:

| | matched | total | rate |
|---|---|---|---|
| Master to LLM BIO, 18 parvas | 58,121 | 61,141 | 95.1% |
| Master to human BIO, 8 parvas | 13,215 | 13,807 | 95.7% |

No parva falls below 89.3%. Per-parva figures are in section D2a.

Two residuals are Phase 1 gate items rather than rounding error: 3,020 unjoined
master sentences, not yet explained, and 314 duplicate normalized keys
concentrated in Shanti (79) and Anushasana (52), which need a tie-break rule.

**Standing rule derived from this:** never validate a join on this corpus against
a sample of parvas. Run all 18 and print per-parva rates.

### D2a. Per-parva join rates on the normalized-text key

| Parva | Master sents | LLM match | Human match |
|---|---|---|---|
| 01 Adi | 5,758 | 90.5% | n/a |
| 02 Sabha | 1,922 | 89.3% | n/a |
| 03 Aranyaka | 8,873 | 94.6% | n/a |
| 04 Virata | 1,541 | 93.4% | n/a |
| 05 Udyoga | 5,514 | 95.0% | n/a |
| 06 Bhishma | 4,751 | 96.7% | 96.7% |
| 07 Drona | 7,748 | 96.0% | n/a |
| 08 Karna | 3,843 | 94.9% | 94.9% |
| 09 Shalya | 3,076 | 95.5% | 95.5% |
| 10 Souptika | 581 | 93.3% | 93.3% |
| 11 Stri | 574 | 94.6% | n/a |
| 12 Shanti | 8,192 | 96.1% | n/a |
| 13 Anushasana | 5,040 | 97.5% | n/a |
| 14 Ashvamedhika | 2,172 | 98.2% | n/a |
| 15 Ashramavasika | 938 | 97.3% | 97.3% |
| 16 Mausala | 287 | 96.2% | 96.2% |
| 17 Mahaprasthanika | 127 | 93.7% | 93.7% |
| 18 Svargarohana | 204 | 92.6% | 92.6% |

The human match rate equalling the LLM rate exactly on all eight human parvas is
not a coincidence and is not a bug in the measurement. It is D6 showing through:
the human files are near-copies of the LLM files, so they join identically.

### D2b. Division of labour between substrates

Adopted as proposed in section 3.1, with only the join key changed.

- **Master CSVs** carry the citation spine and the speech layer. They retain
  punctuation and nested curly quotes; 4,687 sentences show two-level nesting and
  all frame narrators are present.
- **BIO CSVs** carry token spans, human where they exist, LLM otherwise.
- **Every mention carries a `tier` field** recording its source substrate, so
  later analysis can filter to human-touched records without re-deriving
  provenance. Required by section 3.1.

---

## D3. The deliverable is an embedding, not the symbolic graph (2026-09-12)

The supervisors' requirement is to represent the epic mathematically, such that
querying shows the visual embedding and the connections together. The symbolic
graph is demoted to training data.

"Mathematically" is read literally. A force-directed layout is a drawing
algorithm with no semantics. An embedding is a trained model in which distance
carries meaning and vector arithmetic is defined. Presenting the former as the
latter would misrepresent the work, so it is an explicit anti-goal.

Three query modes define done: nearest neighbours in vector space, link
prediction over incomplete triples, and analogy by vector arithmetic.

The target scientific result is the generalised form of the Karna observation.
Karna's vector should sit near the Pandava cluster although every symbolic edge
places him with the Kauravas. Rather than handpicking that example, Phase 8
computes per-entity overlap between top-k graph neighbours and top-k embedding
neighbours and ranks by lowest overlap. The ranked list is the contribution,
because it is produced by a method.

## D4. Citation unit is `(parvan, adhyaya)`; no Sanskrit will be acquired (2026-09-12)

Shloka-level citation is impossible from this corpus. No shloka number is printed
inline anywhere in the source. Front matter gives per-chapter shloka counts
(66,302 parsed), but a count does not locate a verse. This is a corpus
limitation, not a pipeline gap, and more extraction work does not close it.

Acquiring a shloka-aligned Sanskrit text was considered and declined. Most
scholarship cites at chapter level in prose anyway.

Chapter granularity is uneven:

| Parvas | Body chapter heads recovered |
|---|---|
| 6 through 18 | 1,054 of 1,057 |
| Udyoga | 28 of 197 |
| Adi, Sabha, Aranyaka, Virata | 0 of 654 |
| Harivamsha | 118 of 118, but out of scope per D1 |

Where no chapter head exists the passage anchors at `(parvan)` with `adhyaya`
null and `cited: false` at chapter level. **The null stays null.** Not
interpolated from position, not estimated from shloka counts, not carried from a
neighbouring passage. This follows directly from the rule against inventing
citations: an interpolated chapter number is indistinguishable downstream from a
real one.

Anchors key on `(parva, ordinal)`, not on the printed global chapter number.
Printed numbers contain typos, confirmed in Shalya, which prints
`Chapter 1126(7)` where the sequence requires 1226. The printed number is
retained as an observed attribute.

## D5. Entity resolution is construction (2026-09-12)

The corpus `canonical_id` field is a context-free find-and-replace dictionary,
not entity resolution. 90.8% of values are verbatim echoes of the surface form
and only 19 entities have more than one surface form.

It produces exactly the errors the hard rules forbid: Krishna Dvaipayana
collapsed to Krishna in 72 rows, Bharadvaja to Drona in 295, Vasudeva to Krishna
in 150, Kuntibhoja to Kunti in 45. Treated as a weak prior, never as truth.

The Bharadvaja and Krishna Dvaipayana cases are not one-off bugs. They are
automated canonicalisation collapsing a patronymic epithet onto its bearer. That
failure mode is quantified across the corpus in Phase 2 and written up, because
it is a publishable result and costs almost nothing to measure while the phase is
already running.

## D6. Phase 5 is redesigned because the human annotation is contaminated (2026-09-12)

The most consequential finding of the inventory. The human pass changed 4,780 of
382,001 tokens, 1.25%. Mahaprasthanika is byte-identical to the LLM file and
Mausala differs by one token, so two of the eight "human" parvas are unrevised
machine output wearing a human filename.

Consequently the human annotation is not independent of the LLM annotation.
Verifying LLM-proposed triples against it would measure a model against a
lightly edited copy of its own output, and any number produced that way is
meaningless. Phase 5 as originally written is invalid, not merely weak.

Two-part fix:

**The annotation is demoted from evidence to index.** It is used only to retrieve
candidate sentences. Verification happens against Debroy's prose, judging the
proposition against the text rather than against the labels. The prose is not
contaminated; the labels are.

**An uncontaminated calibration set is built.** 300 to 500 sentences
re-annotated from scratch, blind to the LLM output, sampled across parvas, with
the sampling procedure written down before sampling. That set, and only that set,
reports agreement and calibrates the verifier.

No number computed against the existing human files is reported as an
independent evaluation. Where such a number is useful for another reason it is
labelled silver-vs-silver.

Related: the `gold_hand` tier in the master CSV is mislabelled and is not used
for any score. Its parvas are Sabha, Virata and Stri, none of which has a human
file, and its note rate (4.9%) matches the train slice (5.5%), so no extra human
pass occurred.

## D7. Phase 4 uses English dependency parsing; the Sanskrit lexicon is dead (2026-09-12)

There is no Sanskrit in the corpus. Zero Devanagari and zero IAST, in the CSVs
and in the PDFs. The planned Sanskrit verb-to-predicate lexicon cannot be built.

This is not a loss. Sanskrit dependency parsing is thin and would have been the
hardest part of the project. Stanza or spaCy on the Debroy prose does the same
job with mature tooling.

**Translator bias is a stated limitation, recorded here so it reaches the paper
rather than being buried.** Every relation in this graph is mediated by one
translator's word choices. Where Debroy renders a verb as "slew" against
"struck down" against "brought down", the predicate assignment inherits that
choice. The graph describes Debroy's Mahabharata, not the Sanskrit text, and
claims about the epic's relational structure carry that qualification.

## D8. Phase 6 (critical apparatus / recension) is cut, not deferred (2026-09-12)

No variant data exists anywhere in the corpus. The phase was originally
conditional on apparatus data being present. It is not, so the conditional is
removed from the plan rather than left dangling. The phase number is reused by
the embedding work.

## D9. RotatE, not TransE (2026-09-12)

The predicate set contains symmetric relations (`isAlliedWith`, `isEnemyOf`,
`hasSpouse`), antisymmetric ones (`slainBy`, `hasBiologicalParent`) and
composition (parent applied twice is grandparent).

TransE cannot represent symmetry at all. Scoring `h + r ~ t` and `t + r ~ h`
simultaneously forces `r` toward zero, which collapses every ally onto their
ally, destroying precisely the structure this project is trying to show. RotatE
represents symmetry, antisymmetry, inversion and composition. ComplEx is the
fallback. PyKEEN supplies model and evaluation protocol together.

A hard entry gate precedes training: at least 10,000 triples, at least 500
entities of degree 5 or higher with the full degree distribution reported, and no
single relation type above 40% of all triples. Failing any of these, stop and
report rather than train something meaningless and present its numbers.

Two leakage checks are mandatory and documented: derived inverse relations
excluded from the evaluation split, and symmetric pairs deduplicated before
splitting rather than after.

## D10. The schema is derived, because there is no inherited ontology (2026-09-12)

`ontology/mahabharata.ttl` is not in the repo, nor is any seed graph, `.ttl` or
`.rq` file. Checked directly. The change-of-plan document made this a
conditional; this records which branch was taken.

The schema is therefore derived minimally from the predicates named in
`CLAUDE.md`: the kinship primitives, the identity relations, `slainBy`,
`cursedBy`, `isAlliedWith`, `isEnemyOf`, `hasSpouse`, `belongsToClan`, `causes`,
plus the `Mention`, `Passage` and `Telling` classes. If a validated OWL 2 RL
ontology with `IdentityConflationShape` and `PatronymicShape` is supplied later,
adopt it; both shapes bear directly on Phase 2.

## D11. Repository is not under version control (2026-09-12)

Confirmed: not a git repository. Initialisation is the user's call and has not
been requested. If it happens, `build/` is gitignored and there is no remote.

---

## Corpus defects to carry forward

Not decisions, but facts that constrain every phase and are easy to forget.

**`shanti_parva.csv` is not confined to Shanti Parva.** It shares 1,635 long
sentences with `anushasana_tagged.csv` and ends with Mahaprasthanika text. Its
parva labels must be re-derived, not believed. At 18-parva scope this is live
rather than theoretical.

**Eight parvas have a human pass, not the twelve the original brief claimed.**
Bhishma, Karna, Shalya, Souptika, Ashramavasika, Mausala, Mahaprasthanika,
Svargarohana. Of those, Mahaprasthanika and Mausala are effectively unrevised
(D6), leaving six with a genuine pass.
