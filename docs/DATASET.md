# Dataset inventory

Written before any extraction code, as the build plan requires. Everything below
was read from the files. Where a question cannot be answered from the data, it
says so.

Status: awaiting review. No extraction code has been written.

## Correction to the orientation in CLAUDE.md

Three statements in CLAUDE.md do not match the repository. They are load-bearing,
so they are corrected here rather than worked around.

| CLAUDE.md says | The repository has |
|---|---|
| Input is in `data/` | No `data/` directory. Corpus is in two Drive-export folders at the repo root. |
| 12 parvas with human annotation | 8 parvas have a human file. 6 of those carry a real revision pass. |
| Prior work in `ontology/`, `seed/`, `tools/` | None of these exist. No ontology, no seed graph, no viewers, no SHACL shapes. |

The repo is also not a git repository, so nothing here can be committed yet.

Actual corpus locations, both containing one same-named subfolder:

- `Mahabharat_Human_Annotated_Data-20260801T163709Z-1-001/`, 8 CSVs
- `Mahabharata LLM annotated-20260801T163642Z-1-001/`, 18 CSVs

Supporting material: `parva_pdfs/` (20 PDFs), `TheMahabharata.pdf` (5,631 pages),
and two consolidated master CSVs at the repo root.

## The two substrates

There are two separate annotation pipelines in this repo, not one. They disagree
on label space, sentence segmentation and provenance. Choosing between them is
the first real design decision.

**A. Token-level BIO CSVs** (26 files, the two folders above)

Schema is `sentence_no, sentence, word, [POS|pos], tag`, one row per token.
Column presence is inconsistent: 3 human and 3 LLM files carry a POS column,
`Savargarohana_Parva_filled.csv` uses lowercase `pos`, and `KarnaParvaRe.csv`
has a nameless trailing fifth column. The human pass deleted the POS column
from two files relative to their LLM versions.

All punctuation except the period was stripped, and every digit-only token was
dropped, by `preprocess_pdf.py`. This destroyed quotation marks, chapter numbers
and shloka numbers.

**B. Master CSVs** (`mahabharata_ner_by_sentence_master (2).csv`,
`mahabharata_ner_entities_master (2).csv`)

A different pipeline: spaCy prediction plus rule corrections. Retains
punctuation, curly quotation marks, apostrophes and digits. Covers 19 parva
labels including Harivamsha. Carries `split`, `annotation_tier` and a
`canonical_id` column that the BIO CSVs lack.

Substrate B is the only route to a citation spine or to speech nesting.
Substrate A is the only place human-checked spans exist.

## The questions

### What is the annotation unit?

Token. One row per word, with BIO tags, grouped under a `sentence_no` and a
verbatim copy of the whole sentence. Not span-level and not shloka-level.

In the master CSVs the unit is the sentence, with entities as a JSON array.

### What entity types exist, and what is the label inventory?

Substrate A has 9 types:

| Type | B- (human) | B- (LLM) |
|---|---|---|
| PERSON | 16,633 | 71,383 |
| DYNASTY | 3,696 | 11,885 |
| NORP | 2,566 | 16,543 |
| GPE | 1,763 | 5,890 |
| CONCEPT | 1,297 | 11,690 |
| WEAPON | 612 | 1,877 |
| SACRED_TEXT | 174 | 2,102 |
| EVENT | 170 | 706 |
| CELESTIAL | 22 | 0 |

Three defects in the tag column:

- Outside is encoded two ways, `''` (empty) and `O(Clear Entity)`. Any loader
  must normalise both.
- Six malformed `O-<TYPE>` tags exist, five introduced by the human pass.
  Locations: human Bhisma line 117, human Mausala line 2194, human Souptika
  lines 8321, 16711, 18722, and one in LLM Mausala.
- CELESTIAL exists in exactly one file, human Bhishma, as 22 B- and 3 I- tags.
  It is one annotator's local decision, not a corpus-wide class.

Substrate B uses a different label space. It adds LOC (1,984 rows) and
NOT_AN_ENTITY (4,163 rows), and has no CELESTIAL. The two substrates cannot be
merged without an explicit label mapping decision.

### Do mentions carry canonical entity IDs, or only spans and types?

**Only spans and types.** This answers open question 1.

The 26 BIO CSVs have no ID column at all. The master entities CSV has a
`canonical_id` column, but it is a context-free find-and-replace dictionary,
not entity resolution:

- `canonical_id` equals `entity_text` verbatim in 112,527 of 123,994 rows (90.8%).
- 7,798 distinct canonical IDs against 7,849 distinct surface forms, a 0.6% collapse.
- Only **19** canonical entities have more than one surface form. The alias table
  is about 42 genuine epithet pairs, plus 9 case and possessive normalisations.
- No surface form ever receives two different canonical IDs. The mapping cannot
  be context-sensitive by construction.
- 1,585 rows carry a corrupted ID with a glued shloka number or punctuation,
  such as `Krishna!1129`, `Achyuta;951`, `Sunrita,30`.

Because it is context-free, it makes exactly the errors CLAUDE.md's "never guess
an entity link" rule names:

| Mapping | Rows | Why it is wrong |
|---|---|---|
| `Krishna` in "Krishna Dvaipayana" contexts | 72 | The referent is Vyasa, not Vasudeva Krishna |
| `Vasudeva` to `Krishna` in Harivamsha | 150 | Vasudeva there is Anakadundubhi, Krishna's father |
| `Bharadvaja` to `Drona` | 295 | Bharadvaja is a rishi, Drona's father |
| `Kuntibhoja` to `Kunti` | 45 | Kuntibhoja is Kunti's foster father |

**Phase 2 is construction, not validation.** This is where the annotation budget
should go, as the build plan anticipated.

The `notes` column is populated on 6,723 rows (5.4%) with 27 distinct canned
strings. Several explicitly flag unresolved ambiguity, for example
`REVIEW: ambiguous concept-vs-entity (Maya=illusion/demon, Kala=time/Death)`
on 1,089 rows and `Bharata: lineage vs vocative (rule)` on 3,168. These are a
useful starting list of known-hard cases.

### Are character offsets present?

No. The `entities` JSON carries exactly three keys, `text`, `label`,
`canonical_id`. A scan of all 66,802 rows found no `start`, `end` or `offset`
key anywhere.

Offsets are recoverable but not for free. On a 9,367-instance sample, 95.8%
matched at exactly one position, 4.2% matched at several positions, and none
failed to match. The ambiguous fraction needs occurrence-order pairing, and
that pairing is unreliable because entity row order disagrees with `n_entities`
on 2,681 sentences. Budget this as real Phase 1 work and report the ambiguous
fraction rather than silently taking the first match.

### Is the CE address present, or does it need reconstructing?

**Not present on any record, in either substrate.** It must be reconstructed
from the PDFs. The achievable granularity is not uniform.

The source prints chapter heads as `CHAPTER <global>(<parva-relative>)`, plus a
per-section front-matter list of the form `Chapter 1220(1): 52 shlokas`. Body
heads were measured across all 20 PDFs by separating the two forms:

| Parvas | Declared chapters | Body heads found | Spine |
|---|---|---|---|
| 6 Bhishma to 18 Svargarohana | 1,057 | 1,054 | complete |
| Harivamsha (bare `Chapter N` form) | 118 | 118 | complete |
| 5 Udyoga | 197 | 28 | partial, 14% |
| 1 Adi, 2 Sabha, 3 Aranyaka, 4 Virata | 654 | 0 | none |

Adi Parva was checked specifically. All 220 of its "Chapter" occurrences are
front-matter list entries. Debroy's first volume does not print chapter heads in
the body at all, so there is nothing to key on. The bare numerals that look like
chapter numbers are footnote markers, at body font size, incrementing
continuously across pages.

The decisive fact: **all 8 human-annotated parvas (6, 8, 9, 10, 15, 16, 17, 18)
fall in the complete range.** The subset with human-checked spans is exactly the
subset where the chapter spine reconstructs cleanly.

Two extraction hazards to carry forward. Chapter heads are uppercase `CHAPTER`
in the body and mixed case in the front-matter list, so matching must be
case-insensitive and must exclude the `: N shlokas` form. And `get_text()`
silently drops function words on several of these PDFs, so extraction must use
`get_text("words")` as `preprocess_pdf.py` did.

Printed global chapter numbers contain typos and cannot be trusted as keys. One
confirmed case: Shalya's front-matter reads `Chapter 1126(7)` between 1225(6)
and 1227(8), where 1226 is meant. Address by (parva, ordinal position within
parva) instead.

### Are shloka numbers available?

**No.** Individual shloka numbers are printed nowhere in the body text. The
translation is continuous prose, not numbered verse.

Per-chapter shloka **counts** are declared in each section's front-matter, and
66,302 were parsed. That supports a statement like "adhyaya 1 contains 52
shlokas" but gives no way to place a given English sentence on a specific
shloka.

So the citation ceiling is (parvan, adhyaya). Under the "never invent a citation"
rule, shloka must be left null rather than interpolated from sentence position
against declared counts. Phase 1's gate as written, "every mention resolves to a
valid CE address", cannot be met at shloka level and should be restated at
chapter level.

### Which parvas are human-annotated, and to what depth?

8 parvas have a human file. The depth is thin: a light correction pass over LLM
output, not independent annotation.

| # | Parva | Tokens | Tokens changed | Rate |
|---|---|---|---|---|
| 6 | Bhishma | 133,365 | 624 | 0.47% |
| 8 | Karna | 104,969 | 2,048 | 1.95% |
| 9 | Shalya | 81,841 | 1,416 | 1.73% |
| 10 | Souptika | 20,382 | 325 | 1.59% |
| 15 | Ashramavasika | 26,812 | 299 | 1.12% |
| 18 | Svargarohana | 4,995 | 67 | 1.34% |
| 16 | Mausala | 6,924 | 1 | 0.01% |
| 17 | Mahaprasthanika | 2,713 | 0 | 0.00% |

Total: 4,780 of 382,001 tokens changed, 1.25%.

Mahaprasthanika is byte-identical to the LLM file (same md5). Mausala differs by
one token. Those two are unrevised LLM output in a folder labelled human.
**Six parvas carry a real human pass.**

The edits are overwhelmingly add or remove of entity status on the same token
rather than type correction. Across the set, `B-PERSON` to `O` is 898 and `O` to
`B-PERSON` is 796. Bhishma is a different kind of pass: 417 of its 624 edits are
the single pattern `O` to `B-GPE`, a targeted geography sweep. A single "human
precision" figure across the 8 files would be misleading.

Humans also repaired text the extraction had eaten, restoring dropped leading
capitals such as `anjaya` to `Sanjaya` and `aishampayana` to `Vaishampayana`,
61 fixes in Karna and 54 in Shalya. These were not back-ported to the LLM files,
which is why those two pairs are not perfectly token-aligned.

**This weakens the project's central premise and needs a decision.** The build
plan says "models have memorized the text, so they are good candidate
generators; the annotated corpus is the verifier". But the verifier is the same
LLM's output with 1.25% of tokens edited. Verifying LLM-proposed triples against
it is closer to self-agreement than to grounding. See Risks below.

Harivamsha has no token-level annotation on either side. It appears only in the
master CSVs (5,660 sentences) and as a PDF.

### Is there a gold / silver / dev / test split?

Two schemes exist and they contradict each other.

**Master CSVs:** train 61,565 / dev 1,200 / test 4,037 sentences. `split` is a
perfect 1:1 function of `annotation_tier`: test maps to `gold_hand`, train and
dev to `silver_rule`. The split is whole-parva.

The problem: the `gold_hand` parvas are **02_Sabha, 04_Virata and 11_Stri**,
none of which has a human-annotated file anywhere on disk. Their note rate
(4.9%) is indistinguishable from the train slice (5.5%), so there is no evidence
of an extra human pass. **This tier is mislabelled. Using it would validate the
KG against LLM output labelled as human.**

**train_delta:** holds out the 8 human parvas. This is the defensible gold set:
27,181 sentences and 27,016 spans, or roughly 26,213 after dropping
Mahaprasthanika and Mausala. `train_delta/predictions/*.csv` carries `llm_tag`,
`pred_tag` and `human_tag` side by side per token, which is more convenient than
the DocBins. Do not use `train_spacy_lg` for anything gold-facing: four of its
nine training parvas are gold test parvas.

Recommendation: ignore the master `annotation_tier` column entirely and define
the validation set as the 6 genuinely revised parvas.

### Is the source Sanskrit, a translation, or both?

**English translation only.** Bibek Debroy's translation of the BORI critical
edition.

Searched for Devanagari (U+0900 to U+097F) and IAST diacritics across every CSV,
`.py`, `.tex` and `.json` in the repo, and extracted text from
`TheMahabharata.pdf`, `parva_pdfs/06_Bhishma.pdf` and
`parva_pdfs/frontmatter.pdf`. Zero Devanagari and zero IAST in any corpus or
source file. The only hits are inside `.venv` (spaCy's own language data) and
two incidental diacritics in `2.py` and `7.py`.

No normalization pass is needed, because there is no Sanskrit to normalize.
This answers open question 2 and forces a change to Phase 4: the verb-triggered
lexicon (`han` to `slainBy`, `śap` to `cursedBy`) cannot be built. Phase 4 must
run on English verbs, with no Sanskrit to validate against. The build plan's
fallback, "shloka-aligned Debroy", is in fact the only source, and it is not
shloka-aligned.

### Is any critical apparatus or variant data included?

No. No starred passages, no variant readings, no manuscript footnotes, no
appendix passages, and no shloka-aligned Sanskrit anywhere in the repo. The
footnotes present are the translator's explanatory notes.

**Phase 6 should be skipped cleanly.** This answers open question 4.

### Is Phase 3 (speaker attribution) feasible?

Yes, but only from substrate B. Measured on the master by_sentence CSV:

- 5,809 of 66,801 sentences (8.7%) carry an explicit speech frame, 6,045 frames total.
- All frame narrators are present: Vaishampayana 878, Bhishma 516, Sanjaya 415,
  Janamejaya 109, Narada 98, Markandeya 90, Souti 69, Lomasha 54, Brihadashva 34.
- Nested curly quotes survived. 4,687 sentences show two-level `‘ ... “` nesting.

Gita spot-check, showing the nesting intact:

> ... How did Kounteya Pandava counter them?" 'Sanjaya said, "On seeing the
> soldiers of Dhritarashtra's sons arranged in battle formation ...

Two caveats. The pattern has false positives that need a gazetteer filter
("Having" 160, "He" 107). And this works only on substrate B: the BIO CSVs had
every quotation mark stripped, so the nesting signal is gone there entirely.

## Counts

| Quantity | Value |
|---|---|
| Human token rows | 382,001 |
| Human sentences | 27,180 |
| Human B- spans | 26,933 |
| LLM token rows | 1,977,334 |
| LLM sentences | 138,727 |
| LLM B- spans | 122,076 |
| Master sentences | 66,802 (66,801 non-null) |
| Master entity rows | 123,994 |
| Unique surface forms (master) | 7,849 |
| Unique canonical IDs (master) | 7,798, of which only 19 are real |
| Unique word forms, human set | 6,324 max per file (Bhishma) |
| Entity types (BIO) | 9 |
| Parvas with human file | 8 |
| Parvas genuinely revised | 6 |
| Declared chapters, parsed | 1,845 |
| Declared shlokas, parsed | 66,302 |

## Corpus integrity problems found

These need decisions before Phase 1, not after.

1. **`shanti_parva.csv` is not confined to Shanti Parva.** It shares 1,635
   distinct long sentences with `anushasana_tagged.csv`, contains an Anushasana
   end marker, and its last row is Mahaprasthanika text about Yudhishthira and
   the dog. Parva labels on that file cannot be trusted.
2. **`Savargarohana_Parva_filled.csv` starts at `sentence_no` 2** with a
   truncated first word `anamejaya`. The first sentence is missing in both folders.
3. **731 `global_sent_id` values in the entities master are absent from the
   by_sentence master** (1,039 orphaned entity rows). The by_sentence master is
   not a superset.
4. **`n_entities` disagrees with the entity row count on 2,681 sentences.** The
   shortfall of 4,163 exactly equals the NOT_AN_ENTITY count, so the two files
   were serialized at different pipeline stages.
5. **Drop-cap artifact.** The PDF sets the first letter of each chapter as a
   drop cap, which extracts as a separate token. This is why CSV sentences read
   `V Chapter aishampayana said`. Sentence alignment must handle it.
6. **`13_Anushasana.pdf` front-matter lists 250 chapters** against 154 body
   heads, because the PDF splits overlap and Ashvamedhika's list bleeds in.
   Several PDFs also report `cannot find page N in page tree`, so they are
   partially damaged.

## Risks to the project as planned

1. **The verifier is not independent of the generator.** The human layer is
   LLM output with 1.25% of tokens changed. Phase 5's groundedness rate would
   measure agreement between an LLM and a lightly-edited copy of the same LLM's
   output. Fixing this needs either fresh independent annotation on a sample, or
   an explicit reframing of what Phase 5 measures.
2. **No shloka-level citation is possible.** The ontology and the gate language
   both assume (parvan, adhyaya, shloka). The ceiling is (parvan, adhyaya).
3. **Sanskrit-dependent work is out.** Phase 4's verb lexicon and Phase 6's
   recension layer have no data to stand on.
4. **Five parvas cannot be chapter-addressed at all** (Adi, Sabha, Aranyaka,
   Virata, and 86% of Udyoga). Those are 654 chapters and include the entire
   frame story. They can carry parva-level citation only.
5. **The two substrates have different label spaces.** LOC and NOT_AN_ENTITY
   exist in one, CELESTIAL in the other. A mapping decision is required before
   either can feed the graph.

## Questions that need your answer

1. ~~**Which substrate is authoritative?**~~ **Settled, see `docs/DECISIONS.md`
   D2.** The division of labour stands: citation spine and speech layer from the
   master CSVs, entity spans from the BIO CSVs. The join key does not. The
   `sent_id + 1 == sentence_no` join recommended here was re-measured across all
   18 parvas and fails completely on Karna, Shalya, Mahaprasthanika and
   Svargarohana, because the master CSV is a sparse subset that kept its original
   indices while the BIO files re-segmented. The earlier 99.4 to 99.8% figure
   came from a five-parva sample that happened to exclude all four failures. The
   join is on normalized sentence text instead, measured at 95.1% to the LLM BIO
   files and 95.7% to the human ones.
2. **Does the project accept a chapter-level citation ceiling?** If shloka-level
   is required, the answer is to acquire a shloka-aligned Sanskrit text, which
   this repo does not have.
3. **How should the verifier circularity be handled?** Options: annotate a fresh
   independent sample, restrict Phase 5 to the 6 revised parvas and report the
   limitation, or reframe Phase 5 as a consistency measure rather than grounding.
4. **Scope: all 18 parvas or the 6 with a real human pass?** These give very
   different projects. The 6 are fully chapter-addressable and human-checked.
5. **Is the `gold_hand` tier abandoned?** The recommendation is yes, it is
   mislabelled.
6. **Label mapping between substrates:** keep CELESTIAL, keep LOC, and what
   becomes of NOT_AN_ENTITY?

## Method note

Facts here came from direct file reads plus a six-way parallel investigation
with adversarial verification. Three of the six investigation agents and all
verification agents were cut off by a session limit. The CE-address, Sanskrit
and speech-attribution sections were therefore measured directly rather than
agent-reported, and the human-versus-LLM diff was computed twice independently
with matching results. Numbers not independently confirmed are the master-CSV
offset-recovery sample (9,367 instances, one measurement) and the
`sent_id + 1 == sentence_no` join rate, which was subsequently re-measured and
found wrong; see `docs/DECISIONS.md` D2.
