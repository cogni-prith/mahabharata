# Main explorer

## Semantic mode (2026-09-16)

Open `http://127.0.0.1:8765/index.html` after starting the local service:

`.venv/bin/python src/serve_explorer.py --port 8765`

The same page now has **Text & concepts** and **Graph model** modes. Text search
retrieves source passages and typed profiles for arbitrary questions. Try `war`,
`Battles at Kurukshetra`, `dharma`, or `Weapons of Arjuna`. Descriptions are quoted
source text, not generated answers. The ontology control shows the class
definitions; coloured filters identify candidate types, not factions. Person and
Concept readings of Dharma remain separate. Stars are not automatically deities.

The semantic map uses a separate text encoder and seeded UMAP projections. Its
geometry is not mixed with RotatE. Places in event accounts are proposal-backed
retrieval links, not a claim that those places have the largest cosine score.
Event links and classifications remain unverified. No interface can certify that
all extraction mistakes have been found.

Opening `out/index.html` as a file still runs the original graph without a server.
It cannot run semantic inference, and the semantic control reports that clearly.
The local service is read-only and writes nothing to the corpus.

It no longer follows that nothing is sent remotely. One endpoint, `/api/chat`,
sends the question and its retrieved context to a third-party model, and the
**Ask** mode on this page uses it. That is D27. It applies to Ask and to
nothing else: **Text & concepts** and **Graph model** remain local, and opening
`out/index.html` from `file://` still runs the graph with no server and no
request of any kind.

Ask appears only when the service reports it, and it says on screen which
state it is in. Started without `GROQ_API_KEY`, the endpoint still answers:
it returns the retrieved facts and passages, generates nothing, sends nothing,
and the banner says so. The key is read from the environment of the process
you start and is never served to the browser.

Rebuild the separate semantic artefacts, with the service stopped:

- `.venv/bin/pip install -r requirements-semantic.txt`
- `.venv/bin/python src/semantic_index.py`
- `.venv/bin/python src/semantic_search.py --build`
- `.venv/bin/python src/serve_explorer.py --port 8765`

Builds write only `build/semantic/` and `out/semantic.trig`, not KGE training
outputs. The first vector build downloads the public pretrained model. The
manifest records the model revision, source/vector hashes, encoding method and
projection seed. Every subsequent service start loads local cached weights.

Checks:

- `.venv/bin/python src/test_semantic.py`
- `env TMPDIR=/dev/shm .venv/bin/python src/test_semantic_browser.py` (service running)

The second check covers real search, invalid requests, Host/Origin checks, path
traversal, source citations, class ambiguity, 3D pixels and motion, desktop/mobile,
offline mode, and the unchanged graph query path. These are regression checks,
not independent accuracy measurements. The automated Aikido scan could not run
because its API key/OS keychain was unavailable; it is not reported as passing.

## Ask mode (2026-09-21)

A third mode beside **Text & concepts** and **Graph model**. It answers a typed
question from the extracted graph and the passage index together, and it is the
only part of this interface that leaves the machine.

Start it with a key to generate answers, or without one to see the evidence:

- `.venv/bin/pip install -r requirements-chat.txt`
- `GROQ_API_KEY=... .venv/bin/python src/serve_explorer.py --port 8765`

The model is `openai/gpt-oss-120b`, chosen from what the account serves rather
than assumed. Not `groq/compound`: it carries its own web search, and a
grounded answer must not have a route back to the open internet.

Spelling and grammar do not have to be right. `Arjun` reaches Arjuna and
`whos arjuns father` parses. Grammar repair is an instruction in the prompt,
not a second model.

The answer sits beside the evidence it was built from, in four tabs. **Facts**
are asserted triples, each with its chapter citation or the word uncited, its
confidence and its extraction pattern. **Passages** are the retrieved source
excerpts. **Names** shows how each name in the question was resolved, and
separates the unresolved ones. **Sent** is the exact context the model was
given, or would have been given.

Two things the panel draws rather than trusting prose to say. An appellation is
never an identity: `Radheya` retrieves Karna's records as a candidate, and the
candidate is marked unresolved with its confidence and rival count wherever it
appears, including on every fact reached through it. And spelling variants are
not merged: `Yudhistira` reaches both `Yudhisthira` and `Yudhishthira`, which
are separate entities in this graph, and the panel says nothing claims they are
one person.

Answers are generated text over retrieved evidence. Retrieval is cited.
Generation is not independently verified, and the panel says so on every turn.

Checks, with the service running:

- `.venv/bin/python src/test_chat.py` (offline, no key, no service)
- `env TMPDIR=/dev/shm .venv/bin/python src/test_chat_browser.py`

The browser check covers the endpoint guards, the three-way mode switch, the
unresolved and variant markers, the citation on every fact, that the page makes
no request off this machine other than the one the service makes for it, and
that the answer renderer cannot be made to emit a tag from model output.

## Lineage mode (2026-09-23)

A fourth mode, drawn from the curated lineage table in `annotations/lineage/`.
It works from `file://` with no server and sends nothing. D28 records why the
table is shown beside the graph and not merged into it.

- **Question box.** The main query bar answers parents, children, siblings,
  grandparents, grandchildren, ancestors, descendants, co-parents, and how two
  people are related. Answers are computed from the table in the page. Each
  edge shows a chapter badge when the asserted graph states it, and
  `curated, uncited` when it does not. Kin triples the text holds and the
  table lacks are listed as `in the text, not in the table`.
- **Tree.** Three generations up and two down from the focus, with everyone
  who shares a parent on the focus row.
- **Graph.** All 240 people, one row per generation. Drag to pan, scroll to
  zoom, select anyone to refocus.

Solid edges are stated in the text. Dashed edges are the table alone.

## Original graph mode

Updated 2026-09-15 in response to the request to change `out/index.html`.
The entry point is the graph and vector explorer. Annotation pages are not part
of its navigation or launch flow.

The page presents the extracted graph, the real 3D UMAP projection, and an
entity/evidence inspector together. Three.js and OrbitControls provide orbit,
pan, zoom, and rendering; the point positions are the existing `xyz` projection.
The 2D switch uses the existing `xy` projection. No vectors were retrained.

Natural-language relation questions, incomplete triples, nearest neighbours,
and arithmetic analogies use `out/kg-core.js`. Queries outside the supported
grammar receive an explicit no-match response. A spelling lookup from Draupadi
to Droupadi is disclosed on screen. Fights are not relabelled as enmity; parent
edges do not infer gender. The contested Abhimanyu slaying stays unasserted.

`src/export_viewer_evidence.py` exports full evidence passages from the current
relations parquet to `out/evidence.js`. The UI keeps extractor confidence,
chapter citation, and telling depth visible with the passage. The graph now
uses a category-coloured, predicate-grouped fan, inspired by the supplied
`Downloads/knowledge-graph (1).html`. Initially each predicate shows up to six
neighbours, with the group total visible; Show all expands every group. Category
filters affect the fan only. Both / Graph / Space modes change pane allocation.
The inspector and result table still expose all connections.

Questions such as `How is Karna related to Arjuna?` first display the source
passages describing both as children of Kunti. Generic parentage is also visible
in the profile and fan, without guessing biological versus foster parentage.
Other extracted paths remain secondary. Each step retains its predicate
direction and evidence. This is a path in an imperfect extracted graph, not a
verified claim of family relationship.

## Missing answers and source retrieval

The supplied reference contains 153 entities and 388 hand-authored seed edges;
none is cited. Its teaching handler already supports `who taught Karna`, using
seed links to Drona and Parashurama. Those seed links were not imported, merged
into the asserted graph, or used for training. Its first-hit substring name
resolution was not adopted because it can silently choose ambiguous entities.

In the trained graph, Karna has no `instructedBy` edge even though the question
parses. The evidence exporter now also retrieves teaching sentences from the
mention spine and writes `window.KGTeaching` beside the existing evidence data.
Only explicit named subjects and recipients in restricted teaching/learning
dependency patterns receive a displayed teacher/student binding. Negated and
modal clauses, pronouns, and unnamed recipients do not receive such a binding.
Other retrieved sentences remain context passages, not relationship assertions.

Examples read directly from the prose:

- Karna Parva 24: Bhargava imparts his knowledge of dhanurveda to Karna.
- Adi Parva 123: Drona teaches Arjuna fighting from different mounts and on foot.

The source name `Bhargava` is preserved, not silently resolved to Parashurama.
Source matches show full text, citation status and sentence ID. Whether a bound
pair also exists in the training graph is checked against the relation data.
Retrieval does not change asserted triples, embeddings, or evaluation metrics.

## Relation integrity corrections

The viewer cannot make the underlying graph correct by redrawing it. Three
Karna-parent records were traced to their actual evidence during this update:

- Aranyaka 241, sentence 14431: Dhritarashtra's son speaks to Karna. The
  extraction incorrectly makes Karna the son (`poss_kinship`, score 0.722).
- Bhishma 93, sentence 6869: Duryodhana, addressed by Karna, speaks to
  Duhshasana. The extraction incorrectly makes Duryodhana Karna's parent.
- Shalya 63, sentence 5764: a list contains Bhishma, Shantanu's son, then
  Karna. The extraction incorrectly attaches Karna to Shantanu.

These are upstream argument-attachment failures, not query-parser failures.
They are now excluded by shared extractor guards: kinship requires one directly
named bearer and holder; appositions with competing named bearers abstain.
Role phrases at a new clause boundary cannot inherit the previous clause's
object as their bearer. This removes Bhishma/Krishna parentage and the false
Bhishma/Chedi and Bhishma/Madra rulership links. Archived parentage candidates
are re-parsed too, so they cannot restore corrected errors through source links.
Coreference may bind only the root of its exact mention span, must not resolve
first/second-person speech through the narrator, and must abstain on competing
names. No reference seed edges were imported to replace rejected records.

Vasava is addressed as Indra in Anushasana 12. The inspector shows that passage
without merging the entity vectors. A divine parent must not be labelled a
biological parent. Explicit divine, foster and biological language keeps its
type; generic `son of` statements remain source matches with `kind unspecified`.
For example, Karna/Radha remains accessible without asserting biological
parentage. Sanjaya/Vasava and Keshava/Vasava do not survive the attachment guards.

`src/review_viewer_graph.py` rechecks relation records using the shared extractor
and writes `out/reviewed_relations.ttl`, `build/reviewed_relations.parquet`,
`build/withheld_relations.parquet`, and `build/relation_review.md`.
`window.KGReview` carries the corresponding viewer snapshot. Unresolved source
records from `build/relation_candidates.parquet` survive repeated exports even
after the main pipeline has removed them from its asserted graph.

Current counts are generated in `build/relation_review.md`; they can change as
the concurrent pipeline incorporates corrections. Unspecified parentage now
uses the existing `bornTo` predicate where emitted by that pipeline. Earlier
candidate passages remain accessible without guessing a more specific type.
Withheld records are not all disproved relationships. This is a rule-based
recheck, not an independent precision estimate; vocative telling records have
not been rechecked here.

Death extraction requires explicit names and literal death verbs. `destroy`
does not establish a death. Conflict assertions also reject arguments obtained
only through coreference or noun-phrase linking: those guesses produced Karna's
false slayers and several false opponents. Rejected evidence remains available.
For Karna's inspected death accounts, Kiriti is grouped under Arjuna in the
display, with the self-identification passage from Virata 39 accessible beside
the answer. This is not a global alias merge or a change to the vectors.

The eight colours are K-means groups of learned entity vectors, using seed
20260917. They do not identify factions, families, confidence, or UMAP regions.
The cluster buttons filter the scatter; their tooltips show group size and
prominent names. The independent UMAP projection determines point positions.

The page compares its displayed triples with the bundled model graph and reports
whether they match. Display-only predicates cannot acquire invented vectors:
the graph and vector relation indexes are separate. This viewer export does not
train a model or overwrite concurrent training artefacts or annotation work.

All graph-mode scoring is local; that pane makes no external request and needs
no server. Ask mode is the exception and is covered above and in D27. Predictions and analogy candidates are marked predicted and uncited.
The model remains below the scientific entry gate; its status and evaluation
details are accessible in the header. UI correctness is not model validation.

## Rebuild and verify

- `.venv/bin/python src/export_viewer_evidence.py`
- `.venv/bin/python src/test_relation_integrity.py`
- `node src/phase7_check.js`
- `.venv/bin/python src/test_viewer_browser.py`
- `.venv/bin/python src/lineage_export.py`
- `.venv/bin/python src/test_lineage_browser.py`

Only when changing the vendor bundle:

- `npm ci --prefix tools/viewer --no-audit --no-fund`
- `npm run build --prefix tools/viewer`

The bundle is committed with the page for offline use. Versions and source
dependencies are locked in `tools/viewer/package-lock.json`; licenses accompany
the bundle in `out/vendor/`. Playwright is a development-only browser test tool.
