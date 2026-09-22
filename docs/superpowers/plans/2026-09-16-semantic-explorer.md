# Cited Semantic Explorer Implementation Plan

> Execute inline with superpowers:executing-plans; preserve concurrent edits.

**Goal:** Make the existing explorer search concepts, events and source passages.
**Architecture:** A local SQLite/text-embedding index augments, not replaces,
the existing graph model. Excerpts and proposals retain their provenance.
**Tech stack:** Python stdlib HTTP/SQLite, Sentence Transformers, rdflib/SHACL,
existing NumPy/UMAP and vendored Three.js. No web framework.
**Spec:** `docs/superpowers/specs/2026-09-16-semantic-explorer-design.md`.

## Constraints

Never read the blind calibration key; never overwrite training outputs; never
publish. Event relations are proposals, not training triples. Bind 127.0.0.1.
Record model ID/revision, input hashes, seed and limitations. Preserve file mode.

## Tasks

- [ ] Ontology and corpus index: `ontology/mahabharata.ttl`, `ontology/shapes.ttl`,
  `src/semantic_index.py`, `src/test_semantic.py`. Test exact text/citation
  boundaries, content IDs, type ambiguity and separate event occurrences before
  implementation. `build_index(root)` writes a new SQLite DB and named-graph
  proposal export; it does not mutate the original graph.
- [ ] Retrieval and API: `src/semantic_search.py`, `src/serve_explorer.py`.
  `SearchIndex.search(query, kind, limit)` returns ranked nodes and passages;
  `node(id)` returns exact excerpts and candidate relations. Test query bounds,
  SQLite quoting, localhost-only HTTP, traversal rejection and missing IDs.
  Build real embeddings and seeded projections; run `war` retrieval checks.
- [ ] Same-page UI: add `out/semantic.js` and `out/semantic.css`, extend
  `out/index.html` without rewriting `out/explorer.js`. Use the existing vendor
  library for the map and native tabs/details for evidence. Test service-off,
  pending, empty, error, citation, selection, filtering and dimension states.
- [ ] Verify: extend `src/test_semantic.py` with an opt-in live browser check;
  run existing graph tests, semantic/schema/API tests and Playwright screenshots
  at 390, 768 and 1440px with canvas pixel checks. Inspect rendered images.
  Record measured coverage and limitations. Start the loopback service and
  provide its URL. Do not commit other agents' work.
