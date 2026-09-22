# Cited semantic explorer

Approved in conversation on 2026-09-16. The user approved richer ontology,
event relations, semantic retrieval and a localhost service, replacing the
static-only restriction for this feature. No publication or remote inference
is authorized.

## Scope

Keep `out/index.html` and its existing graph explorer. Add text search and a
separately labelled semantic map of people, places, groups, concepts, objects,
works and event mentions. Do not mix text vectors and RotatE coordinates.
The three new domain relations are `occurredAt`, `hasParticipant`, `hasTopic`.
This does not expand the human calibration inventory or train RotatE again.

## Evidence

Index all master sentences for the eighteen scoped parvas. Existing mention
anchors provide chapters where available; other sentences remain uncited at
chapter level. Chunk only within one citation context. Preserve exact text,
sentence IDs and content hashes. Descriptions are excerpts, not LLM summaries.
Names are not silently merged; different types of Dharma remain separate.
Event mentions are distinct occurrences, not one event per generic word.
Annotation labels provide retrieval candidates, not verified ontology types.
Automatic event relationships remain explicitly scored proposals, in a
separate named graph. Passage co-occurrence is never a character relation.

## Architecture

Use SQLite FTS5 plus normalized pretrained sentence embeddings for retrieval.
Encode text profiles with the same encoder, project using seeded UMAP, and
label map colours by proposed ontology class. Keep rankings distinct from
explicit relations. Never force Kurukshetra to a chosen coordinate.
A stdlib HTTP service binds only to loopback, serves only `out/`, validates
Host/Origin and bounded JSON input, and exposes no write or shell endpoint.
Model weights are downloaded once; corpus text and queries stay local.

## Interface and verification

On localhost, semantic search accepts arbitrary queries; the Graph model tab
retains existing queries and predictions. File-open mode remains functional
with the existing graph and explicitly reports semantic service unavailability.
The semantic view provides a dense 3D map, class filters, a ranked result list,
and description/evidence/relations tabs with passage citations and uncertainty.
Test real war/Kurukshetra retrieval, ambiguous types, uncited records, distinct
event occurrences, malicious queries, local-only serving, SHACL validation,
browser interaction, canvas pixels, and desktop/mobile layouts. These are
regressions and retrieval smoke checks, not independent factual evaluation.
