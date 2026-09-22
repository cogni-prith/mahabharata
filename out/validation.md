# Output validation

Automated, from `src/validate_outputs.py`. 25 checks, 1 failed.

This checks the hard rules that are mechanically checkable. It cannot check
whether a triple is true; that is what the precision samples are for.

| check | result | detail |
|---|---|---|
| relations.ttl parses | pass | 22397 triples |
| proposed.ttl parses | pass | 234 triples |
| entities.ttl parses | pass | 12053 triples |
| passages.ttl parses | pass | 13691 triples |
| tellings.ttl parses | pass | 28260 triples |
| no proposal appears in the asserted graph | **FAIL** | 1668 asserted, 25 proposed, 1 shared |
| no Proposal node in relations.ttl | pass | by rdf:type |
| no Assertion node in proposed.ttl | pass | by rdf:type |
| every proposal carries a score and status proposed | pass | 26 proposals |
| every chapter-cited assertion names a passage | pass | 0 without a passage |
| every uncited assertion names no passage | pass | 0 carrying an address they should not have |
| every cited passage exists in passages.ttl | pass | 1086 distinct references, 0 missing |
| every entity in an assertion exists in entities.ttl | pass | 666 entities used, 3865 declared, 0 missing |
| every telling passage exists in passages.ttl | pass | 1769 references, 0 missing |
| every telling names a narrator or declares it unresolved | pass | 5465 tellings, 373 unresolved, 0 malformed |
| no owl:sameAs anywhere | pass | 0 found |
| predicate set within the recorded inventory | pass | 11 in use (10 asserted, 2 proposed), 0 outside: bornTo, cursedBy, defeatedBy, foughtWith, hasDivineProgenitor, hasSpouse, instructedBy, isEnemyOf, rules, slainBy, speaksTo |
| hand-annotated inventory at or under 15 | pass | 15 annotated, 2 extractor-only |
| entity_emb.bin matches the index | pass | 224400 floats, 561 entities x 400 |
| relation_emb.bin matches the index | pass | 3600 floats, 9 relations x 400 |
| no NaN or infinity in the embedding | pass | checked 228000 floats |
| every embedded entity appears in the asserted graph | pass | 561 embedded, 0 outside the graph |
| projection covers every embedded entity | pass | 561 coordinates, 561 entities |
| both seeds recorded | pass | train 20260917, umap 20260917 |
| below-gate flag present in every embedding artefact | pass | index, projection.json, projection.js |

## Failures

- no proposal appears in the asserted graph: 1668 asserted, 25 proposed, 1 shared

