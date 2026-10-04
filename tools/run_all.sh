#!/usr/bin/env bash
# The whole pipeline, in order, one command per line. Run from the repo root.
# Every step reads only committed inputs and files earlier steps wrote, so the
# run is reproducible from the two corpus folders and the two master CSVs.
#
# Phase 6 trains below its entry gate, which fails. That is D19 in
# docs/DECISIONS.md, it was instructed, and every output says so on its face.
# Steps marked HUMAN cannot run unattended and are listed where they belong.
set -euo pipefail
PY=.venv/bin/python

$PY src/build_spine.py                 # chapter boundaries from parva_pdfs -> build/spine.json
$PY src/phase1_mentions.py             # mentions + passages, reports the substrate join rate
$PY src/phase2_entities.py             # appellation table, entity declarations
$PY src/phase2_audit.py                # canonical_id failure counts, candidate link census
$PY src/phase3_tellings.py             # speech frames and diegetic depth
$PY src/phase3_spotcheck.py            # Phase 3 gate: the Gita and the Nala tale, asserts
$PY src/phase4b_coref.py               # coreference over the prose, feeds phase 4
$PY src/test_possessed_kin.py          # the possessed-kinship guard, asserts. D21
$PY src/phase4_relations.py            # relation extraction, two passes, writes relations.parquet and ttl
$PY src/phase4d_verbs.py               # verb ceiling measurement, evidence for the predicate set
$PY src/phase4_sample.py               # draw the precision sample
$PY src/phase4e_sample.py              # draw the sample for the expanded patterns
# HUMAN: judge a new sample into annotations/, keyed on the triple and the pattern as annotations/phase4e_judged.csv is. A row id is a position in relations.parquet and does not survive a re-extraction
$PY src/phase4_precision.py            # precision from the judgements in annotations/
# OPTIONAL: entity normalization. Uncomment both to build Phases 5 to 8 on the folded graph.
# $PY src/phase2b_confidence.py        # confidence per mention, build/er/ and the page build/er/er.html
# $PY src/phase4f_apply_er.py          # relation endpoints through accepted links; raw kept in build/relations_raw.parquet
$PY src/phase5_propose.py              # proposals, separate graph, never asserted
$PY src/phase5_blind_sample.py         # draw the blind set, docs/PHASE5_SAMPLING.md, seed 20260914. Refuses to overwrite
$PY src/annot_pages.py                 # build/annot_blind.html and build/annot_entities.html from public columns only
$PY src/test_annotation.py             # the export validator and a real Chrome round trip, asserts
# HUMAN: annotate build/annot_blind.html (419 passages) and build/annot_entities.html (246 links).
# HUMAN: do not open build/blind_calibration_key.csv first. Install with annot_pages.py --import PATH
$PY src/phase5_calibrate.py            # Phase 5 gate from the blind judgements. No-op with a message until they exist
$PY src/phase2_score.py                # Phase 2 gate from the entity census. No-op with a message until it exists
$PY src/phase6_gate.py                 # entry gate. FAILS on all three conditions
$PY src/phase6_train.py                # RotatE. Below the gate, on instruction. D19
$PY src/phase7_project.py              # UMAP, projection.json, projection.js
$PY src/export_viewer_evidence.py     # full source passages for the offline inspector
$PY src/lineage_export.py        # curated lineage table + text check -> out/lineage-data.js. D28
node src/phase7_check.js               # the three query modes end to end
$PY src/phase8_divergence.py           # graph vs embedding neighbourhoods
$PY src/phase8b_alias.py               # cosine as an alias candidate generator
$PY src/validate_outputs.py            # 25 integrity checks over out/. Non-zero exit on any failure

echo "Open out/index.html in a browser. No server needed."
