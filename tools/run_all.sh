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
$PY src/phase2_audit.py                # canonical_id failure counts
# HUMAN: judge build/entity_audit_sample.csv (200 rows, seed 20260912) for the Phase 2 gate
$PY src/phase3_tellings.py             # speech frames and diegetic depth
$PY src/phase3_spotcheck.py            # Phase 3 gate: the Gita and the Nala tale, asserts
$PY src/phase4b_coref.py               # coreference over the prose, feeds phase 4
$PY src/phase4_relations.py            # relation extraction, two passes, writes relations.parquet and ttl
$PY src/phase4d_verbs.py               # verb ceiling measurement, evidence for the predicate set
$PY src/phase4_sample.py               # draw the precision sample
$PY src/phase4e_sample.py              # draw the sample for the expanded patterns
# HUMAN: judge the samples into build/phase4e_wrong.txt and build/phase4g_judged.csv
$PY src/phase4_precision.py            # precision from the judgements on record
$PY src/phase5_propose.py              # proposals, separate graph, never asserted
# HUMAN: the blind 300-500 sentence calibration set, which the Phase 5 gate needs
$PY src/phase6_gate.py                 # entry gate. FAILS on all three conditions
$PY src/phase6_train.py                # RotatE. Below the gate, on instruction. D19
$PY src/phase7_project.py              # UMAP, projection.json, projection.js
node src/phase7_check.js               # the three query modes end to end
$PY src/phase8_divergence.py           # graph vs embedding neighbourhoods
$PY src/phase8b_alias.py               # cosine as an alias candidate generator
$PY src/validate_outputs.py            # 24 integrity checks over out/. Non-zero exit on any failure

echo "Open out/index.html in a browser. No server needed."
