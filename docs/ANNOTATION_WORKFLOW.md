# Annotation instrument

Approved implementation, 2026-09-14. Both review pages run directly from disk.

1. Complete the blind relation review first in `build/annot_blind.html`.
   Its 419 frozen passages contain no model output. Keep the private calibration
   key closed. Each sentence needs an explicit decision; uncertainty belongs to
   the individual triple.
2. Review candidate links in `build/annot_entities.html`. All candidates for an
   appellation appear together with their available passages. Several candidates
   may be valid in different contexts. Decisions do not merge entities.
3. Export drafts regularly. Browser recovery is best effort; exported JSON is
   the durable record. Importing a file previews its identity and replaces the
   current draft only after confirmation. Changed source content is rejected.
4. Export a completed review with an annotator name and the applicable
   attestation. Install it with `python src/annot_pages.py --import PATH`.
   Existing annotation files are preserved unless `--replace` is supplied.
5. Run `python src/phase5_calibrate.py` and `python src/phase2_score.py`.
   Calibration cannot read the private key until a complete submitted blind
   review passes validation. The alias report reuses entity rulings.

The fifteen manual predicates are the recorded fourteen plus `bornTo`, which the
proposal layer already emits and the recorded inventory did not name. That +1 is
recorded in D22 and still wants a ruling; dropping it costs one regenerated page
while nothing has been annotated. `isAlliedWith` and `isEnemyOf` remain automated
inventory terms. No predicate is inferred from co-occurrence during review.

Review exports are versioned, identify their exact dataset, and carry literal
source records. No automatic acceptance, entity merge, or retraining follows a
review. Per-predicate matches are lexical agreement with a reviewer, not proof
of textual truth. The A, B and C strata are reported separately.

Verification covers content changes, missing and duplicate rows, unfinished
reviews, key access guards, script-safe embedding, export/import round trips,
storage failures, and desktop/mobile layout.
