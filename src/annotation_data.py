"""Public review inputs and export validation. Never opens the calibration key."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
PREDICATES = {
    "hasBiologicalParent": "Child to biological parent.",
    "hasLegalParent": "Child to socially recognised parent, distinguished by the text.",
    "hasFosterParent": "Child to the parent who raised them.",
    "hasDivineProgenitor": "Child to the divinity who fathered them.",
    "bornTo": "Child to parent; birth is stated without specifying parentage kind.",
    "hasSpouse": "Spouse to spouse; either order.",
    "slainBy": "Victim to killer; a killing that occurred, not a threat or prediction.",
    "defeatedBy": "Loser to winner; defeat short of death.",
    "foughtWith": "Two participants in an actual fight; either order.",
    "speaksTo": "Speaker to addressee.",
    "cursedBy": "Cursed person to the person who cursed them.",
    "instructedBy": "Student to teacher.",
    "rules": "Ruler to place ruled.",
    "isAmshaOf": "Portion or incarnation to its originating divinity.",
    "rebornAs": "Earlier life to later life; not an identity merge.",
}
AUTOMATED = set(PREDICATES) | {"isAlliedWith", "isEnemyOf"}
SOURCES = {
    "blind": ("build/blind_calibration_sheet.csv",),
    "entities": ("build/entity_audit_sample.csv", "build/appellation_evidence.json"),
}
COLUMNS = {
    "blind": ("key", "parva", "adhyaya", "sentence"),
    "entities": ("appellation", "candidate_entity", "appellation_label", "entity_label"),
}
FILENAMES = {"blind": "phase5_blind.json", "entities": "phase2_entities.json"}


def digest(value):
    data = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def make_dataset(task, rows):
    if task not in SOURCES or not rows or len({r["key"] for r in rows}) != len(rows):
        raise ValueError("Unknown task, empty input, or duplicate source keys")
    predicates = PREDICATES if task == "blind" else {}
    data = {"schema_version": SCHEMA, "task": task, "rows": rows, "predicates": predicates}
    data["dataset_id"] = digest(data)
    data["sheet_id"] = (hashlib.sha1(" ".join(sorted(r["key"] for r in rows)).encode())
                        .hexdigest()[:12] if task == "blind" else None)
    return data


def load_dataset(task, root=ROOT):
    """Project only whitelisted columns, even if an upstream CSV gains columns."""
    paths = SOURCES[task]
    with (root / paths[0]).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not set(COLUMNS[task]) <= set(reader.fieldnames or []):
            raise ValueError("Missing public source columns for " + task)
        rows = [{k: r[k] for k in COLUMNS[task]} for r in reader]
    if task == "entities":
        with (root / paths[1]).open(encoding="utf-8") as f:
            evidence = json.load(f)["evidence"]
        for r in rows:
            pair = [r["appellation"], r["candidate_entity"]]
            r["key"] = digest(pair)[:20]
            ev = evidence.get("||".join(pair), {})
            r["sightings"] = [{k: s.get(k) for k in ("parva", "adhyaya", "sentence")}
                              for s in ev.get("sightings", [])]
            r["total_sightings"] = ev.get("total", 0)
        rows.sort(key=lambda r: (r["appellation"].casefold(), r["candidate_entity"].casefold()))
    return make_dataset(task, rows)


def blank_record(source, task):
    out = {"key": source["key"], "source": source, "status": "undecided", "notes": ""}
    out.update({"triples": []} if task == "blind" else {"corrected_referent": ""})
    return out


def blank_review(data):
    return {"schema_version": SCHEMA, "task": data["task"],
            "dataset_id": data["dataset_id"], "sheet_id": data["sheet_id"],
            "annotator": "", "submitted": False, "attestation": False,
            "rows": [blank_record(r, data["task"]) for r in data["rows"]]}


def validate_review(review, data, require_submitted=False):
    """Reject stale, malformed or incomplete submissions before any private read."""
    if not isinstance(review, dict):
        raise ValueError("Expected a review object")
    expected = set(blank_review(data))
    if set(review) != expected:
        raise ValueError("Unexpected review fields or missing metadata")
    for k in ("schema_version", "task", "dataset_id", "sheet_id"):
        if type(review[k]) is not type(data[k]) or review[k] != data[k]:
            raise ValueError("Review belongs to another task, version, or dataset: " + k)
    if not isinstance(review["annotator"], str) or any(type(review[k]) is not bool
                                                    for k in ("submitted", "attestation")):
        raise ValueError("Invalid reviewer metadata")
    records = review["rows"]
    if not isinstance(records, list) or len(records) != len(data["rows"]):
        raise ValueError("Review must retain every source row, including drafts")
    source = {r["key"]: r for r in data["rows"]}
    seen = set()
    completed = 0
    for r in records:
        if not isinstance(r, dict) or not isinstance(r.get("key"), str):
            raise ValueError("Malformed review row")
        key = r["key"]
        if key not in source or key in seen:
            raise ValueError("Unknown or duplicate row: " + key)
        seen.add(key)
        if set(r) != set(blank_record(source[key], data["task"])) or r["source"] != source[key]:
            raise ValueError("Changed literal source or unexpected fields: " + key)
        if not isinstance(r["notes"], str):
            raise ValueError("Notes must be text")
        if data["task"] == "blind":
            if r["status"] not in ("undecided", "none", "triples") or not isinstance(r["triples"], list):
                raise ValueError("Invalid sentence decision")
            filled = bool(r["triples"])
            for t in r["triples"]:
                if (not isinstance(t, dict) or set(t) != {"subject", "predicate", "object", "unclear"}
                        or any(not isinstance(t[k], str) for k in ("subject", "predicate", "object"))
                        or type(t["unclear"]) is not bool
                        or t["predicate"] not in ("", *PREDICATES)):
                    raise ValueError("Malformed triple")
                filled = filled and all(t[k].strip() for k in ("subject", "predicate", "object"))
            if r["status"] == "none" and r["triples"]:
                raise ValueError("A no-relation decision cannot contain triples")
            complete = r["status"] == "none" or (r["status"] == "triples" and bool(filled))
        else:
            if r["status"] not in ("undecided", "yes", "no", "not_a_person", "unclear"):
                raise ValueError("Invalid entity decision")
            if not isinstance(r["corrected_referent"], str):
                raise ValueError("Corrected referent must be text")
            complete = r["status"] != "undecided"
        completed += bool(complete)
    if review["submitted"] or require_submitted:
        if (not review["submitted"] or not review["attestation"] or not review["annotator"].strip()
                or completed != len(records)):
            raise ValueError("A submitted review needs every decision, reviewer name, and attestation")
    return completed
