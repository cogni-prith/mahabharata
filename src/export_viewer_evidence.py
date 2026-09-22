"""Export full relation passages for the offline explorer without retraining."""
import json
import re
from pathlib import Path
import pandas as pd
from review_viewer_graph import review

ROOT = Path(__file__).resolve().parents[1]


def teaching_pair(doc, names):
    """Bind only named arguments of explicit teaching, imparting or learning."""
    def named(token):
        if token.pos_ != "PROPN" or token.text not in names:
            return None
        if any(c.dep_ in ("compound", "conj") and c.pos_ == "PROPN" for c in token.children):
            return None
        return token.text

    for verb in doc:
        if verb.dep_ != "ROOT" or verb.lemma_ not in ("teach", "impart", "learn", "study"):
            continue
        if any(t.dep_ == "neg" or t.lower_ in ("if", "unless", "whether", "never")
               or t.tag_ == "MD" for t in verb.sent) or "?" in verb.sent.text:
            continue
        subjects = [named(c) for c in verb.children if c.dep_ == "nsubj"]
        subjects = [n for n in subjects if n]
        if len(subjects) != 1:
            continue
        if verb.lemma_ == "teach":
            objects = [named(c) for c in verb.children if c.dep_ in ("dobj", "dative")]
        else:
            prep = "to" if verb.lemma_ == "impart" else "from" if verb.lemma_ == "learn" else "under"
            objects = [named(o) for c in verb.children if c.dep_ == "prep" and c.lower_ == prep
                       for o in c.children if o.dep_ == "pobj"]
            if verb.lemma_ == "impart" and not any(c.dep_ == "dobj" and c.lemma_ in
                    ("knowledge", "science", "instruction", "teaching") for c in verb.children):
                continue
        objects = [n for n in objects if n]
        if len(objects) == 1 and objects[0] != subjects[0]:
            return (objects[0], subjects[0]) if verb.lemma_ in ("learn", "study") else (subjects[0], objects[0])
    return None, None


def teaching_passages(entities):
    import spacy
    mentions = pd.read_parquet(ROOT / "build/mentions.parquet")
    sentences = mentions.drop_duplicates(["parva", "sent_id"])
    sentences = sentences[sentences.sentence.str.contains(
        r"\b(?:impart(?:ed|s)?|taught|teach(?:es|ing)?|learn(?:ed|t)?|studied)\b", flags=re.I, na=False)]
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    # Regression guards use synthetic grammar examples, never evaluation data.
    names = set(entities)
    assert teaching_pair(nlp("Bhargava imparted knowledge to Karna."), names) == ("Bhargava", "Karna")
    assert teaching_pair(nlp("Bhargava did not teach Karna."), names) == (None, None)
    assert teaching_pair(nlp("Bhargava may teach Karna."), names) == (None, None)
    assert teaching_pair(nlp("Bhargava imparted strength to Karna."), names) == (None, None)
    name_pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, names)) + r")\b")
    output = []
    for r, doc in zip(sentences.itertuples(), nlp.pipe(sentences.sentence, batch_size=128)):
        found = sorted(set(name_pattern.findall(r.sentence)))
        if not found:
            continue
        teacher, student = teaching_pair(doc, names)
        chapter = int(r.adhyaya_ordinal) if pd.notna(r.adhyaya_ordinal) else None
        output.append({"names": found, "teacher": teacher, "student": student,
                       "text": r.sentence, "parva": r.parva, "chapter": chapter,
                       "cited": bool(r.chapter_cited) and chapter is not None,
                       "sentence_id": int(r.sent_id), "status": "source match"})
    return output


def main():
    idx = json.loads((ROOT / "out/embedding_index.json").read_text())
    entities = {n: i for i, n in enumerate(idx["entities"])}
    relations = {n: i for i, n in enumerate(idx["relations"])}
    original = pd.read_parquet(ROOT / "build/relations.parquet")
    frame, withheld, review_stats, sources = review(original)
    for name in sorted(set(frame.predicate) - set(relations)):
        relations[name] = len(relations)
    records = {}
    edges = {}
    for r in frame.itertuples():
        if r.subject not in entities or r.object not in entities or r.predicate not in relations:
            continue
        key = "%d:%d:%d" % (entities[r.subject], relations[r.predicate], entities[r.object])
        chapter = int(r.adhyaya_ordinal) if pd.notna(r.adhyaya_ordinal) else None
        item = {"parva": r.parva, "chapter": chapter,
                "cited": bool(r.cited) and chapter is not None,
                "text": r.evidence if isinstance(r.evidence, str) else "",
                "confidence": float(r.confidence), "depth": int(r.depth),
                "sentence_id": int(r.sent_id), "pattern": r.pattern}
        if item not in records.setdefault(key, []):
            records[key].append(item)
        if key not in edges:
            cite = r.parva + (" %d" % chapter if item["cited"] else "")
            edges[key] = [entities[r.subject], relations[r.predicate], entities[r.object], cite,
                          0, int(item["cited"]), item["text"][:300], float(r.confidence), int(r.depth)]
        edges[key][4] += 1
    encoded = json.dumps(records, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    teaching = teaching_passages(entities)
    triples = set(zip(original.subject, original.predicate, original.object))
    for passage in teaching:
        passage["in_training_graph"] = ((passage["student"], "instructedBy", passage["teacher"]) in triples
                                          if passage["teacher"] else None)
    source_json = json.dumps(teaching, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    rejected = []
    for r in withheld.itertuples():
        rejected.append({"subject": r.subject, "predicate": r.predicate, "object": r.object,
                         "parentage_unspecified": r.reason.startswith("Parentage is stated"),
                         "reason": r.reason, "text": r.evidence, "parva": r.parva,
                         "chapter": int(r.adhyaya_ordinal) if pd.notna(r.adhyaya_ordinal) else None,
                         "cited": bool(r.cited), "sentence_id": int(r.sent_id)})
    vasava = sources[sources.sentence.str.contains("replied to Indra", regex=False, na=False)
                     & sources.sentence.str.contains("O Vasava", regex=False, na=False)]
    notes = {}
    if len(vasava):
        s = vasava.iloc[0]
        notes["Vasava"] = {"text": "Vasava is used as a name for Indra in this passage. Divine parentage is distinct from biological, legal or foster parentage. This note does not merge entity vectors.",
                           "evidence": s.sentence, "parva": vasava.index[0][0],
                           "chapter": int(s.adhyaya_ordinal) if pd.notna(s.adhyaya_ordinal) else None,
                           "cited": bool(s.chapter_cited)}
    parentage = [r for r in rejected if r["parentage_unspecified"]]
    for r in frame[frame.predicate == "bornTo"].itertuples():
        parentage.append({"subject": r.subject, "object": r.object, "parentage_unspecified": True,
                          "text": r.evidence, "parva": r.parva,
                          "chapter": int(r.adhyaya_ordinal) if pd.notna(r.adhyaya_ordinal) else None,
                          "cited": bool(r.cited), "sentence_id": int(r.sent_id)})
    # This identification is scoped to the inspected Karna death accounts.
    # Arjuna's ten-name declaration does not resolve homonyms across the epic.
    death_accounts = {}
    declaration = sources[sources.sentence.str.contains("Arjuna replied", regex=False, na=False)
                          & sources.sentence.str.contains("my ten names", regex=False, na=False)]
    slayers = set(frame.loc[(frame.subject == "Karna") & (frame.predicate == "slainBy"), "object"])
    if {"Arjuna", "Kiriti"} <= slayers and len(declaration):
        s = declaration.iloc[0]
        if "Kiriti" in s.sentence:
            death_accounts["Karna"] = {"slayer": "Arjuna", "appellations": ["Kiriti"],
                "text": s.sentence, "parva": declaration.index[0][0],
                "chapter": int(s.adhyaya_ordinal) if pd.notna(s.adhyaya_ordinal) else None,
                "cited": bool(s.chapter_cited), "sentence_id": int(declaration.index[0][1])}
    snapshot = {"base_entities": idx["entities"], "base_relations": idx["relations"],
                "relations": list(relations), "edges": list(edges.values()), "withheld": rejected,
                "stats": review_stats, "notes": notes, "parentage": parentage,
                "death_accounts": death_accounts}
    review_json = json.dumps(snapshot, ensure_ascii=True, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    (ROOT / "out/evidence.js").write_text("window.KGEvidence=" + encoded + ";\nwindow.KGTeaching=" + source_json + ";\nwindow.KGReview=" + review_json + ";\n")
    total = sum(map(len, records.values()))
    (ROOT / "build/viewer_evidence_report.md").write_text(
        "# Viewer evidence export\n\n%d relation groups; %d full evidence records.\n"
        "\n%d additional teaching passages retrieved from mentions.parquet. "
        "%d have explicit named teaching arguments. These are source matches, not additions "
        "to the asserted or training graphs. Names are not merged.\n"
        % (len(records), total, len(teaching), sum(bool(r["teacher"]) for r in teaching)))
    print("Exported %d evidence records for %d relation groups; %d teaching passages" % (total, len(records), len(teaching)))


if __name__ == "__main__":
    main()
