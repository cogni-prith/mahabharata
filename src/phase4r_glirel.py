"""Phase 4r. Zero-shot relation extraction with GLiREL, over the whole corpus once.

The ten predicates of Phase 4 come from hand-written spaCy patterns, and the
graph has no mother or father, no ally, enemy or charioteer. GLiREL
(Boylan et al. 2025, jackboyla/glirel-large-v0) classifies the relation between
two given entity spans against labels written as plain text, with no training.
It does not find entities itself, so the spans come from what this project
already has: the Phase 1 name mentions and the Phase 4b coreference spans, which
let "he" stand for the person it was resolved to.

Output is a candidate layer, never asserted: build/glirel_relations.parquet,
one row per (sentence, head span, tail span) whose best label is a relation,
with its score. Scores are not probabilities. A threshold has to be set against
a hand-judged sample before any of this is used as fact.

Coreference runs before this and needs a GPU; so does this, in practice. On a
CPU it is a few sentences a second, so it runs on Kaggle.

Run: python src/phase4r_glirel.py [--limit N] [--dry-run] [--model NAME]
  --dry-run   build every input and check each span maps back to its text, with
              no model loaded. This is the part that can be verified anywhere.
"""
import argparse
import collections
import json
import os
import re
import time

import pandas as pd

MODEL = "jackboyla/glirel-large-v0"
OUT = "build/glirel_relations.parquet"
PARTS = "build/glirel_parts"
REPORT = "build/glirel_report.md"
CHECKPOINT = 2000          # sentences per saved part, so a lost session resumes
BATCH = 16                 # sentences per GPU call
MAX_TOKENS = 200           # longer sentences are skipped and counted, not cut
MAX_WIDTH = 12             # the model's max_width: a wider entity span cannot be scored
THRESHOLD = 0.10           # kept low on purpose: the cut-off is set later, on judged data
ENTITY_LABELS = ("PERSON", "GPE")

# Head -> label -> tail, phrased so the sentence "head label tail" reads true.
# "no relation" absorbs pairs that are merely named together.
LABELS = [
    "is the mother of", "is the father of", "is the son of", "is the daughter of",
    "is the wife of", "is the husband of", "is the brother of", "is the sister of",
    "is the grandson of", "is the teacher of", "is the charioteer of",
    "is an ally of", "is an enemy of", "is a friend of", "killed",
    "is the king of", "no relation",
]
# Labels whose tail must be a place; every other label is person to person.
PLACE_TAIL = {"is the king of"}

TOKEN = re.compile(r"[^\W_]+|[^\w\s]")


def tokenize(sentence):
    """Tokens for GLiREL and each token's character span.

    Words and punctuation are separate tokens, as spaCy writes them, so
    "Kunti’s" gives Kunti a token of its own and "Ayoda-Dhoumya" is two words.
    """
    found = list(TOKEN.finditer(sentence))
    return [m.group() for m in found], [(m.start(), m.end()) for m in found]


def token_span(offsets, start, end):
    """Character span -> (first token, last token) it covers, or None."""
    toks = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    return (toks[0], toks[-1]) if toks else None


def find_name(tokens, surface, hint):
    """Token span of a name in the sentence, or None.

    Phase 1's tok_start counts words in the NER file's own word list, which
    splits hyphens and drops footnote numbers, so a third of them do not line
    up with the sentence text. The name is found by its words instead, and the
    stored position only chooses between repeated occurrences.
    """
    want = [w.lower() for w in TOKEN.findall(surface) if w[0].isalnum()]
    if not want:
        return None
    words = [i for i, t in enumerate(tokens) if t[0].isalnum()]
    low = [tokens[i].lower() for i in words]
    hits = [k for k in range(len(low) - len(want) + 1) if low[k:k + len(want)] == want]
    if not hits:
        return None
    k = min(hits, key=lambda k: abs(k - hint))
    return words[k], words[k + len(want) - 1]


def build_inputs(mentions, coref):
    """One record per sentence with at least two entity spans."""
    m = mentions[mentions.label.isin(ENTITY_LABELS)]
    sents = mentions.drop_duplicates(["parva", "sent_id"]).set_index(["parva", "sent_id"])
    co = {k: g for k, g in coref.groupby(["parva", "sent_id"])} if len(coref) else {}
    stats = collections.Counter()
    out = []
    keys = sorted(set(zip(m.parva, m.sent_id)) | set(co))
    by_sent = {k: g for k, g in m.groupby(["parva", "sent_id"])}
    for key in keys:
        if key not in sents.index:
            continue
        s = sents.loc[key]
        tokens, offsets = tokenize(str(s.sentence))
        if len(tokens) > MAX_TOKENS:
            stats["skipped_long"] += 1
            continue
        ents, used = [], set()
        for r in by_sent.get(key, pd.DataFrame()).itertuples():
            span = find_name(tokens, r.surface, r.tok_start)
            if span is None:
                stats["mention_not_found"] += 1
                continue
            if span[1] - span[0] + 1 > MAX_WIDTH:
                stats["mention_too_wide"] += 1
                continue
            if used & set(range(span[0], span[1] + 1)):
                continue          # the same words, already taken by another mention
            ents.append({"span": list(span), "type": r.label, "surface": r.surface,
                         "name": r.surface, "via": "mention"})
            used.update(range(span[0], span[1] + 1))
        for r in co.get(key, pd.DataFrame()).itertuples():
            if r.person != 3:
                continue          # I and you inside speech: a different problem, noted in 4b
            span = token_span(offsets, int(r.start), int(r.end))
            if span is None or used & set(range(span[0], span[1] + 1)):
                continue          # the span is already a named mention
            if span[1] - span[0] + 1 > MAX_WIDTH:
                stats["coref_too_wide"] += 1
                continue          # a long description; the model cannot score it
            ents.append({"span": list(span), "type": "PERSON",
                         "surface": r.surface, "name": r.canonical, "via": "coref"})
            used.update(range(span[0], span[1] + 1))
        if len(ents) < 2:
            continue
        stats["sentences"] += 1
        stats["entities"] += len(ents)
        stats["coref_entities"] += sum(e["via"] == "coref" for e in ents)
        out.append({"parva": key[0], "sent_id": int(key[1]),
                    "adhyaya_ordinal": s.adhyaya_ordinal, "cited": bool(s.chapter_cited),
                    "tokens": tokens, "entities": ents})
    return out, stats


def check_spans(records):
    """Every span must read back as its surface, mentions and coref alike."""
    bad = []
    for rec in records:
        for e in rec["entities"]:
            text = " ".join(rec["tokens"][e["span"][0]:e["span"][1] + 1])
            if norm(text) != norm(e["surface"]):
                bad.append((rec["parva"], rec["sent_id"], e["via"], e["surface"], text))
    return bad


def norm(s):
    return re.sub(r"[\W_]+", "", s).lower()


def pos_of(rel, side):
    """GLiREL's span for head or tail, as the inclusive [start, end] it was given.

    glirel 1.2.1 takes entity spans with an inclusive end and returns them with
    an exclusive one: model.py writes 'head_pos': [start, end + 1]. Looked up as
    returned, no relation matches any entity and every row is silently lost.
    """
    p = rel.get("%s_pos" % side)
    return None if p is None else [p[0], p[1] - 1]


def load_model(model_name, device):
    """GLiREL, loaded around a version clash in its own loader.

    glirel 1.2.1's _from_pretrained requires `proxies` and `resume_download`,
    which huggingface_hub 1.x no longer passes, so GLiREL.from_pretrained fails
    with a TypeError. The loader only forwards them to a download it skips when
    the files are already on disk. So the two files are fetched here with the
    current API, and the loader is called on that folder.
    """
    from glirel import GLiREL
    local = model_name
    if not os.path.isdir(model_name):
        from huggingface_hub import snapshot_download
        local = snapshot_download(model_name,
                                  allow_patterns=["pytorch_model.bin", "glirel_config.json"])
    return GLiREL._from_pretrained(model_id=local, revision=None, cache_dir=None,
                                   force_download=False, proxies=None, resume_download=False,
                                   local_files_only=True, token=None, map_location=device)


def run(records, model_name, done_parts):
    import torch
    model = load_model(model_name, "cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(PARTS, exist_ok=True)
    shown = False
    returned = kept = 0
    rows, t0, done_now = [], time.time(), 0
    todo = [i for i in range(len(records)) if i // CHECKPOINT not in done_parts]
    for b in range(0, len(todo), BATCH):
        idx = todo[b:b + BATCH]
        batch = [records[i] for i in idx]
        ners = [[[e["span"][0], e["span"][1], e["type"], e["surface"]] for e in rec["entities"]]
                for rec in batch]
        outs = model.batch_predict_relations([rec["tokens"] for rec in batch], LABELS,
                                             threshold=THRESHOLD, ner=ners, top_k=1)
        for i, rec, rels in zip(idx, batch, outs):
            new = extract(rec, rels)
            rows += new
            kept += len(new)
            returned += sum(r["label"] != "no relation" for r in rels)
            if not shown and rels:
                print("first raw GLiREL output:", rels[0], flush=True)
                shown = True
            # A position mismatch drops every relation without an error. If the
            # model has returned relations and none survived, stop and say so.
            if returned >= 50 and kept == 0:
                raise SystemExit("%d relations returned and none matched an entity span. "
                                 "Check pos_of() against this glirel version's output." % returned)
            done_now += 1
            if (i + 1) % CHECKPOINT == 0 or i == len(records) - 1:
                part = i // CHECKPOINT
                pd.DataFrame(rows).to_parquet("%s/part_%04d.parquet" % (PARTS, part))
                print("sentences %d/%d, part %d saved, %d relations in it, %.1f sentences/s"
                      % (i + 1, len(records), part, len(rows),
                         done_now / max(time.time() - t0, 1e-6)), flush=True)
                rows = []


def extract(rec, rels):
    """The kept rows for one sentence: a real label, both spans known, types right."""
    by_span = {tuple(e["span"]): e for e in rec["entities"]}
    rows = []
    for rel in rels:
        if rel["label"] == "no relation":
            continue
        h = by_span.get(tuple(pos_of(rel, "head") or ()))
        t = by_span.get(tuple(pos_of(rel, "tail") or ()))
        if h is None or t is None or h["name"] == t["name"]:
            continue
        want_place = rel["label"] in PLACE_TAIL
        if h["type"] != "PERSON" or (t["type"] == "GPE") != want_place:
            continue          # type constraints, applied after the model
        rows.append({"parva": rec["parva"], "sent_id": rec["sent_id"],
                     "adhyaya_ordinal": rec["adhyaya_ordinal"], "cited": rec["cited"],
                     "head": h["name"], "head_surface": h["surface"], "head_via": h["via"],
                     "head_span": h["span"], "label": rel["label"],
                     "tail": t["name"], "tail_surface": t["surface"], "tail_via": t["via"],
                     "tail_span": t["span"], "score": float(rel["score"])})
    return rows


def report(stats, frame, args):
    lines = ["# Phase 4r. GLiREL zero-shot relation extraction", "",
             "Generated by `src/phase4r_glirel.py`. Candidates only, never asserted. "
             "Scores are not probabilities; no threshold has been judged yet.", "",
             "- Model: `%s`, top label per pair, kept at score >= %.2f" % (args.model, THRESHOLD),
             "- Sentences with two or more entity spans: %d (%d skipped as longer than %d tokens)"
             % (stats["sentences"], stats["skipped_long"], MAX_TOKENS),
             "- Entity spans: %d, of which from coreference: %d"
             % (stats["entities"], stats["coref_entities"]),
             "- Relations kept: %d" % len(frame), ""]
    if len(frame):
        lines += ["| label | relations | score >= 0.5 | median score |", "|---|---|---|---|"]
        for lab, g in frame.groupby("label"):
            lines.append("| %s | %d | %d | %.2f |" % (lab, len(g), (g.score >= 0.5).sum(),
                                                     g.score.median()))
    open(REPORT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=0, help="first N sentences only: a smoke test")
    ap.add_argument("--dry-run", action="store_true", help="build inputs and check spans, no model")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    mentions = pd.read_parquet("build/mentions.parquet")
    coref = pd.read_parquet("build/coref.parquet") if os.path.exists("build/coref.parquet") \
        else pd.DataFrame()
    records, stats = build_inputs(mentions, coref)
    if args.limit:
        records = records[:args.limit]
    bad = check_spans(records)
    print("sentences %d, entity spans %d (coref %d), long skipped %d, names not found %d, "
          "spans wider than %d skipped %d, span mismatches %d"
          % (len(records), sum(len(r["entities"]) for r in records),
             sum(e["via"] == "coref" for r in records for e in r["entities"]),
             stats["skipped_long"], stats["mention_not_found"], MAX_WIDTH,
             stats["mention_too_wide"] + stats["coref_too_wide"], len(bad)), flush=True)
    for b in bad[:10]:
        print("  span mismatch:", b)
    if args.dry_run:
        return

    done = set()
    if os.path.isdir(PARTS) and not args.limit:
        done = {int(f[5:9]) for f in os.listdir(PARTS) if f.startswith("part_")}
        if done:
            print("resuming: parts %s already saved" % sorted(done), flush=True)
    run(records, args.model, done)
    parts = sorted(os.listdir(PARTS))
    frame = pd.concat([pd.read_parquet(os.path.join(PARTS, f)) for f in parts],
                      ignore_index=True) if parts else pd.DataFrame()
    frame.to_parquet(OUT)
    report(stats, frame, args)
    json.dump({"model": args.model, "limit": args.limit, "rows": len(frame)},
              open("build/glirel_manifest.json", "w"))


if __name__ == "__main__":
    main()
