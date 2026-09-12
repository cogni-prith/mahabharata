"""Phase 1. Mention layer and citation spine.

Every annotated span becomes a Mention anchored to a Passage at
(parvan, adhyaya). Writes build/mentions.parquet, out/passages.ttl and
build/phase1_report.md.

Two joins happen here and both are measured in the report.

Sentence to chapter: the corpus was derived from these PDFs, so a normalized
corpus sentence should occur verbatim in the normalized PDF stream. Sentences
are searched in order with a cursor that only moves forward, which resolves
repeated sentences to the right occurrence and keeps the whole pass linear.

Master to BIO: on normalized sentence text, never on sent_id. See DECISIONS.md
D2. The index join drops four parvas silently.
"""
import bisect, csv, json, os, re, sys
from collections import defaultdict, deque

import pandas as pd

csv.field_size_limit(10 ** 9)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
OUT = os.path.join(ROOT, "out")
LLM_DIR = os.path.join(ROOT, "Mahabharata LLM annotated-20260801T163642Z-1-001",
                       "Mahabharata LLM annotated")
HUM_DIR = os.path.join(ROOT, "Mahabharat_Human_Annotated_Data-20260801T163709Z-1-001",
                       "Mahabharat_Human_Annotated_Data")
MASTER = os.path.join(ROOT, "mahabharata_ner_by_sentence_master (2).csv")

# Provisional base IRI. Nothing is published, so this does not resolve and the
# choice is not final; see the report.
BASE = "http://example.org/mbh/"

LLM_FILES = {
    "01_Adi": "AdiParva_tagged_filled.csv", "02_Sabha": "sabha_parva_tagged.csv",
    "03_Aranyaka": "Aranyaka_tagged.csv", "04_Virata": "virata_parva_tagged.csv",
    "05_Udyoga": "udoyogaparva_filled LLM.csv", "06_Bhishma": "Bhisma_tagged.csv",
    "07_Drona": "DronaParva_tagged.csv", "08_Karna": "KarnaParva_tagged_filled.csv",
    "09_Shalya": "ShalyaParva_tagged_filled.csv", "10_Souptika": "Souptika_tagged.csv",
    "11_Stri": "stri_parva_tagged.csv", "12_Shanti": "shanti_parva.csv",
    "13_Anushasana": "anushasana_tagged.csv", "14_Ashvamedhika": "AshwamedhaParva_tagged.csv",
    "15_Ashramavasika": "AshramaVasika_parva_filled LLM.csv", "16_Mausala": "Mausala_tagged.csv",
    "17_Mahaprasthanika": "MahaPrasthanika_tagged_filled.csv",
    "18_Svargarohana": "Savargarohana_Parva_filled.csv",
}
# Human files. Mahaprasthanika and Svargarohana are listed for completeness but
# are effectively unrevised machine output; see DECISIONS.md D6. The tier field
# records which file a mention came from, not how much revision it received.
HUM_FILES = {
    "06_Bhishma": "Bhisma_tagged-checked.csv", "08_Karna": "KarnaParvaRe.csv",
    "09_Shalya": "ShalyaParvaRe.csv", "10_Souptika": "Souptika_tagged-checked.csv",
    "15_Ashramavasika": "AshramaVasika_parva_filled.csv",
    "16_Mausala": "Mausala_tagged-checked.csv",
    "17_Mahaprasthanika": "MahaPrasthanika_tagged_filled.csv",
    "18_Svargarohana": "Savargarohana_Parva_filled.csv",
}


def normalize(text):
    """Must match build_spine.normalize and preprocess_pdf.py."""
    text = re.sub(r"['’]s\b", "", str(text), flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(w.lower() for w in text.split(" ") if not w.isdigit())


def read_bio(path):
    """sentence_no -> (sentence, [(surface, label, tok_start, tok_end)]).

    Outside is spelled "" in the human files and "O(Clear Entity)" in the LLM
    files. An I- tag with no open span starts one: the tagger emits those and
    dropping them would silently lose mentions.
    """
    sents, spans = {}, defaultdict(list)
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        idx = defaultdict(int)
        cur = {}
        for row in csv.DictReader(fh):
            sn = row.get("sentence_no")
            if sn is None:
                continue
            if sn not in sents:
                sents[sn] = row.get("sentence")
            word = (row.get("word") or "").strip()
            tag = (row.get("tag") or "").strip()
            pos = idx[sn]
            idx[sn] += 1
            if not word:
                continue
            if tag.startswith("B-") or (tag.startswith("I-") and sn not in cur):
                if sn in cur:
                    spans[sn].append(cur.pop(sn))
                cur[sn] = [word, tag[2:], pos, pos + 1]
            elif tag.startswith("I-") and sn in cur and cur[sn][1] == tag[2:]:
                cur[sn][0] += " " + word
                cur[sn][3] = pos + 1
            else:
                if sn in cur:
                    spans[sn].append(cur.pop(sn))
        for sn, s in cur.items():
            spans[sn].append(s)
    return sents, spans


def bio_index(sents):
    """normalized sentence -> queue of sentence_no, so repeated sentences are
    consumed in document order instead of all binding to the first match."""
    idx = defaultdict(deque)
    for sn, txt in sents.items():
        k = normalize(txt)
        if k:
            idx[k].append(sn)
    return idx


def main():
    os.makedirs(BUILD, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    spine = json.load(open(os.path.join(BUILD, "spine.json"), encoding="utf-8"))
    master = pd.read_csv(MASTER).dropna(subset=["sentence"])

    rows, report = [], []
    stats = defaultdict(lambda: defaultdict(int))
    labels = defaultdict(int)

    for parva in sorted(LLM_FILES):
        st = stats[parva]
        chapters = spine[parva]["chapters"]
        offsets = [c["norm_offset"] for c in chapters]
        text = open(os.path.join(BUILD, "pdftext", parva + ".txt"),
                    encoding="utf-8").read()

        mm = master[master.parva == parva].sort_values("sent_id")
        st["master_sentences"] = len(mm)

        llm_sents, llm_spans = read_bio(os.path.join(LLM_DIR, LLM_FILES[parva]))
        llm_idx = bio_index(llm_sents)
        if parva in HUM_FILES:
            hum_sents, hum_spans = read_bio(os.path.join(HUM_DIR, HUM_FILES[parva]))
            hum_idx = bio_index(hum_sents)
        else:
            hum_sents, hum_spans, hum_idx = {}, {}, {}

        cursor = 0
        for sent_id, sentence, ents in zip(mm.sent_id, mm.sentence, mm.entities):
            key = normalize(sentence)
            if not key:
                continue

            # Locate in the PDF stream. Forward first, then a bounded look-back,
            # because master sentence order and print order disagree occasionally.
            pos = text.find(key, cursor)
            if pos < 0:
                pos = text.find(key)
                st["located_out_of_order"] += 1 if pos >= 0 else 0
            if pos >= 0:
                cursor = pos + len(key)
                # Anchor on the midpoint, not the start. Drop caps put the
                # decorative initial before the chapter head in the extracted
                # stream, so a first-sentence-of-chapter reads
                # "’ ‘S Chapter 1152(2) anjaya said" and starts before the
                # boundary it belongs after. Anchoring on the start put every
                # such sentence one chapter early, which was all of Karna and
                # all of Shalya. The midpoint follows the bulk of the sentence.
                ci = bisect.bisect_right(offsets, pos + len(key) // 2) - 1
                st["located"] += 1
            else:
                ci = -1
                st["unlocated"] += 1

            if ci >= 0:
                ch = chapters[ci]
                passage = "%sp/%s/%d" % (BASE, parva, ch["ordinal"])
                adhyaya, cited = ch["ordinal"], True
                st["chapter_anchored"] += 1
            else:
                ch = None
                passage = "%sp/%s" % (BASE, parva)
                adhyaya, cited = None, False
                st["parva_anchored"] += 1

            # Weak prior only, per DECISIONS.md D5. Never treated as resolution.
            prior = {}
            try:
                for e in json.loads(ents) if isinstance(ents, str) else []:
                    prior[e.get("text")] = e.get("canonical_id")
            except (ValueError, TypeError):
                pass

            for tier, idx, spans in (("human", hum_idx, hum_spans),
                                     ("llm", llm_idx, llm_spans)):
                q = idx.get(key) if idx else None
                if not q:
                    continue
                sn = q.popleft()
                st["joined_" + tier] += 1
                for surface, label, a, b in spans.get(sn, []):
                    labels[label] += 1
                    rows.append({
                        "mention_id": "%sm/%s/%d/%s/%d" % (BASE, parva, sent_id, tier, a),
                        "parva": parva, "sent_id": int(sent_id),
                        "passage_id": passage,
                        "adhyaya_ordinal": adhyaya,
                        "adhyaya_printed_global": ch["printed_global"] if ch else None,
                        "adhyaya_printed_relative": ch["printed_relative"] if ch else None,
                        "chapter_cited": cited,
                        "tier": tier,
                        "surface": surface, "label": label,
                        "tok_start": a, "tok_end": b,
                        "canonical_id_prior": prior.get(surface),
                        "sentence": sentence,
                    })
                    st["mentions_" + tier] += 1
        print("%-20s sents=%6d located=%6d unloc=%5d chap=%6d parva=%5d "
              "j_hum=%6d j_llm=%6d men=%7d"
              % (parva, st["master_sentences"], st["located"], st["unlocated"],
                 st["chapter_anchored"], st["parva_anchored"], st["joined_human"],
                 st["joined_llm"], st["mentions_human"] + st["mentions_llm"]))

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(BUILD, "mentions.parquet"), index=False)
    write_passages(spine)
    write_report(df, stats, labels, spine)
    print("\nmentions: %d   parquet: build/mentions.parquet" % len(df))


def write_passages(spine):
    with open(os.path.join(OUT, "passages.ttl"), "w", encoding="utf-8") as fh:
        fh.write("@prefix mbh: <%s> .\n" % BASE)
        fh.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n")
        fh.write("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n")
        for parva, v in sorted(spine.items()):
            fh.write("mbh:p\\/%s a mbh:Parvan ;\n    rdfs:label %s ;\n"
                     "    mbh:chapterCount %d .\n\n"
                     % (parva, json.dumps(parva.split("_", 1)[1]), v["found"]))
            for c in v["chapters"]:
                fh.write("mbh:p\\/%s\\/%d a mbh:Passage ;\n" % (parva, c["ordinal"]))
                fh.write("    mbh:inParvan mbh:p\\/%s ;\n" % parva)
                fh.write("    mbh:adhyaya %d ;\n" % c["ordinal"])
                fh.write("    mbh:printedGlobalChapter %d ;\n" % c["printed_global"])
                fh.write("    mbh:printedRelativeChapter %d ;\n" % c["printed_relative"])
                if c.get("declared_shlokas"):
                    fh.write("    mbh:declaredShlokas %d ;\n" % c["declared_shlokas"])
                fh.write("    mbh:cited true .\n\n")


def write_report(df, stats, labels, spine):
    tot = lambda k: sum(s[k] for s in stats.values())
    lines = ["# Phase 1 run report", "",
             "Generated by `src/phase1_mentions.py`. Counts are measured, not estimated.",
             "", "## Gate", ""]
    n = len(df)
    ch = int(df.chapter_cited.sum()) if n else 0
    lines += [
        "- Mentions emitted: **%d**" % n,
        "- Resolving to chapter level `(parvan, adhyaya)`: **%d (%.2f%%)**" % (ch, 100.0 * ch / max(n, 1)),
        "- Resolving to parva level only, `adhyaya` null and `cited: false`: **%d (%.2f%%)**"
        % (n - ch, 100.0 * (n - ch) / max(n, 1)),
        "- Chapter spine: **%d of %d** BORI chapters recovered."
        % (sum(v["found"] for v in spine.values()),
           sum(v["bori_chapters"] for v in spine.values())),
        "",
        "## Sentence location against the PDF stream", "",
        "| parva | master sents | located | unlocated | chapter | parva only |",
        "|---|---|---|---|---|---|",
    ]
    for p in sorted(stats):
        s = stats[p]
        lines.append("| %s | %d | %d | %d | %d | %d |" % (
            p, s["master_sentences"], s["located"], s["unlocated"],
            s["chapter_anchored"], s["parva_anchored"]))
    lines += ["| **total** | %d | %d | %d | %d | %d |" % (
        tot("master_sentences"), tot("located"), tot("unlocated"),
        tot("chapter_anchored"), tot("parva_anchored")), ""]

    lines += ["## Substrate join", "",
              "| parva | joined human | joined LLM | mentions human | mentions LLM |",
              "|---|---|---|---|---|"]
    for p in sorted(stats):
        s = stats[p]
        lines.append("| %s | %d | %d | %d | %d |" % (
            p, s["joined_human"], s["joined_llm"], s["mentions_human"], s["mentions_llm"]))
    lines += ["| **total** | %d | %d | %d | %d |" % (
        tot("joined_human"), tot("joined_llm"),
        tot("mentions_human"), tot("mentions_llm")), ""]

    lines += ["## Label inventory", "", "| label | mentions |", "|---|---|"]
    for k, v in sorted(labels.items(), key=lambda x: -x[1]):
        lines.append("| %s | %d |" % (k, v))
    lines += ["", "## Rejections and known gaps", "",
              "- Sentences not found in the PDF stream are anchored at parva level "
              "with `adhyaya` null. They are not assigned a neighbouring chapter.",
              "- The base IRI `%s` is provisional and does not resolve. "
              "Nothing is published." % BASE,
              "- `tier` records the source file, not the amount of human revision. "
              "Mahaprasthanika and Mausala sit in the human folder but are "
              "effectively unrevised machine output (DECISIONS.md D6).",
              ""]
    with open(os.path.join(BUILD, "phase1_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
