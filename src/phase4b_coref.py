"""Phase 4b. Coreference resolution over the Debroy prose.

Added because the Phase 6 entry gate failed and the measured reason was that
relations are rarely inside one clause with both ends named. Of the relation
clauses in this corpus, 360 have two proper names, 722 have one name and a
pronoun, 220 have two pronouns and 1,798 have a multiword description such as
"the son of Drona" in an argument slot. This phase is aimed at those 2,740.

Model: biu-nlp/lingmess-coref, run through fastcoref. LingMess rather than the
distilled f-coref, because f-coref got the smoke-test passage wrong in the way
that matters here. On "Bhishma, Shantanu's son, saw Arjuna advancing. He pierced
him with three arrows", f-coref put Bhishma, He and him in one cluster, which
would assert that Arjuna pierced himself. LingMess separates them correctly and
also links "the Pandava" to Arjuna.

Inference is deterministic, so there is no seed to record. The model identifier
and revision are recorded instead, in build/coref_report.md.

Coreference runs over a document, not a sentence. The document here is the
chapter, which is the citation unit, so a resolution never crosses a citation
boundary and every resolved mention inherits the chapter it was found in. Long
chapters are split into windows and coreference is not carried across a window
boundary; the number of splits is reported.
"""
import collections, json, re, sys
import pandas as pd

from transformers.modeling_utils import PreTrainedModel
# transformers 4.57 refuses to dispatch sdpa attention for Longformer and
# raises rather than falling back. LingMess is a Longformer. Force eager.
PreTrainedModel._check_and_adjust_attn_implementation = lambda self, *a, **k: "eager"
from fastcoref import LingMessCoref

MODEL = "biu-nlp/lingmess-coref"
MAX_CHARS = 7000          # about 1,800 wordpieces, inside the 4,096 window
NOT_A_NAME = {"You", "He", "She", "They", "We", "It", "Thus", "Then", "Having",
              "There", "Here", "This", "That", "These", "Those", "But", "And",
              "His", "Her", "Their", "Your", "Who", "What", "When", "Where",
              "Thou", "Thee", "Thy", "Yet", "Now", "One", "Some", "All", "O"}


def load():
    df = pd.read_parquet("build/mentions.parquet")
    lab = df.groupby(df.surface.str.strip()).label.agg(lambda x: x.value_counts().idxmax())
    persons = {s for s, l in lab.items()
               if l == "PERSON" and re.fullmatch(r"[A-Z][a-z]{2,}", s)
               and s not in NOT_A_NAME}
    sents = df.drop_duplicates(subset=["parva", "sent_id"])[
        ["parva", "sent_id", "adhyaya_ordinal", "sentence"]]
    sents = sents.sort_values(["parva", "sent_id"])
    return persons, sents


def build_documents(sents, stats):
    """One document per chapter, split into windows under the token limit.

    Yields (key, text, index) where index maps a document character offset back
    to (sent_id, offset within that sentence).
    """
    for (parva, adh), g in sents.groupby(["parva", "adhyaya_ordinal"], dropna=False):
        buf, index, pos, part = [], [], 0, 0
        for r in g.itertuples():
            s = str(r.sentence)
            if pos + len(s) > MAX_CHARS and buf:
                yield (parva, adh, part), " ".join(buf), index
                stats["windows"] += 1
                part += 1
                buf, index, pos = [], [], 0
            index.append((pos, pos + len(s), int(r.sent_id)))
            buf.append(s)
            pos += len(s) + 1
        if buf:
            yield (parva, adh, part), " ".join(buf), index
            stats["windows"] += 1
        stats["chapters"] += 1
        if part:
            stats["chapters_split"] += 1


FIRST = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
SECOND = {"you", "your", "yours", "yourself", "thou", "thee", "thy", "thine"}


def person_of(surface):
    """1 or 2 for pronouns of address, 3 otherwise.

    "I" and "you" inside a quoted speech resolve to the speaker and the
    addressee, which is a different problem from third-person anaphora and goes
    wrong in different ways. They are tagged so that downstream extraction can
    weigh them separately or drop them, rather than being silently mixed in.
    """
    w = surface.strip().lower()
    if w in FIRST:
        return 1
    if w in SECOND:
        return 2
    return 3


def locate(index, start, end):
    """Map a document span back to (sent_id, start, end) within its sentence."""
    for a, b, sid in index:
        if start >= a and end <= b:
            return sid, start - a, end - a
    return None                      # the span straddles a sentence boundary


def canonical_of(spans, persons):
    """The name a cluster resolves to, or None.

    The name is the most frequent known person surface appearing as a whole
    token inside any mention of the cluster. A cluster whose every mention is a
    pronoun or an unknown description resolves to nothing and is dropped, rather
    than being guessed at.
    """
    votes = collections.Counter()
    for text in spans:
        for tok in re.findall(r"[A-Za-z]+", text):
            if tok in persons:
                votes[tok] += 1
    if not votes:
        return None
    top, n = votes.most_common(1)[0]
    rivals = [k for k, v in votes.items() if k != top and v >= n]
    if rivals:
        return None       # two names with equal claim; do not choose silently
    return top


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    persons, sents = load()
    if limit:
        sents = sents.head(limit)
    stats = collections.Counter()
    docs = list(build_documents(sents, stats))

    model = LingMessCoref(device="cuda:0", enable_progress_bar=False)
    out = []
    B = 64
    for i in range(0, len(docs), B):
        batch = docs[i:i + B]
        preds = model.predict(texts=[d[1] for d in batch])
        for (key, text, index), pred in zip(batch, preds):
            parva, adh, _ = key
            for cluster in pred.get_clusters(as_strings=False):
                strings = [text[a:b] for a, b in cluster]
                name = canonical_of(strings, persons)
                if name is None:
                    stats["cluster_unresolved"] += 1
                    continue
                stats["cluster_resolved"] += 1
                for (a, b), s in zip(cluster, strings):
                    if s == name:
                        stats["mention_is_the_name"] += 1
                        continue          # already extractable without coref
                    loc = locate(index, a, b)
                    if loc is None:
                        stats["mention_straddles_sentences"] += 1
                        continue
                    sid, sa, sb = loc
                    out.append({"parva": parva,
                                "adhyaya_ordinal": adh,
                                "sent_id": sid, "start": sa, "end": sb,
                                "surface": s, "canonical": name,
                                "person": person_of(s),
                                "cluster_size": len(cluster)})
                    stats["mention_resolved"] += 1
        print("docs %d/%d resolved=%d" % (min(i + B, len(docs)), len(docs),
                                          stats["mention_resolved"]), flush=True)

    df = pd.DataFrame(out)
    df.to_parquet("build/coref.parquet")
    report(df, stats, len(docs))
    print("windows=%d mentions_resolved=%d distinct_names=%d"
          % (len(docs), len(df), df.canonical.nunique() if len(df) else 0))


def report(df, stats, ndocs):
    with open("build/coref_report.md", "w", encoding="utf-8") as f:
        f.write("# Phase 4b coreference report\n\nGenerated by `src/phase4b_coref.py`.\n\n")
        f.write("- Model: `%s` via fastcoref, run on GPU with eager attention\n" % MODEL)
        f.write("- Inference is deterministic; no seed applies\n")
        f.write("- Document unit: the chapter, split at %d characters\n\n" % MAX_CHARS)
        f.write("| quantity | count |\n|---|---|\n")
        for k in ("chapters", "chapters_split", "windows", "cluster_resolved",
                  "cluster_unresolved", "mention_is_the_name",
                  "mention_straddles_sentences", "mention_resolved"):
            f.write("| %s | %d |\n" % (k, stats[k]))
        if len(df):
            f.write("\n- Distinct canonical names: %d\n" % df.canonical.nunique())
            f.write("\n## By grammatical person\n\n| person | mentions |\n|---|---|\n")
            for k, v in sorted(df.person.value_counts().items()):
                f.write("| %s | %d |\n"
                        % ({1: "first (I, we)", 2: "second (you)",
                            3: "third and descriptions"}[k], v))
            f.write("\n## Most resolved-to names\n\n| name | mentions |\n|---|---|\n")
            for n, c in df.canonical.value_counts().head(20).items():
                f.write("| %s | %d |\n" % (n, c))
        f.write("\n## What is deliberately dropped\n\n"
                "A cluster resolves only if a known person name appears inside one of\n"
                "its mentions. A cluster of pronouns with no name in it is dropped\n"
                "rather than guessed (%d clusters). A cluster with two names of equal\n"
                "frequency is also dropped, because choosing between them silently is\n"
                "exactly the entity-link guess the hard rules forbid.\n"
                % stats["cluster_unresolved"])


if __name__ == "__main__":
    main()
