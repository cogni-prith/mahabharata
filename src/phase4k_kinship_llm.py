"""Phase 4k. Kinship extraction with an open language model, over family passages.

The graph misses family facts the text states plainly, because they are stated
in ways a one-sentence extractor cannot read:

    "The chief of the Yadus was named Shura and he was Vasudeva's father.
     His daughter was named Pritha..."                       (01_Adi 104)
    "O Pritha! ... You are Shura's beloved daughter."         (03_Aranyaka 287)

This reads every passage with a kinship word and a person in it, three
sentences wide, and asks an instruction-tuned model for the family relations
the passage itself states, each with an exact quote. It runs on a Kaggle GPU.

Nothing here is asserted. Output is a candidate layer, build/kinship_llm.parquet,
and two guards run before a row is kept:
  - both names must occur in the passage, or be the resolved name of a
    pronoun the coreference step marked in it, and a row resting on such a
    resolved name is flagged via_hint, since coreference is sometimes wrong;
  - the quote must occur in the passage.
A model that invents a name or a quote is caught here, not trusted. Accuracy is
still unmeasured until a judged sample exists.

Run: python src/phase4k_kinship_llm.py [--limit N] [--dry-run] [--model NAME] [--fp16]
  --dry-run  build every passage and prompt, with no model. Verifiable anywhere.
"""
import argparse
import collections
import json
import os
import re
import time

import pandas as pd

MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT = "build/kinship_llm.parquet"
RAW_OUT = "build/kinship_llm_raw.parquet"
PARTS = "build/kinship_parts"
REPORT = "build/kinship_llm_report.md"
CHECKPOINT = 500       # passages per saved part, so a lost session resumes
BATCH = 8
MAX_NEW_TOKENS = 320
MAX_SENTENCES = 5      # sentences per passage; a run of family sentences longer than this is split

KIN = re.compile(r"\b(father|mother|sons?|daughters?|wife|wives|husband|brothers?|sisters?|"
                 r"grand(?:son|daughter|father|mother)s?|uncle|aunt|nephew|niece|begot|"
                 r"begotten|sired|womb|offspring|married|marry|born|gave birth|giving birth|"
                 r"give birth|gives birth|birth|adoptive|adopted|foster)\b", re.I)
# A kinship word used only to address someone: a teacher's "Veda, my son!" or
# "Come here, my son" made the teacher a father twice on the third Kaggle run.
ADDRESS = re.compile(r"\b(my|o|dear)\s+(dear\s+)?(sons?|daughters?|child|children|boy|girl)\b", re.I)
# Titles and forms of address the model returned as names: "Bhagavan".
TITLES = {"bhagavan", "lord", "king", "queen", "rishi", "brahmana", "maharshi", "devarshi",
          "sage", "god", "goddess", "father", "mother", "son", "daughter", "preceptor"}
TYPES = {"father", "mother", "parent", "adoptive_father", "adoptive_mother",
         "divine_father", "spouse", "sibling"}
PARENT_TYPES = TYPES - {"spouse", "sibling"}
SPOUSE = re.compile(r"\b(wife|wives|husband|married|marry|marriage|wed|wedded|bride|consort)\b", re.I)
SIBLING = re.compile(r"\b(brothers?|sisters?|siblings?|twins?)\b", re.I)
# The model's own words for a type, mapped onto the allowed ones. On the first
# Kaggle run it answered "son" for "Ugrashrava, the son of Lomaharshana", which
# is right and was being thrown away. "A is the son of B" keeps a as the child
# and b as the parent, so the direction carries over unchanged.
TYPE_WORDS = {
    "son": "parent", "daughter": "parent", "child": "parent", "son_of": "parent",
    "daughter_of": "parent", "child_of": "parent", "offspring": "parent",
    "father_of": "father", "mother_of": "mother",
    "wife": "spouse", "husband": "spouse", "married": "spouse", "consort": "spouse",
    "brother": "sibling", "sister": "sibling", "siblings": "sibling",
    "foster_father": "adoptive_father", "foster_mother": "adoptive_mother",
    "adopted_father": "adoptive_father", "adopted_mother": "adoptive_mother",
    "divine_parent": "divine_father", "god_father": "divine_father",
}

SYSTEM = """You extract family relations from one passage of an English translation of the Mahabharata.

Report only relations that THIS passage states or directly implies. Never use your own knowledge of the epic, even when you are sure of it. If the passage states no family relation, return an empty list.

Relation types:
- father, mother, parent: B is the father / mother / parent of A (use parent when the passage does not say which)
- adoptive_father, adoptive_mother: B raised or adopted A
- divine_father: a god B fathered A
- spouse: A and B are married
- sibling: A and B are brothers or sisters

Rules:
- Two people named in the same passage are NOT related unless the passage says how. Killing, fighting, teaching, learning from, speaking to or praising someone is never a family relation.
- spouse only when the passage says wife, husband, married or wedded. sibling only when it says brother, sister or twin.
- "a" and "b" are personal names exactly as written, never a description ("his son", "ten thousand sons", "Satyavati's son").
- Address forms are not facts. "O son of Kunti", "O father", "O daughter" only address someone; report them only if the passage also makes clear who is meant and the relation is literal.
- "his", "her", "he", "she", "you" may stand for a person. Use the HINTS when they give the person, and use the names exactly as written in the passage or the hints.
- "evidence" must be an exact quote copied from the passage that contains BOTH people, by name or by the pronoun that stands for them, and the words that show the relation. Keep it short, but never cut a person out of it.

Answer with JSON only, no other text:
{"relations": [{"a": "<child, or first person>", "b": "<parent, or second person>", "type": "<one type>", "evidence": "<exact quote>"}]}

"type" must be exactly one of: father, mother, parent, adoptive_father, adoptive_mother, divine_father, spouse, sibling. For a child, put the child in "a" and the parent in "b"; never use "son" or "daughter" as a type.

Examples.
Passage: "Ugrashrava, the son of Lomaharshana, approached them."
{"relations": [{"a": "Ugrashrava", "b": "Lomaharshana", "type": "father", "evidence": "Ugrashrava, the son of Lomaharshana"}]}
Passage: "Ruru had two sons, named Shunaka and Vitihavya."
{"relations": [{"a": "Shunaka", "b": "Ruru", "type": "father", "evidence": "Ruru had two sons, named Shunaka and Vitihavya"}, {"a": "Vitihavya", "b": "Ruru", "type": "father", "evidence": "Ruru had two sons, named Shunaka and Vitihavya"}]}
Passage: "Karna was killed by Arjuna, who had learnt the use of weapons from Drona."
{"relations": []}"""


def norm(s):
    return re.sub(r"\s+", " ", s.replace("’", "'").replace("‘", "'")
                  .replace("“", '"').replace("”", '"')).strip().lower()


def build_passages(mentions, coref):
    """Passages of family sentences with their neighbours, never crossing a chapter.

    Each candidate sentence (a kinship word and a person) wants the sentence
    before and after it as context. Family sentences cluster, in genealogy
    chapters especially, so one window per sentence read the same lines two or
    three times and returned the same fact as often. Instead every sentence
    wanted by some candidate is marked, runs of marked sentences become
    passages of at most MAX_SENTENCES, and nothing is read twice. A chapter
    boundary always ends a passage, so a relation's chapter is never wrong.
    """
    sents = mentions.drop_duplicates(["parva", "sent_id"]).sort_values(["parva", "sent_id"])
    sents = sents[["parva", "sent_id", "adhyaya_ordinal", "chapter_cited", "sentence"]]
    persons = mentions[mentions.label == "PERSON"]
    names = persons.groupby(["parva", "sent_id"]).surface.agg(lambda x: sorted(set(x)))
    hints = {}
    if len(coref):
        for (p, s), g in coref.groupby(["parva", "sent_id"]):
            hints[(p, s)] = sorted({(r.surface, r.canonical) for r in g.itertuples()})
    out, stats = [], collections.Counter()
    sents = sents.assign(chapter=sents.adhyaya_ordinal.fillna(-1))
    for (parva, _), g in sents.groupby(["parva", "chapter"], sort=False):
        rows = list(g.itertuples())
        cand = [KIN.search(str(r.sentence)) is not None
                and ((parva, r.sent_id) in names.index or (parva, r.sent_id) in hints) for r in rows]
        wanted = [any(cand[max(0, i - 1):i + 2]) for i in range(len(rows))]
        stats["candidate_sentences"] += sum(cand)
        runs, run = [], []
        for i, w in enumerate(wanted):
            if w:
                run.append(i)
            if run and (not w or i == len(rows) - 1):
                runs.append(run)
                run = []
        for run in runs:
            for c in range(0, len(run), MAX_SENTENCES):
                chunk = run[c:c + MAX_SENTENCES]
                if any(cand[i] for i in chunk):
                    out.append(passage(parva, [rows[i] for i in chunk], names, hints))
                    stats["passages"] += 1
    return out, stats


def passage(parva, win, names, hints):
    """One passage record: its sentences numbered [S1].., its names and pronoun hints."""
    r = win[0]
    text = " ".join("[S%d] %s" % (k + 1, x.sentence) for k, x in enumerate(win))
    # Names written in the passage, and separately the names coreference gives
    # for its pronouns. Coreference is sometimes wrong: in 01_Adi 104 it reads
    # "His daughter" as Radha. So a hint name is never shown as a name of the
    # passage, and a relation resting on one is flagged.
    who, hint, hinted = set(), [], set()
    for x in win:
        who.update(names.get((parva, x.sent_id), []))
        for surface, canonical in hints.get((parva, x.sent_id), []):
            hint.append('"%s" = %s' % (surface, canonical))
            hinted.add(canonical)
    pairs = sorted({(s, c) for x in win for s, c in hints.get((parva, x.sent_id), [])})
    return {"parva": parva, "sent_id": int(r.sent_id),
            "sent_ids": [int(x.sent_id) for x in win],
            "sentences": [str(x.sentence) for x in win],
            "adhyaya_ordinal": r.adhyaya_ordinal, "cited": bool(r.chapter_cited),
            "text": text, "names": sorted(who), "hint_names": sorted(hinted - who),
            "hint_pairs": pairs, "hints": sorted(set(hint))[:16]}


def quote_sentence(p, ev):
    """The sentence the quote sits in, so a relation cites its own line, not the passage's first."""
    q = norm(ev)
    for sid, s in zip(p.get("sent_ids", [p["sent_id"]]), p.get("sentences", [])):
        if q in norm(s):
            return sid
    return p["sent_id"]


def prompt(p):
    hints = "\n".join("- " + h for h in p["hints"]) or "- none"
    return ("PASSAGE:\n%s\n\nNAMES IN THE PASSAGE: %s\n\nHINTS (who a pronoun or description "
            "refers to, from an automatic step; may be wrong):\n%s"
            % (p["text"], ", ".join(p["names"]) or "none", hints))


def parse(raw):
    """The model's JSON list of relations, or None if it gave no usable JSON."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    rels = data.get("relations") if isinstance(data, dict) else None
    return rels if isinstance(rels, list) else None


def check(p, rel, stats):
    """A relation kept only if its type is known, its names and its quote are in the passage."""
    if not isinstance(rel, dict):
        stats["rejected_shape"] += 1
        return None
    a, b = str(rel.get("a", "")).strip(), str(rel.get("b", "")).strip()
    typ = str(rel.get("type", "")).strip().lower().replace(" ", "_").replace("-", "_")
    if typ not in TYPES and typ in TYPE_WORDS:
        stats["type_translated"] += 1
        typ = TYPE_WORDS[typ]
    ev = str(rel.get("evidence", "")).strip()
    if typ not in TYPES or not a or not b or norm(a) == norm(b):
        stats["rejected_type_or_empty"] += 1
        return None
    text, q = norm(p["text"]), norm(ev)
    if not ev or q not in text:
        stats["rejected_quote_not_in_passage"] += 1
        return None
    # A name, not a description: "ten thousand sons", "son" and "Satyavati's
    # son" all came back as names on the first Kaggle run.
    for n in (a, b):
        words = n.split()
        if (any(w[0].islower() for w in words) or re.search(r"\d|['’]s\b", n)
                or norm(n) in TITLES):
            stats["rejected_not_a_name"] += 1
            return None
    # Both people must be in the quote itself, by name or by a pronoun that
    # coreference resolved to them. Kept only in the passage, the model paired
    # any two names it saw: "Karna spouse Arjuna" from "killed by Arjuna".
    hinted = {}
    for surface, canonical in p.get("hint_pairs", []):
        hinted.setdefault(norm(canonical), []).append(norm(surface))
    via_hint = False
    for n in (a, b):
        if re.search(r"\b%s\b" % re.escape(norm(n)), q):
            continue
        if any(re.search(r"\b%s\b" % re.escape(s), q) for s in hinted.get(norm(n), [])):
            via_hint = True
            continue
        stats["rejected_name_not_in_quote"] += 1
        return None
    # The quote must say something about family, and spouse or sibling must
    # say marriage or brotherhood: every spouse on the first run was invented.
    if not KIN.search(ev) or (typ == "spouse" and not SPOUSE.search(ev)) \
            or (typ == "sibling" and not SIBLING.search(ev)):
        stats["rejected_no_kinship_in_quote"] += 1
        return None
    if not KIN.search(ADDRESS.sub(" ", ev)):
        stats["rejected_address_only"] += 1
        return None
    # "A had three sons, named B" and "A ... his son was B" make A the parent.
    # The model read both the wrong way round on the first run.
    if typ in PARENT_TYPES:
        A, B = re.escape(norm(a)), re.escape(norm(b))
        if re.search(r"\b%s\b[^.;]*\b(had|begot|fathered|sired|gave birth to)\b[^.;]*\b%s\b" % (A, B), q) \
                or re.search(r"\b%s\b[^.;]*\b(his|her) (sons?|daughters?) (was|were|named)\b[^.;]*\b%s\b"
                             % (A, B), q):
            a, b = b, a
            stats["direction_fixed"] += 1
    stats["kept_via_hint" if via_hint else "kept"] += 1
    return {"a": a, "b": b, "type": typ, "evidence": ev, "via_hint": via_hint,
            "sent_id": quote_sentence(p, ev)}


def load_model(name, fp16):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"device_map": "auto", "dtype": torch.float16}
    if not fp16:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    model.eval()
    return tok, model


def generate(tok, model, passages):
    import torch
    texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": prompt(p)}],
                                     tokenize=False, add_generation_prompt=True) for p in passages]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def run(passages, args, done_parts=(), model_fn=None, parts=None):
    tok, model = (None, None) if model_fn else load_model(args.model, args.fp16)
    raw_gen = model_fn or (lambda batch: generate(tok, model, batch))

    def gen(batch):
        """Generate, halving the batch on a GPU out-of-memory error.

        A few sentences run past 500 words, and a batch that stacked several of
        them ran a T4 out of memory 2,000 passages into the first full run. The
        batch is split and retried, down to one passage, instead of failing.
        """
        try:
            return raw_gen(batch)
        except Exception as e:
            if "out of memory" not in str(e).lower() or len(batch) == 1:
                raise
            import torch
            torch.cuda.empty_cache()
            stats["batches_split_for_memory"] += 1
            half = len(batch) // 2
            return gen(batch[:half]) + gen(batch[half:])
    parts = parts or PARTS
    os.makedirs(parts, exist_ok=True)
    size = getattr(args, "batch", BATCH)
    stats, rows, raws, t0, done_now, shown = collections.Counter(), [], [], time.time(), 0, False
    # Resume by passage, not by part file. Every passage with a saved raw answer
    # is done, whichever run or earlier output it came from. Resuming by part
    # number skipped 960 passages on the first full run: a 20-passage smoke-test
    # part_0000 was taken for a finished 500-passage one.
    todo = [i for i, p in enumerate(passages) if key(p) not in done_parts]
    stamp, n_part = time.strftime("%Y%m%d%H%M%S"), 0
    print("%s%d passages to read, %d already done" % (getattr(args, "tag", ""), len(todo),
                                                       len(passages) - len(todo)), flush=True)
    for b in range(0, len(todo), size):
        idx = todo[b:b + size]
        batch = [passages[i] for i in idx]
        for i, p, raw in zip(idx, batch, gen(batch)):
            if not shown:
                print("first raw model output:", raw[:400].replace("\n", " "), flush=True)
                shown = True
            rels = parse(raw)
            readable = rels is not None
            if not readable:
                stats["unparsable_output"] += 1
                rels = []
            n_kept = 0
            for rel in rels:
                kept = check(p, rel, stats)
                if kept:
                    n_kept += 1
                    rows.append({"parva": p["parva"], "sent_id": kept["sent_id"],
                                 "adhyaya_ordinal": p["adhyaya_ordinal"], "cited": p["cited"],
                                 "a": kept["a"], "b": kept["b"], "type": kept["type"],
                                 "evidence": kept["evidence"], "via_hint": kept["via_hint"],
                                 "model": args.model})
            # Every answer is kept as given, so a dropped or unreadable relation
            # can be looked at afterwards, and a judged sample can find what was
            # missed and not only what was wrong.
            raws.append({"parva": p["parva"], "sent_ids": ",".join(map(str, p["sent_ids"])),
                         "raw": raw, "parsed": readable,
                         "relations_returned": len(rels), "relations_kept": n_kept})
            done_now += 1
            stats["passages_read"] += 1
            if done_now % CHECKPOINT == 0 or done_now == len(todo):
                # Each run names its own parts, so no run can overwrite another's.
                name = "%s_%04d" % (stamp, n_part)
                pd.DataFrame(rows).to_parquet("%s/part_%s.parquet" % (parts, name))
                pd.DataFrame(raws).to_parquet("%s/raw_%s.parquet" % (parts, name))
                # The stats so far go with the part, so a run that is cut off
                # still reports what it read.
                json.dump(dict(stats), open("%s/stats_%s.json" % (parts, name), "w"))
                print("%spassages %d/%d, part %s saved, %d relations in it, %.2f passages/s"
                      % (getattr(args, "tag", ""), done_now, len(todo), name, len(rows),
                         done_now / max(time.time() - t0, 1e-6)), flush=True)
                stats, rows, raws, n_part = collections.Counter(), [], [], n_part + 1
    return stats


def key(p):
    """A passage's identity: its parva and its sentence ids."""
    return (p["parva"], ",".join(map(str, p["sent_ids"])))


def done_keys():
    """Every passage some saved raw answer covers, from any run or attached output."""
    done = set()
    for d, _, fs in os.walk(PARTS):
        for f in fs:
            if f.startswith("raw_") and f.endswith(".parquet"):
                r = pd.read_parquet(os.path.join(d, f), columns=["parva", "sent_ids"])
                done.update(zip(r.parva, r.sent_ids))
    return done


def merge(args):
    """Every saved part from every shard: the output file, the report, the manifest."""
    files = sorted(os.path.join(d, f) for d, _, fs in os.walk(PARTS) for f in fs)
    frames = [pd.read_parquet(f) for f in files if os.path.basename(f).startswith("part_")]
    frames = [f for f in frames if len(f)]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    stats = collections.Counter()
    for f in files:
        if os.path.basename(f).startswith("stats_"):
            stats.update(json.load(open(f)))
    if any("prev" in os.path.basename(os.path.dirname(f)) for f in files):
        stats["note_counts_exclude_attached_earlier_output"] = 1
    raws = [pd.read_parquet(f) for f in files if os.path.basename(f).startswith("raw_")]
    raws = [r for r in raws if len(r)]
    raw = pd.concat(raws, ignore_index=True) if raws else pd.DataFrame()
    # A passage read twice (an earlier output attached, then read again) counts once.
    if len(raw):
        raw = raw.drop_duplicates(["parva", "sent_ids"], keep="last")
    if len(frame):
        frame = frame.drop_duplicates(["parva", "sent_id", "a", "b", "type", "evidence"])
    raw.to_parquet(RAW_OUT)
    frame.to_parquet(OUT)
    stats.pop("passages_read", 0)
    report(len(raw), frame, stats, args)
    json.dump({"model": args.model, "limit": args.limit, "rows": len(frame),
               "shards": args.shards, "quantization": "fp16" if args.fp16 else "4bit"},
              open("build/kinship_llm_manifest.json", "w"))


def report(n_passages, frame, stats, args):
    lines = ["# Phase 4k. Kinship relations read by a language model", "",
             "Generated by `src/phase4k_kinship_llm.py`. Candidates only, never asserted. "
             "Every kept row has both names and its quote inside its own passage; accuracy "
             "is unmeasured until a judged sample exists.", "",
             "- Model: `%s` (%s)" % (args.model, "fp16" if args.fp16 else "4-bit"),
             "- Passages read: %d" % n_passages,
             "- Relations kept: %d" % len(frame)]
    lines += ["- %s: %d" % (k, v) for k, v in sorted(stats.items())]
    if len(frame):
        lines += ["", "| type | relations |", "|---|---|"]
        lines += ["| %s | %d |" % (t, n) for t, n in frame.type.value_counts().items()]
    open(REPORT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=0, help="first N passages only: a smoke test")
    ap.add_argument("--dry-run", action="store_true", help="build passages and prompts, no model")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--fp16", action="store_true", help="load in fp16 instead of 4-bit")
    ap.add_argument("--batch", type=int, default=BATCH, help="passages per generate call")
    ap.add_argument("--shard", type=int, default=0, help="this process's share: 0 .. shards-1")
    ap.add_argument("--shards", type=int, default=1,
                    help="run this many processes, one per GPU, each on every shards-th passage")
    ap.add_argument("--merge", action="store_true",
                    help="no model: combine every shard's saved parts into the output")
    args = ap.parse_args()
    args.tag = "[shard %d] " % args.shard if args.shards > 1 else ""
    if args.merge:
        merge(args)
        return

    mentions = pd.read_parquet("build/mentions.parquet")
    coref = pd.read_parquet("build/coref.parquet") if os.path.exists("build/coref.parquet") \
        else pd.DataFrame()
    passages, _ = build_passages(mentions, coref)
    if args.limit:
        passages = passages[:args.limit]
    print("passages %d, with coreference hints %d" % (
        len(passages), sum(bool(p["hints"]) for p in passages)), flush=True)
    if args.dry_run:
        print("\n--- example prompt ---\n" + prompt(passages[0]))
        return

    # Two T4s split one model's layers and take turns; two copies of the 4-bit
    # model, one per GPU, each on half the passages, run side by side instead.
    passages = [p for j, p in enumerate(passages) if j % args.shards == args.shard]
    parts = os.path.join(PARTS, "shard%d" % args.shard)
    run(passages, args, done_keys(), parts=parts)
    if args.shards == 1:
        merge(args)


if __name__ == "__main__":
    main()
