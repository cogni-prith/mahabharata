"""Phase 4w. Killed, taught and charioteer relations read by a language model.

The hand-judged packets (src/phase4l_relations_check.py) found the graph's own
slainBy relations 88% right but few, and GLiREL's killed, teacher and
charioteer candidates 3-10% right: of no use. This reads every passage with a
killing, teaching or chariot-driving word and a person in it, the way Phase 4k
reads family passages, and asks an instruction-tuned model for the relations
the passage itself states, each with an exact quote.

Passages, prompts, guards and the Kaggle run follow src/phase4k_kinship_llm.py,
whose helpers this imports. Guards before a row is kept:
  - type known; both names are names, not titles or descriptions;
  - the quote is in the passage, holds both people (by name, or by a pronoun
    coreference resolved, then flagged via_hint) and the type's own words;
  - the quote is not a vow, threat, wish, question or condition ("I will
    kill Karna", "who can slay him?"), the commonest false killing;
  - "B was slain by A" and "B learnt from A" are put the right way round.
Nothing here is asserted. Output is a candidate layer, build/war_llm.parquet,
whose accuracy is unmeasured until a hand-judged packet exists.

Run: python src/phase4w_war_llm.py --dry-run             passages and one prompt, no model
     python src/phase4w_war_llm.py --groq --limit 20     smoke test on Groq (about 30k tokens)
     python src/phase4w_war_llm.py [--shard i --shards n] full run on a GPU (Kaggle)
     python src/phase4w_war_llm.py --merge               combine saved parts
"""
import argparse
import collections
import json
import os
import re
import time

import pandas as pd

from phase4k_kinship_llm import TITLES, key, load_model, norm, parse, passage, prompt, quote_sentence

MODEL = "Qwen/Qwen2.5-7B-Instruct"
GROQ_MODEL = "openai/gpt-oss-120b"
OUT = "build/war_llm.parquet"
RAW_OUT = "build/war_llm_raw.parquet"
PARTS = "build/war_parts"
REPORT = "build/war_llm_report.md"
CHECKPOINT = 250
BATCH = 8
MAX_NEW_TOKENS = 320
MAX_SENTENCES = 5

WORDS = {
    "killed": r"kill(?:ed|s|ing)?|sl[ae]w|slain|slay(?:s|ing)?|slaughter(?:ed)?|struck down|"
              r"killer|slayer|deprived \w+ of (?:his |their |her )?li(?:fe|ves)|sent \w+ to (?:the abode of )?yama",
    "taught": r"taught|teach(?:es|ing)?|instruct(?:ed|ion|s)?|discipl(?:e|es)|pupils?|preceptor|guru|"
              r"acharya|teacher|learnt|learned|studied",
    "charioteer": r"charioteers?|drove|driver|drive|driving|reins|sarathi",
}
TYPE_RE = {t: re.compile(r"\b(?:%s)\b" % w, re.I) for t, w in WORDS.items()}
ANY = re.compile(r"\b(?:%s)\b" % "|".join(WORDS.values()), re.I)
# A killing that has not happened: the commonest wrong "killed" in a war epic.
IRREALIS = re.compile(r"\b(will|shall|would|wish(?:es|ed)? to|desir\w* to|vow\w*|want\w* to|tried to|"
                      r"try to|attempt\w* to|intend\w* to|about to|if|whether|unless|cannot|can't|"
                      r"unable|let me|let us|may|might|must)\b|\?", re.I)
TYPE_WORDS = {"kill": "killed", "slew": "killed", "slain": "killed", "slain_by": "killed", "killed_by": "killed",
              "teacher": "taught", "teaches": "taught", "teach": "taught", "instructed": "taught",
              "preceptor": "taught", "teacher_of": "taught", "learnt_from": "taught",
              "charioteer_of": "charioteer", "drove": "charioteer", "driver": "charioteer"}

SYSTEM = """You extract three kinds of relations from one passage of an English translation of the Mahabharata: who killed whom, who taught whom, and who was whose charioteer.

Report only relations that THIS passage states as having happened. Never use your own knowledge of the epic, even when you are sure of it. If the passage states none, return an empty list.

Relation types:
- killed: A killed B. The passage says B was killed, slain or slaughtered by A, or that A slew B.
- taught: A taught B, or B learnt from A, as a teacher or preceptor (weapons, the Vedas, a body of knowledge).
- charioteer: A drove B's chariot; A was B's charioteer.

Rules:
- Only what happened. A vow, threat, wish, plan, order, question, prophecy, curse or condition is not a relation: "I will kill Karna", "he wished to slay him", "who can kill him?", "if you teach him".
- Wounding, piercing, striking, defeating, attacking, routing or fighting is not killing. Advice given in one conversation is not teaching.
- Keep the roles in passive sentences: "Karna was slain by Arjuna" gives a = Arjuna, b = Karna. "Arjuna learnt from Drona" gives a = Drona, b = Arjuna.
- "a" and "b" are personal names exactly as written, never a description ("the king", "his son", "the charioteer", "the preceptor").
- "his", "her", "he", "she", "you" may stand for a person. Use the HINTS when they give the person, and use the names exactly as written in the passage or the hints.
- "evidence" must be an exact quote copied from the passage that contains BOTH people, by name or by the pronoun that stands for them, and the words that show the relation. Copy one unbroken stretch of the text, never joined with "..." or reworded. Keep it short, but never cut a person out of it.

Answer with JSON only, no other text:
{"relations": [{"a": "<killer, teacher or driver>", "b": "<the one killed, taught or driven>", "type": "<killed, taught or charioteer>", "evidence": "<exact quote>"}]}

Examples.
Passage: "Dhrishtadyumna killed Drona, who was seated on his chariot."
{"relations": [{"a": "Dhrishtadyumna", "b": "Drona", "type": "killed", "evidence": "Dhrishtadyumna killed Drona"}]}
Passage: "The brave Karna was slain by Arjuna in battle."
{"relations": [{"a": "Arjuna", "b": "Karna", "type": "killed", "evidence": "Karna was slain by Arjuna"}]}
Passage: "Shalya became Karna's charioteer. Arjuna learnt the use of weapons from Drona."
{"relations": [{"a": "Shalya", "b": "Karna", "type": "charioteer", "evidence": "Shalya became Karna's charioteer"}, {"a": "Drona", "b": "Arjuna", "type": "taught", "evidence": "Arjuna learnt the use of weapons from Drona"}]}
Passage: "Bhima vowed that he would kill Duhshasana. Arjuna pierced Karna with ten arrows."
{"relations": []}"""


def build_passages(mentions, coref):
    """As in Phase 4k: runs of marked sentences with their neighbours, never crossing a chapter."""
    sents = mentions.drop_duplicates(["parva", "sent_id"]).sort_values(["parva", "sent_id"])
    sents = sents[["parva", "sent_id", "adhyaya_ordinal", "chapter_cited", "sentence"]]
    names = mentions[mentions.label == "PERSON"].groupby(["parva", "sent_id"]).surface.agg(lambda x: sorted(set(x)))
    hints = {}
    if len(coref):
        for (p, s), g in coref.groupby(["parva", "sent_id"]):
            hints[(p, s)] = sorted({(r.surface, r.canonical) for r in g.itertuples()})
    out, stats = [], collections.Counter()
    sents = sents.assign(chapter=sents.adhyaya_ordinal.fillna(-1))
    for (parva, _), g in sents.groupby(["parva", "chapter"], sort=False):
        rows = list(g.itertuples())
        cand = [ANY.search(str(r.sentence)) is not None
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


def check(p, rel, stats):
    """A relation kept only if it passes every guard in the module docstring."""
    if not isinstance(rel, dict):
        stats["rejected_shape"] += 1
        return None
    a, b = str(rel.get("a", "")).strip(), str(rel.get("b", "")).strip()
    typ = str(rel.get("type", "")).strip().lower().replace(" ", "_").replace("-", "_")
    if typ not in TYPE_RE and typ in TYPE_WORDS:
        stats["type_translated"] += 1
        typ = TYPE_WORDS[typ]
    ev = str(rel.get("evidence", "")).strip()
    if typ not in TYPE_RE or not a or not b or norm(a) == norm(b):
        stats["rejected_type_or_empty"] += 1
        return None
    q = norm(ev)
    if not ev or q not in norm(p["text"]):
        stats["rejected_quote_not_in_passage"] += 1
        return None
    for n in (a, b):
        if any(w[0].islower() for w in n.split()) or re.search(r"\d|['’]s\b", n) or norm(n) in TITLES \
                or norm(n) in {"charioteer", "teacher", "slayer", "killer"}:
            stats["rejected_not_a_name"] += 1
            return None
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
    if not TYPE_RE[typ].search(ev):
        stats["rejected_no_type_word_in_quote"] += 1
        return None
    if IRREALIS.search(ev):
        stats["rejected_not_a_fact"] += 1
        return None
    A, B = re.escape(norm(a)), re.escape(norm(b))
    if typ == "killed" and re.search(r"\b%s\b[^.;]*\b(was|were|had been|has been|is|lay)\b[^.;]*\b(killed|slain|"
                                     r"slaughtered)\b[^.;]*\bby\b[^.;]*\b%s\b" % (A, B), q):
        a, b = b, a
        stats["direction_fixed"] += 1
    if typ == "taught" and re.search(r"\b%s\b[^.;]*\b(learnt|learned|studied)\b[^.;]*\b(from|under)\b[^.;]*\b%s\b"
                                     % (A, B), q):
        a, b = b, a
        stats["direction_fixed"] += 1
    stats["kept_via_hint" if via_hint else "kept"] += 1
    return {"a": a, "b": b, "type": typ, "evidence": ev, "via_hint": via_hint, "sent_id": quote_sentence(p, ev)}


def gpu_answers(model_name, fp16):
    import torch
    tok, model = load_model(model_name, fp16)

    def answer(batch):
        texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": prompt(p)}],
                                         tokenize=False, add_generation_prompt=True) for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer


def groq_answers(model_name):
    from phase4k_judge import DailyLimit, ask    # Groq only: Kaggle runs never import chat.py
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise SystemExit('GROQ_API_KEY is not set. In PowerShell: '
                         '$env:GROQ_API_KEY = (Get-Content "$HOME\\.groq_key" -Raw).Trim()')
    used = collections.Counter()

    def answer(batch):
        out = []
        for p in batch:
            try:
                payload = ask(prompt(p), groq_key, model_name, system=SYSTEM)
            except DailyLimit as e:
                raise SystemExit("Groq daily limit reached; rerun later to resume.\n%s" % e)
            used["tokens"] += payload.get("usage", {}).get("total_tokens", 0)
            out.append(payload["choices"][0]["message"].get("content") or "")
        print("  Groq tokens so far: %d" % used["tokens"], flush=True)
        return out
    return answer


def done_keys():
    done = set()
    for d, _, fs in os.walk(PARTS):
        for f in fs:
            if f.startswith("raw_") and f.endswith(".parquet"):
                r = pd.read_parquet(os.path.join(d, f), columns=["parva", "sent_ids"])
                done.update(zip(r.parva, r.sent_ids))
    return done


def run(passages, answer, model_name, parts):
    os.makedirs(parts, exist_ok=True)
    done = done_keys()
    todo = [p for p in passages if key(p) not in done]
    print("%d passages to read, %d already done" % (len(todo), len(passages) - len(todo)), flush=True)
    stats, rows, raws, t0, stamp, n_part = collections.Counter(), [], [], time.time(), time.strftime("%Y%m%d%H%M%S"), 0
    for b in range(0, len(todo), BATCH):
        batch = todo[b:b + BATCH]
        for p, raw in zip(batch, answer(batch)):
            rels = parse(raw)
            readable = rels is not None
            stats["unparsable_output"] += not readable
            n_kept = 0
            for rel in rels or []:
                kept = check(p, rel, stats)
                if kept:
                    n_kept += 1
                    rows.append({"parva": p["parva"], "sent_id": kept["sent_id"], "adhyaya_ordinal": p["adhyaya_ordinal"],
                                 "cited": p["cited"], "a": kept["a"], "b": kept["b"], "type": kept["type"],
                                 "evidence": kept["evidence"], "via_hint": kept["via_hint"], "model": model_name})
            raws.append({"parva": p["parva"], "sent_ids": ",".join(map(str, p["sent_ids"])), "raw": raw,
                         "parsed": readable, "relations_returned": len(rels or []), "relations_kept": n_kept})
        n = b + len(batch)
        if n % CHECKPOINT < BATCH or n == len(todo):
            name = "%s_%04d" % (stamp, n_part)
            pd.DataFrame(rows).to_parquet("%s/part_%s.parquet" % (parts, name))
            pd.DataFrame(raws).to_parquet("%s/raw_%s.parquet" % (parts, name))
            json.dump(dict(stats), open("%s/stats_%s.json" % (parts, name), "w"))
            print("passages %d/%d, part %s saved, %.2f passages/s" % (n, len(todo), name, n / max(time.time() - t0, 1e-6)),
                  flush=True)
            stats, rows, raws, n_part = collections.Counter(), [], [], n_part + 1


def merge(model_name):
    files = sorted(os.path.join(d, f) for d, _, fs in os.walk(PARTS) for f in fs)
    frames = [f for f in (pd.read_parquet(f) for f in files if os.path.basename(f).startswith("part_")) if len(f)]
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(["parva", "sent_id", "a", "b", "type", "evidence"]) \
        if frames else pd.DataFrame()
    raws = [r for r in (pd.read_parquet(f) for f in files if os.path.basename(f).startswith("raw_")) if len(r)]
    raw = pd.concat(raws, ignore_index=True).drop_duplicates(["parva", "sent_ids"], keep="last") if raws else pd.DataFrame()
    stats = collections.Counter()
    for f in files:
        if os.path.basename(f).startswith("stats_"):
            stats.update(json.load(open(f)))
    raw.to_parquet(RAW_OUT)
    frame.to_parquet(OUT)
    lines = ["# Phase 4w. Killed, taught and charioteer relations read by a language model", "",
             "Generated by `src/phase4w_war_llm.py`. Candidates only, never asserted; accuracy is "
             "unmeasured until a hand-judged packet exists.", "",
             "- Model(s): %s" % ", ".join(sorted(frame.model.unique())) if len(frame) else "- Model: %s" % model_name,
             "- Passages read: %d" % len(raw), "- Relations kept: %d" % len(frame)]
    lines += ["- %s: %d" % (k, v) for k, v in sorted(stats.items())]
    if len(frame):
        lines += ["", "| type | relations | via hint |", "|---|---|---|"]
        lines += ["| %s | %d | %d |" % (t, len(g), int(g.via_hint.sum())) for t, g in frame.groupby("type")]
    open(REPORT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def spread(passages, n):
    """n passages spread evenly over the epic, so a smoke test is not all Adi Parva."""
    step = max(1, len(passages) // n)
    return passages[::step][:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=0, help="N passages spread over the epic: a smoke test")
    ap.add_argument("--dry-run", action="store_true", help="build passages and one prompt, no model")
    ap.add_argument("--groq", action="store_true", help="answer on Groq instead of a local GPU")
    ap.add_argument("--model", default=None)
    ap.add_argument("--fp16", action="store_true", help="GPU: load in fp16 instead of 4-bit")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="no model: combine saved parts into the output")
    args = ap.parse_args()
    model_name = args.model or (GROQ_MODEL if args.groq else MODEL)
    if args.merge:
        merge(model_name)
        return
    mentions = pd.read_parquet("build/mentions.parquet")
    coref = pd.read_parquet("build/coref.parquet") if os.path.exists("build/coref.parquet") else pd.DataFrame()
    passages, stats = build_passages(mentions, coref)
    print("candidate sentences %d, passages %d" % (stats["candidate_sentences"], len(passages)), flush=True)
    if args.limit:
        passages = spread(passages, args.limit)
    if args.dry_run:
        print("\n--- system prompt: %d characters ---\n--- example user prompt ---\n%s" % (len(SYSTEM), prompt(passages[0])))
        return
    passages = [p for j, p in enumerate(passages) if j % args.shards == args.shard]
    answer = groq_answers(model_name) if args.groq else gpu_answers(model_name, args.fp16)
    run(passages, answer, model_name, os.path.join(PARTS, "groq" if args.groq else "shard%d" % args.shard))
    if args.shards == 1:
        merge(model_name)


if __name__ == "__main__":
    main()
