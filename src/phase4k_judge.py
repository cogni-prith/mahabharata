"""Phase 4k judge pilot: a language model checks each kinship relation against
its passage, with no coreference hint, and is scored against the hand verdicts.

The judge sees the claim (the `claim` column, e.g. "Pandu is the father of
Arjuna"), the marked sentence, up to three sentences before it and one after,
all in the same chapter. It does not see the coreference hints: a wrong hint is
what made the via-hint relations fail (28% correct), so the judge has to work
out who each pronoun means for itself. It answers in JSON: who the pronouns
that matter refer to, then yes / no / cant_tell, then the exact words that
support a yes.

One claim per request, temperature 0. Answers go to
build/kinship_judge_runs/<run>.jsonl, one line per relation, so a stopped run
resumes where it left off. A per-minute rate limit is waited out; a daily limit
stops the run cleanly.

The split and the baseline come from src/phase4k_judge_split.py. Tuning rounds
use the tune half only. The test half is judged once, with the frozen prompt,
and needs --final.

Pass criteria, fixed before the judge ran (test half, "cant_tell" = not kept):
  via-hint  precision >= 80% and at least half the correct ones kept
  direct    at least 80% of the correct ones kept, and precision above the
            gender check's 77%
Nothing here is published: passages are verbatim text of a copyrighted
translation, and the runs quote them.

Run:  python src/phase4k_judge.py --run tune1                 judge the tune half
      python src/phase4k_judge.py --run tune1 --score         score a run, show disagreements
      python src/phase4k_judge.py --run test --split test --final
      python src/phase4k_judge.py --run full --split full      judge every direct relation
      python src/phase4k_judge.py --run full --collect         sort it into three files

Result (2026-10-26, prompt 90232c4a): on the test half, direct relations met
the criteria (94% precision, all 34 correct kept); via-hint relations did not
(79%). So the full run judges direct relations only, and via-hint ones stay
dropped. Full run: 732 direct relations -> 468 yes, 77 cant_tell, 187 no. A
fresh hand-judged 50 from the yes answers (src/phase4k_sample.py --post-judge)
was 45 correct: 90%, 95% CI 79-96%. Its errors were relations by marriage or
upbringing read as parentage ("the sons of King Drupada" for his sons-in-law)
and names that point at someone else (Krishna for Draupadi).
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat import AGENT, ENDPOINT, MODEL, retry_after
from phase4k_judge_split import BASELINE, JUDGED, SPLIT, WANT, genders, norm, wilson

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "build" / "kinship_judge_runs"
BEFORE, AFTER = 3, 1
VERDICTS = ("yes", "no", "cant_tell")

PROMPT = """You check one family relation claimed from a passage of the Mahabharata (Ganguli's English translation).

Judge from the passage alone. Do not use what you know of the epic from elsewhere: a claim that is true in the story but not supported by these sentences is "cant_tell".

Work in this order.
1. referents: list each pronoun or descriptive phrase ("he", "his", "my mother", "that king", "the son of Pritha") in the passage that the claim depends on, and who it refers to by the passage alone. Write "unclear" if the passage does not settle it. Note who is speaking when a speaker says "I", "my" or "you".
2. verdict:
   "yes"       the passage states the relation, or makes it plain, between these two people and in this direction (who is the parent and who is the child).
   "no"        the passage says something different: another person, the reverse direction, a different relation, or no relation between them.
   "cant_tell" the passage is about these people but does not settle the claim, or a referent it depends on is unclear.
3. span: for "yes", the shortest exact words copied from the passage that support it; otherwise "".

Points that decide many cases:
- For a father or mother claim, judge whether the passage makes the second person a parent of the first ("X, the son of Y", "born to Y", "Y's son X"). Do not answer "cant_tell" only because the passage does not say whether the parent is a man or a woman; that is checked separately. Answer "no" if the passage shows the parent is of the other sex.
- Praise phrases are not people: "Indra among men", "tiger among men", "bull among the Bharatas", "best of the Kurus" describe someone; they do not name the god Indra or anyone else.
- The epic calls one person by several names. If the passage uses both names in the claim for the same person (for example, one man addressed as both "Partha" and "Dhananjaya"), they are not relatives of each other: answer "no".

Answer with JSON only: {"referents": [{"phrase": "...", "refers_to": "..."}], "verdict": "yes|no|cant_tell", "span": "..."}"""
PROMPT_ID = hashlib.sha1(PROMPT.encode()).hexdigest()[:8]


def passages(rows):
    """Up to BEFORE sentences before the marked one and AFTER after, same chapter."""
    m = pd.read_parquet(ROOT / "build/mentions.parquet",
                        columns=["parva", "sent_id", "adhyaya_ordinal", "sentence"])
    m = m.drop_duplicates(["parva", "sent_id"]).sort_values(["parva", "sent_id"])
    by_parva = {p: g.reset_index(drop=True) for p, g in m.groupby("parva")}
    out = {}
    for r in rows.itertuples():
        g = by_parva[r.parva]
        i = g.index[g.sent_id == r.sent_id][0]
        here = g.iloc[i].adhyaya_ordinal
        same = lambda x: x == here or (pd.isna(x) and pd.isna(here))
        lines = []
        for x in g.iloc[max(0, i - BEFORE):i + AFTER + 1].itertuples():
            if same(x.adhyaya_ordinal):
                mark = ">> " if x.sent_id == r.sent_id else "   "
                lines.append(mark + x.sentence.strip())
        out[r.id] = "\n".join(lines)
    return out


def message(claim, passage):
    return ("Claim: %s\n\nPassage (the claim was read from the sentence marked >>):\n%s"
            % (claim, passage))


class DailyLimit(Exception):
    pass


def ask(text, key, model):
    body = json.dumps({"model": model, "temperature": 0, "reasoning_effort": "low",
                       "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": PROMPT},
                                    {"role": "user", "content": text}]}).encode()
    for attempt in range(8):
        req = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json",
            "User-Agent": AGENT})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:400]
            if e.code == 429 and ("per day" in detail or "(TPD)" in detail or "(RPD)" in detail):
                raise DailyLimit(detail)
            if e.code in (429, 500, 502, 503) and attempt < 7:
                time.sleep(retry_after(detail, default=10.0 * (attempt + 1)))
                continue
            raise RuntimeError("Groq returned %d: %s" % (e.code, detail))
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 7:
                time.sleep(10.0 * (attempt + 1))
                continue
            raise RuntimeError("Could not reach Groq: %s" % e)
    raise RuntimeError("Groq kept refusing")


def parse(content):
    try:
        got = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", content or "", re.S)
        try:
            got = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            got = {}
    verdict = str(got.get("verdict", "")).strip().lower().replace("'", "").replace(" ", "_")
    return got, verdict if verdict in VERDICTS else "unparsed"


def squash(s):
    return re.sub(r"\s+", " ", str(s).replace("’", "'").replace("‘", "'")).strip().lower()


def praise(name, passage):
    """True if the passage uses the name only in a praise phrase: "Indra among
    men", "tiger among men". The prompt asks the judge to see this and in
    tuning it did not, so it is checked here, deterministically."""
    words = re.escape(squash(name))
    text = squash(passage)
    named = len(re.findall(r"\b%s\b" % words, text))
    return named > 0 and named == len(re.findall(r"\b%s among\b" % words, text))


def load_direct():
    """Every direct relation, for the full run. Via-hint relations stay dropped:
    the judge missed its criteria on them (79% against 80%)."""
    from phase4k_sample import PHRASE
    rel = pd.read_parquet(ROOT / "build/kinship_llm.parquet")
    rel = rel[~rel.via_hint].drop_duplicates(["parva", "sent_id", "a", "b", "type", "evidence"])
    rel = rel.reset_index(drop=True)
    rel.insert(0, "id", ["D" + hashlib.sha1("|".join(map(str, k)).encode()).hexdigest()[:10]
                         for k in zip(rel.parva, rel.sent_id, rel.a, rel.b, rel.type, rel.evidence)])
    rel["claim"] = [PHRASE[t].format(a=a, b=b) for t, a, b in zip(rel.type, rel.a, rel.b)]
    return rel


def collect(args):
    """Sort the full run: yes -> candidate layer, cant_tell -> review, no -> log."""
    rows = load_direct()
    recs = pd.DataFrame([json.loads(l) for l in (RUNS / (args.run + ".jsonl")).read_text(encoding="utf-8").splitlines()])
    rows = rows.merge(recs[["id", "prompt_id", "verdict", "referents", "span", "span_in_passage"]]
                      .rename(columns={"verdict": "judge"}), on="id", how="left")
    every = passages(rows)
    rows["praise"] = [praise(a, every[i]) or praise(b, every[i]) for i, a, b in zip(rows.id, rows.a, rows.b)]
    rows.loc[rows.praise & (rows.judge == "yes"), "judge"] = "no"
    rows["referents"] = rows.referents.map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else "")
    missing = rows.judge.isna().sum()
    for name, part in (("candidates", rows[rows.judge == "yes"]),
                       ("review", rows[rows.judge.isin(["cant_tell", "unparsed"])]),
                       ("rejected", rows[rows.judge == "no"])):
        part.drop(columns=["claim"]).to_parquet(ROOT / ("build/kinship_direct_%s.parquet" % name), index=False)
        print("build/kinship_direct_%s.parquet: %d" % (name, len(part)))
    if missing:
        print("%d relations not judged yet; rerun the full run to finish them" % missing)


def load(split_name):
    if split_name == "full":
        return load_direct()
    judged = pd.read_csv(JUDGED).merge(pd.read_csv(SPLIT)[["id", "split"]], on="id")
    judged["correct"] = judged.verdict == "correct"
    return judged[judged.split == split_name].reset_index(drop=True)


def run(args):
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit('GROQ_API_KEY is not set. In PowerShell: '
                         '$env:GROQ_API_KEY = (Get-Content "$HOME\\.groq_key" -Raw).Trim()')
    rows = load(args.split)
    if args.limit:
        rows = rows.head(args.limit)
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / (args.run + ".jsonl")
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["id"]] = rec
    prompts = {rec["prompt_id"] for rec in done.values()}
    if prompts and prompts != {PROMPT_ID}:
        raise SystemExit("%s was started with prompt %s; the prompt is now %s. Use a new --run name."
                         % (path.name, sorted(prompts), PROMPT_ID))
    text = passages(rows)
    todo = rows[~rows.id.isin(done)]
    print("run %s, prompt %s, %s half: %d done, %d to judge" % (args.run, PROMPT_ID, args.split, len(done), len(todo)))
    tokens = 0
    with path.open("a", encoding="utf-8") as out:
        for n, r in enumerate(todo.itertuples(), 1):
            try:
                payload = ask(message(r.claim, text[r.id]), key, args.model)
            except DailyLimit as e:
                print("daily limit reached after %d; rerun the same command later to resume.\n%s" % (n - 1, e))
                break
            content = payload["choices"][0]["message"].get("content") or ""
            got, verdict = parse(content)
            span = str(got.get("span", "") or "")
            rec = {"id": r.id, "prompt_id": PROMPT_ID, "model": args.model, "verdict": verdict,
                   "referents": got.get("referents", []), "span": span,
                   "span_in_passage": bool(span) and squash(span) in squash(text[r.id]),
                   "raw": content if verdict == "unparsed" else "",
                   "usage": payload.get("usage", {})}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            tokens += rec["usage"].get("total_tokens", 0)
            print("  %3d/%d %s %-9s %s" % (n, len(todo), r.id, verdict, r.claim))
    print("tokens this session: %d" % tokens)


def score(args):
    path = RUNS / (args.run + ".jsonl")
    recs = pd.DataFrame([json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()])
    rows = load(args.split).merge(recs, on="id", how="inner", suffixes=("_human", ""))
    rows = rows.rename(columns={"verdict": "judge", "verdict_human": "human"})
    every = passages(rows)
    rows["praise"] = [praise(a, every[i]) or praise(b, every[i]) for i, a, b in zip(rows.id, rows.a, rows.b)]
    rows["kept"] = (rows.judge == "yes") & ~rows.praise
    print("praise-phrase check overrules %d yes answers" % ((rows.judge == "yes") & rows.praise).sum())
    gender = genders()
    b_gender = rows.b.map(lambda x: gender.get(norm(x)))
    want = rows.type.map(WANT)
    rows["gender_ok"] = want.isna() | b_gender.isna() | (b_gender == want)
    base = json.loads(Path(BASELINE).read_text(encoding="utf-8")).get(args.split, {})
    tokens = sum(u.get("total_tokens", 0) for u in rows.usage)
    print("run %s, prompt %s, %s half: %d judged, %d tokens (%.0f a claim)"
          % (args.run, ",".join(sorted(rows.prompt_id.unique())), args.split, len(rows),
             tokens, tokens / max(1, len(rows))))
    verdicts = {}
    for via, g in rows.groupby("via_hint"):
        label = "via_hint" if via else "direct"
        k, n, c = int((g.kept & g.correct).sum()), int(g.kept.sum()), int(g.correct.sum())
        p = k / n if n else 0.0
        keep = k / c if c else 0.0
        ci = wilson(k, n)
        print("\n%s: %d judged, %d correct by hand" % (label.upper(), len(g), c))
        print(pd.crosstab(g.human, g.judge).to_string())
        print("judge keeps %d, %d correct: precision %s, keeps %d of %d correct (%.0f%%)"
              % (n, k, "%.0f%% (95%% CI %.0f-%.0f%%)" % (100 * p, 100 * ci[0], 100 * ci[1]) if n else "-",
                 k, c, 100 * keep))
        # The prompt leaves the parent's sex to the gender check, so the two
        # are also scored together. The criteria apply to the judge alone.
        both = g.kept & g.gender_ok
        kb = int((both & g.correct).sum())
        print("judge + gender check keeps %d, %d correct: precision %s, keeps %d of %d correct"
              % (both.sum(), kb, "%.0f%%" % (100 * kb / both.sum()) if both.sum() else "-", kb, c))
        for rule, s in base.get(label, {}).items():
            print("  baseline %-13s keeps %d, precision %s" % (rule, s["kept"],
                  "%.0f%%" % (100 * s["precision"]) if s["kept"] else "-"))
        if via:
            verdicts[label] = n > 0 and p >= 0.8 and k * 2 >= c
        else:
            verdicts[label] = keep >= 0.8 and p > base.get(label, {}).get("gender check", {}).get("precision", 0.77)
        print("  criteria %s" % ("MET" if verdicts[label] else "not met"))
    yes_spans = rows[rows.kept]
    if len(yes_spans):
        print("\nspans copied exactly from the passage: %d of %d yes answers"
              % (yes_spans.span_in_passage.sum(), len(yes_spans)))
    if args.split == "test":
        print("\nPILOT %s" % ("PASSES" if all(verdicts.values()) else "DOES NOT PASS"))
    else:
        print("\n(tune half: for prompt tuning only; the pass decision is on the test half)")

    wrong = rows[rows.kept != rows.correct]
    print("\nDISAGREEMENTS (%d):" % len(wrong))
    text = passages(wrong) if len(wrong) else {}
    for r in wrong.itertuples():
        print("\n%s  %s  hand: %s  judge: %s" % (r.id, "via-hint" if r.via_hint else "direct", r.human, r.judge))
        print("  claim: %s" % r.claim)
        if isinstance(r.note, str) and r.note.strip():
            print("  your note: %s" % r.note.strip())
        for ref in r.referents or []:
            if isinstance(ref, dict):
                print("  referent: %s -> %s" % (ref.get("phrase"), ref.get("refers_to")))
        if r.span:
            print("  span: %s" % r.span)
        if args.passages:
            print("  " + text[r.id].replace("\n", "\n  "))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="name of this run, e.g. tune1")
    ap.add_argument("--split", default="tune", choices=("tune", "test", "full"),
                    help="full = every direct relation in build/kinship_llm.parquet")
    ap.add_argument("--collect", action="store_true",
                    help="sort a full run into candidates / review / rejected files")
    ap.add_argument("--final", action="store_true", help="required to judge the test half")
    ap.add_argument("--score", action="store_true", help="score an existing run instead of judging")
    ap.add_argument("--passages", action="store_true", help="with --score, print each disagreement's passage")
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N (a smoke test)")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    if args.collect:
        return collect(args)
    if args.score:
        return score(args)
    if args.split == "test" and not args.final:
        raise SystemExit("The test half is judged once, with the frozen prompt. Add --final to do that.")
    run(args)


if __name__ == "__main__":
    main()
