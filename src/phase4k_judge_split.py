"""Phase 4k judge pilot, Stage 2: split the hand-judged kinship sample and score
the rules the judge has to beat.

Reads annotations/kinship_judged.csv (local only, never committed), writes
build/kinship_judge_split.csv (id -> tune/test) and prints the baseline on the
test half. No language model is called.

Split: 25 direct + 25 via-hint for tuning the prompt, 50 + 50 held out. Each
half keeps the share of correct relations its group has (stratified on
via_hint x correct), with a fixed seed so the split never moves. wrong and
wrong_direction both count as wrong; there are no unsure verdicts.

Rules scored:
  current rule  keep every direct relation, drop every via-hint one
  gender check  current rule, minus parent relations whose parent has the
                wrong gender (a mother who is a man, a father who is a woman)

The gender check was designed after all 150 verdicts were seen, so its score
here may be slightly flattering.
"""
import collections
import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
JUDGED = ROOT / "annotations" / "kinship_judged.csv"
SPLIT = BUILD / "kinship_judge_split.csv"
BASELINE = BUILD / "kinship_judge_baseline.json"
SEED = 4
TUNE_PER_GROUP = 25

MASC = {"he", "him", "his", "himself"}
FEM = {"she", "her", "herself", "hers"}
WANT = {"mother": "f", "father": "m", "divine_father": "m", "adoptive_father": "m"}


def norm(s):
    return re.sub(r"\s+", " ", str(s).replace("’", "'")).strip().lower()


def split(judged):
    """Stratified tune/test split; the counts per stratum are proportional."""
    judged = judged.assign(correct=judged.verdict == "correct")
    parts = []
    for via, group in judged.groupby("via_hint"):
        for _, stratum in group.groupby("correct"):
            n = round(len(stratum) * TUNE_PER_GROUP / len(group))
            tune = stratum.sample(n=n, random_state=SEED).id
            parts.append(stratum.assign(split=stratum.id.isin(tune).map({True: "tune", False: "test"})))
    out = pd.concat(parts).sort_values("id")
    for via, group in out.groupby("via_hint"):
        assert (group.split == "tune").sum() == TUNE_PER_GROUP, via
    return out


def genders():
    """Gender per person: majority of plain he/she pronouns coreference gave
    them (>= 5 and >= 3:1), else their mother/father role from GLiREL."""
    coref = pd.read_parquet(BUILD / "coref.parquet", columns=["surface", "canonical"])
    count = collections.defaultdict(collections.Counter)
    for s, c in zip(coref.surface.map(norm), coref.canonical.map(norm)):
        if s in MASC:
            count[c]["m"] += 1
        elif s in FEM:
            count[c]["f"] += 1
    gender = {}
    for c, k in count.items():
        if k["m"] + k["f"] >= 5:
            if k["m"] >= 3 * k["f"]:
                gender[c] = "m"
            elif k["f"] >= 3 * k["m"]:
                gender[c] = "f"
    for r in pd.read_csv(BUILD / "parent_roles.csv").itertuples():
        if r.role in ("mother", "father"):
            gender.setdefault(norm(r.parent), "f" if r.role == "mother" else "m")
    return gender


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes out of n."""
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 3), round(centre + half, 3)]


def score(rows, keep):
    kept = rows[keep]
    k, n, c = int(kept.correct.sum()), len(kept), int(rows.correct.sum())
    return {"judged": len(rows), "correct": c, "kept": n, "kept_correct": k,
            "precision": round(k / n, 3) if n else None, "precision_ci95": wilson(k, n),
            "retention": round(k / c, 3) if c else None}


def main():
    judged = pd.read_csv(JUDGED)
    judged = split(judged)
    BUILD.mkdir(exist_ok=True)
    judged[["id", "via_hint", "split"]].to_csv(SPLIT, index=False)

    gender = genders()
    b_gender = judged.b.map(lambda x: gender.get(norm(x)))
    want = judged.type.map(WANT)
    judged["gender_ok"] = want.isna() | b_gender.isna() | (b_gender == want)

    report = {}
    for name, part in judged.groupby("split"):
        report[name] = {}
        for group, rows in part.groupby("via_hint"):
            label = "via_hint" if group else "direct"
            report[name][label] = {
                "current rule": score(rows, pd.Series(not group, index=rows.index)),
                "gender check": score(rows, (not group) & rows.gender_ok),
            }
    BASELINE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("people with a gender: %d" % len(gender))
    for name in ("tune", "test"):
        print("\n%s half" % name.upper())
        for label, rules in report[name].items():
            first = next(iter(rules.values()))
            print("  %s: %d judged, %d correct" % (label, first["judged"], first["correct"]))
            for rule, s in rules.items():
                if s["kept"]:
                    print("    %-13s keeps %2d, %2d correct: precision %.0f%% (95%% CI %.0f-%.0f%%), retention %.0f%%"
                          % (rule, s["kept"], s["kept_correct"], 100 * s["precision"],
                             100 * s["precision_ci95"][0], 100 * s["precision_ci95"][1], 100 * s["retention"]))
                else:
                    print("    %-13s keeps 0" % rule)
    print("\nwrote %s and %s" % (SPLIT.relative_to(ROOT), BASELINE.relative_to(ROOT)))


if __name__ == "__main__":
    main()
