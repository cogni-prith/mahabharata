"""Phase 4k check. A hand-judged sample of the language-model kinship relations.

Phase 4k's relations are candidates. Their accuracy is unknown until a person
reads a sample against the text, and the smoke tests suggested it differs
sharply between two groups: relations whose two names are both written in the
quote ("direct"), and relations that rest on a pronoun the coreference step
resolved ("via hint"), where a wrong hint can mislead the model.

So the sample is stratified: half direct, half via hint, drawn with a fixed
seed so it can be redrawn exactly. It is written as a local page,
build/kinship_check.html, which shows each relation as a plain sentence beside
its quote and the sentences around it. The page keeps the judgements in the
browser and exports them as a CSV. Nothing here is published: the sentences are
verbatim text of a copyrighted translation.

Run:  python src/phase4k_sample.py                  draw the sample, write the page
      python src/phase4k_sample.py --import FILE    store the exported judgements in
                                                   annotations/, report accuracy
"""
import argparse
import json
import os
import random

import pandas as pd

SRC = "build/kinship_llm.parquet"
SAMPLE = "build/kinship_check_sample.csv"
PAGE = "build/kinship_check.html"
JUDGED = "annotations/kinship_judged.csv"
SEED = 20261007
PER_GROUP = 75
VERDICTS = ("correct", "wrong", "wrong_direction", "unsure")

PHRASE = {
    "father": "{b} is the father of {a}",
    "mother": "{b} is the mother of {a}",
    "parent": "{b} is a parent of {a}",
    "adoptive_father": "{b} is the adoptive father of {a}",
    "adoptive_mother": "{b} is the adoptive mother of {a}",
    "divine_father": "{b} is the divine father of {a}",
    "spouse": "{a} and {b} are married",
    "sibling": "{a} and {b} are siblings",
}


def draw():
    rel = pd.read_parquet(SRC)
    # One row per distinct relation and quote: the same reading of the same
    # line is one judgement, not several.
    rel = rel.drop_duplicates(["parva", "sent_id", "a", "b", "type", "evidence"]).reset_index(drop=True)
    rng = random.Random(SEED)
    picked = []
    for hinted in (False, True):
        pool = list(rel.index[rel.via_hint == hinted])
        picked += rng.sample(pool, min(PER_GROUP, len(pool)))
    rng.shuffle(picked)
    s = rel.loc[picked].reset_index(drop=True)
    s.insert(0, "id", ["K%03d" % (i + 1) for i in range(len(s))])
    s["claim"] = [PHRASE[t].format(a=a, b=b) for t, a, b in zip(s.type, s.a, s.b)]
    return s


def context(sample):
    """The quote's sentence with the one before and after it, in the same chapter."""
    m = pd.read_parquet("build/mentions.parquet").drop_duplicates(["parva", "sent_id"])
    m = m.sort_values(["parva", "sent_id"])
    by_parva = {p: g.reset_index(drop=True) for p, g in m.groupby("parva")}
    hints = {}
    if os.path.exists("build/coref.parquet"):
        c = pd.read_parquet("build/coref.parquet")
        for (p, s), g in c.groupby(["parva", "sent_id"]):
            hints[(p, s)] = sorted({'"%s" = %s' % (r.surface, r.canonical) for r in g.itertuples()})
    out = []
    for r in sample.itertuples():
        g = by_parva[r.parva]
        i = g.index[g.sent_id == r.sent_id][0]
        here = g.iloc[i].adhyaya_ordinal
        rows = [x for x in g.iloc[max(0, i - 1):i + 2].itertuples()
                if x.adhyaya_ordinal == here or (pd.isna(x.adhyaya_ordinal) and pd.isna(here))]
        out.append({"sentences": [{"text": x.sentence, "main": x.sent_id == r.sent_id} for x in rows],
                    "cite": "%s %s" % (r.parva, "" if pd.isna(here) else int(here)),
                    "hints": sum((hints.get((r.parva, x.sent_id), []) for x in rows), [])[:8]})
    return out


def page(sample, ctx):
    items = []
    for r, c in zip(sample.itertuples(), ctx):
        items.append({"id": r.id, "claim": r.claim, "a": r.a, "b": r.b, "type": r.type,
                      "evidence": r.evidence, "via_hint": bool(r.via_hint), "parva": r.parva,
                      "sent_id": int(r.sent_id), **c})
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", data)


def import_judged(path):
    j = pd.read_csv(path)
    missing = {"id", "verdict"} - set(j.columns)
    if missing:
        raise SystemExit("%s lacks columns %s. Use the page's Download button." % (path, sorted(missing)))
    s = pd.read_csv(SAMPLE)
    j = s.merge(j[["id", "verdict"] + (["note"] if "note" in j.columns else [])], on="id", how="left")
    bad = j.verdict.notna() & ~j.verdict.isin(VERDICTS)
    if bad.any():
        raise SystemExit("unknown verdicts: %s" % sorted(j.verdict[bad].unique()))
    os.makedirs(os.path.dirname(JUDGED), exist_ok=True)
    j.to_csv(JUDGED, index=False)
    done = j[j.verdict.notna() & (j.verdict != "unsure")]
    print("stored %s: %d judged, %d unsure, %d not judged"
          % (JUDGED, len(done), int((j.verdict == "unsure").sum()), int(j.verdict.isna().sum())))
    for name, g in [("direct", done[~done.via_hint]), ("via hint", done[done.via_hint]), ("all", done)]:
        if len(g):
            ok = (g.verdict == "correct").sum()
            print("  %-9s correct %3d / %3d = %3.0f%%   (wrong direction %d)"
                  % (name, ok, len(g), 100 * ok / len(g), (g.verdict == "wrong_direction").sum()))
    by_type = done.groupby(["via_hint", "type"]).verdict.apply(
        lambda v: "%d/%d" % ((v == "correct").sum(), len(v)))
    print("\nby type (correct/judged), via_hint False = direct:\n" + by_type.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--import", dest="imp", metavar="FILE", help="exported judgements CSV")
    args = ap.parse_args()
    if args.imp:
        import_judged(args.imp)
        return
    sample = draw()
    sample.to_csv(SAMPLE, index=False)
    open(PAGE, "w", encoding="utf-8").write(page(sample, context(sample)))
    print("sample of %d (%d direct, %d via hint), seed %d -> %s\npage -> %s"
          % (len(sample), int((~sample.via_hint).sum()), int(sample.via_hint.sum()), SEED, SAMPLE, PAGE))


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kinship check</title>
<style>
:root{--bg:#f7f6f2;--card:#fff;--ink:#1f2328;--mute:#656d76;--line:#d8dee4;--mark:#fff1b8;
 --ok:#1a7f37;--bad:#cf222e;--dir:#9a6700;--uns:#6e7781;--accent:#0969da}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--card:#161b22;--ink:#e6edf3;--mute:#8d96a0;
 --line:#30363d;--mark:#4d3d00;--ok:#3fb950;--bad:#f85149;--dir:#d29922;--uns:#8b949e;--accent:#4493f8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
 font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:860px;margin:0 auto;padding:20px 16px 60px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between}
h1{font-size:20px;margin:0}.mute{color:var(--mute);font-size:14px}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;margin:12px 0 18px}
.bar>div{height:100%;background:var(--accent);width:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px}
.claim{font-size:22px;font-weight:600;margin:4px 0 6px}
.tag{display:inline-block;font-size:12px;padding:2px 8px;border-radius:10px;border:1px solid var(--line);color:var(--mute);margin-right:6px}
.tag.hint{border-color:var(--dir);color:var(--dir)}
.ctx{margin:16px 0;padding:12px 14px;border-left:3px solid var(--line);color:var(--mute)}
.ctx p{margin:6px 0}.ctx p.main{color:var(--ink)}
mark{background:var(--mark);color:inherit;padding:0 2px;border-radius:3px}
.hints{font-size:13px;color:var(--mute);margin-top:6px}
.btns{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:16px}
button{font:inherit;padding:10px 8px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--accent)}
button.v.correct.on{background:var(--ok);color:#fff;border-color:var(--ok)}
button.v.wrong.on{background:var(--bad);color:#fff;border-color:var(--bad)}
button.v.wrong_direction.on{background:var(--dir);color:#fff;border-color:var(--dir)}
button.v.unsure.on{background:var(--uns);color:#fff;border-color:var(--uns)}
.nav{display:flex;gap:8px;justify-content:space-between;margin-top:14px;flex-wrap:wrap}
textarea{width:100%;margin-top:12px;font:inherit;font-size:14px;padding:8px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
.help{font-size:13px;color:var(--mute);margin-top:18px}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;font-size:12px}
@media (max-width:560px){.btns{grid-template-columns:repeat(2,1fr)}.claim{font-size:19px}}
</style></head><body><main>
<header><div><h1>Kinship check</h1><div class="mute">Is each relation true <b>according to the passage shown</b>? Your knowledge of the epic can help, but the text must say it.</div></div>
<button id="dl">Download judgements</button></header>
<div class="bar"><div id="prog"></div></div>
<div class="mute" id="count"></div>
<div class="card" id="card"></div>
<div class="nav"><button id="prev">&larr; Previous</button><button id="next_open">Next unjudged</button><button id="next">Next &rarr;</button></div>
<div class="help"><kbd>1</kbd> correct &nbsp;<kbd>2</kbd> wrong &nbsp;<kbd>3</kbd> wrong direction (the right two people, swapped, e.g. child given as parent) &nbsp;<kbd>4</kbd> unsure &nbsp;<kbd>&larr;</kbd><kbd>&rarr;</kbd> move. Judgements are kept in this browser until you download them.</div>
</main>
<script>
const ITEMS=__DATA__, KEY="kinship-check-v1", LABEL={correct:"Correct",wrong:"Wrong",wrong_direction:"Wrong direction",unsure:"Unsure"};
let J={}; try{J=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let at=0; const first=ITEMS.findIndex(x=>!(J[x.id]&&J[x.id].verdict)); if(first>=0) at=first;
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(J))}catch(e){}};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function mark(text,quote){const t=esc(text),q=esc(quote);const i=t.indexOf(q);return i<0?t:t.slice(0,i)+"<mark>"+q+"</mark>"+t.slice(i+q.length)}
function render(){
 const x=ITEMS[at], v=(J[x.id]||{}).verdict, note=(J[x.id]||{}).note||"";
 const done=ITEMS.filter(i=>J[i.id]&&J[i.id].verdict).length;
 document.getElementById("prog").style.width=(100*done/ITEMS.length)+"%";
 document.getElementById("count").textContent=`Item ${at+1} of ${ITEMS.length} · ${done} judged`;
 document.getElementById("card").innerHTML=
  `<div><span class="tag">${esc(x.id)}</span><span class="tag">${esc(x.type.replace("_"," "))}</span>`+
  (x.via_hint?`<span class="tag hint">rests on a pronoun hint</span>`:"")+`<span class="tag">${esc(x.cite)}</span></div>`+
  `<div class="claim">${esc(x.claim)}</div><div class="mute">Quote the model gave: “${esc(x.evidence)}”</div>`+
  `<div class="ctx">${x.sentences.map(s=>`<p class="${s.main?"main":""}">${s.main?mark(s.text,x.evidence):esc(s.text)}</p>`).join("")}</div>`+
  (x.via_hint&&x.hints.length?`<div class="hints">Pronoun hints the model was given (automatic, may be wrong): ${x.hints.map(esc).join("; ")}</div>`:"")+
  `<div class="btns">${Object.keys(LABEL).map((k,n)=>`<button class="v ${k} ${v===k?"on":""}" data-v="${k}">${n+1} · ${LABEL[k]}</button>`).join("")}</div>`+
  `<textarea id="note" rows="2" placeholder="Optional note">${esc(note)}</textarea>`;
 document.querySelectorAll("button.v").forEach(b=>b.onclick=()=>judge(b.dataset.v));
 document.getElementById("note").oninput=e=>{J[x.id]=Object.assign(J[x.id]||{},{note:e.target.value});save()};
}
function judge(v){const x=ITEMS[at];J[x.id]=Object.assign(J[x.id]||{},{verdict:v});save();if(at<ITEMS.length-1)at++;render()}
document.getElementById("prev").onclick=()=>{if(at>0)at--;render()};
document.getElementById("next").onclick=()=>{if(at<ITEMS.length-1)at++;render()};
document.getElementById("next_open").onclick=()=>{const i=ITEMS.findIndex(x=>!(J[x.id]&&J[x.id].verdict));if(i>=0)at=i;render()};
document.addEventListener("keydown",e=>{if(e.target.tagName==="TEXTAREA")return;
 const k={"1":"correct","2":"wrong","3":"wrong_direction","4":"unsure"}[e.key];
 if(k)judge(k);else if(e.key==="ArrowLeft"&&at>0){at--;render()}else if(e.key==="ArrowRight"&&at<ITEMS.length-1){at++;render()}});
document.getElementById("dl").onclick=()=>{
 const q=s=>'"'+String(s??"").replace(/"/g,'""')+'"';
 const rows=[["id","verdict","note"].join(",")].concat(ITEMS.map(x=>[x.id,(J[x.id]||{}).verdict||"",(J[x.id]||{}).note||""].map(q).join(",")));
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows.join("\n")],{type:"text/csv"}));
 a.download="kinship_judged.csv";a.click()};
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
