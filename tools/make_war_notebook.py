"""Write tools/mahabharata_war_kaggle.ipynb from the kinship notebook and the current sources.

The notebook carries its scripts inline (%%writefile cells), so it is rebuilt
from src/ whenever src/phase4w_war_llm.py or src/phase4k_kinship_llm.py
changes: python tools/make_war_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
kin = json.loads((ROOT / "tools/mahabharata_kinship_kaggle.ipynb").read_text(encoding="utf-8"))
cells = kin["cells"]


def src(text):
    return text.splitlines(keepends=True)


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src(text)}


def writefile(name):
    body = (ROOT / "src" / name).read_text(encoding="utf-8")
    return code("%%%%writefile /kaggle/working/mahabharata-main/src/%s\n%s" % (name, body))


def swap(cell, pairs):
    text = "".join(cell["source"])
    for old, new in pairs:
        assert old in text, old
        text = text.replace(old, new)
    return code(text)


intro = {"cell_type": "markdown", "metadata": {}, "source": src(
    "# Mahabharata knowledge graph: Phase 4w, killed / taught / charioteer with a language model\n\n"
    "Runs `src/phase4w_war_llm.py` on a Kaggle GPU. The same open model as the kinship run "
    "(`Qwen/Qwen2.5-7B-Instruct`, 4-bit) reads every passage with a killing, teaching or chariot-driving "
    "word and a person in it (about 3,100 passages) and returns who killed whom, who taught whom and who "
    "drove whose chariot, **as that passage itself states it**, each with an exact quote.\n\n"
    "Output is a **candidate layer**: nothing is asserted. Guards drop invented quotes, names that are "
    "titles, vows and threats (\"I will kill Karna\"), and fix passive direction. Accuracy is unmeasured "
    "until a hand-judged packet exists.\n\n"
    "Inputs: one dataset holding `mentions.parquet` and `coref.parquet` (the same as the kinship run). "
    "Settings: Accelerator GPU T4 x2, Internet on. Run Cell 5 with `LIMIT = 40` first, read Cell 6, then "
    "set `LIMIT = 0` for the full run. Download the zip from Cell 7; it unpacks into the repo root.\n\n"
    "The output quotes a copyrighted translation: keep the notebook's output private.\n")}
setup = swap(cells[1], [("kinship_parts", "war_parts"), ("kinship_llm", "war_llm")])
run = swap(cells[5], [("build/kinship_parts", "build/war_parts"),
                      ('base = [sys.executable, "src/phase4k_kinship_llm.py", "--model", MODEL, "--batch", str(BATCH)]',
                       'base = [sys.executable, "src/phase4w_war_llm.py", "--model", MODEL]'),
                      ('subprocess.run([sys.executable, "src/phase4k_kinship_llm.py", "--merge"',
                       'subprocess.run([sys.executable, "src/phase4w_war_llm.py", "--merge"'),
                      ("BATCH = 16                             # passages per GPU call; lower to 8 if a GPU runs out of memory\n", "")])
show = code(
    "# Cell 6: read the results before trusting them: the counts, then examples per type.\n"
    "import pandas as pd\nfrom IPython.display import Markdown, display\n"
    'display(Markdown(Path("build/war_llm_report.md").read_text(encoding="utf-8")))\n'
    'rel = pd.read_parquet("build/war_llm.parquet")\n'
    "if len(rel):\n"
    '    for typ, g in rel.groupby("type"):\n'
    '        print("\\n==", typ)\n'
    "        for r in g.head(6).itertuples():\n"
    '            print("  %s -> %s  [%s %s]%s  \\"%s\\"" % (r.a, r.b, r.parva, r.sent_id,\n'
    '                  " (via hint)" if r.via_hint else "", r.evidence[:120]))\n')
bundle = code(
    "# Cell 7: bundle for the Output tab. The zip unpacks into the repo root.\n"
    "# The parquet holds short quotes from the translation: keep it private.\n"
    "import zipfile\n"
    'dest = Path("/kaggle/working/mahabharata_war_outputs.zip")\n'
    'with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:\n'
    '    for rel_path in ["build/war_llm.parquet", "build/war_llm_raw.parquet", "build/war_llm_report.md"]:\n'
    "        z.write(WORK / rel_path, rel_path)\n"
    'print(dest, "%.1f MB" % (dest.stat().st_size / 1e6))\n'
    "if LIMIT:\n"
    '    print("WARNING: smoke test of %d passages. Set LIMIT = 0 for the full run." % LIMIT)\n')
nb = {"cells": [intro, setup, cells[2], cells[3], writefile("phase4k_kinship_llm.py"), writefile("phase4w_war_llm.py"),
                run, show, bundle],
      "metadata": kin["metadata"], "nbformat": kin.get("nbformat", 4), "nbformat_minor": kin.get("nbformat_minor", 5)}
out = ROOT / "tools/mahabharata_war_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", out.relative_to(ROOT), "with", len(nb["cells"]), "cells")
