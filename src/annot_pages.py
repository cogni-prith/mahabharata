"""Generate offline review pages from public inputs, or install a review export."""
import argparse
import json
import os
import tempfile
from pathlib import Path

from annotation_data import ROOT, SOURCES, COLUMNS, FILENAMES, load_dataset, validate_review


def safe_json(data):
    # A script element is parsed as HTML before JSON. Escape its delimiter.
    return json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")


def render(data):
    template = (ROOT / "src/annotation.html").read_text(encoding="utf-8")
    script = (ROOT / "src/annotation.js").read_text(encoding="utf-8")
    return template.replace("/* ANNOTATION_APP */", script).replace("ANNOTATION_DATA", safe_json(data))


def install(path, replace=False):
    review = json.loads(path.read_text(encoding="utf-8"))
    task = review.get("task")
    if task not in SOURCES:
        raise ValueError("Unknown review task")
    completed = validate_review(review, load_dataset(task))
    folder = ROOT / "annotations"
    folder.mkdir(exist_ok=True)
    dest = folder / FILENAMES[task]
    if dest.exists() and not replace:
        raise ValueError("Existing review preserved; use --replace for an intentional replacement")
    if dest.exists():
        from annotation_data import digest
        previous = dest.read_bytes()
        backup = folder / (dest.stem + ".previous-" + digest(previous.hex())[:12] + ".json")
        if not backup.exists():
            backup.write_bytes(previous)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder, delete=False) as f:
        json.dump(review, f, ensure_ascii=True, indent=2)
        tmp = f.name
    os.replace(tmp, dest)
    print("Installed %s: %d/%d decisions; %s" %
          (dest, completed, len(review["rows"]), "submitted" if review["submitted"] else "draft"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--import", dest="import_path", type=Path)
    p.add_argument("--replace", action="store_true")
    args = p.parse_args()
    if args.import_path:
        install(args.import_path, args.replace)
        return
    report = ["# Annotation page generation", "", "No private calibration key was read.", "",
              "| task | records | dataset |", "|---|---|---|"]
    for task in SOURCES:
        data = load_dataset(task)
        dest = ROOT / "build" / ("annot_%s.html" % task)
        dest.write_text(render(data), encoding="utf-8")
        report.append("| %s | %d | %s |" % (task, len(data["rows"]), data["dataset_id"]))
        print("%s: %d records" % (dest, len(data["rows"])))
    (ROOT / "build/annotation_report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as e:
        raise SystemExit(str(e))
