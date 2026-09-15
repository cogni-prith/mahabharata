"""Checks for the annotation desk. Never opens the calibration key.

Two halves. The first is the export validator, which is pure Python and is the
thing standing between a day of annotation and a silently mismatched score. The
second drives the generated page in real Chrome, because the durable record is a
file the browser writes and a round trip is the only way to know it round trips.

Run: .venv/bin/python src/test_annotation.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from annotation_data import (COLUMNS, SOURCES, blank_review, load_dataset,
                             validate_review)

ROOT = Path(__file__).resolve().parents[1]
KEY = "blind_calibration_key"


def rejects(review, data, why, require_submitted=False):
    try:
        validate_review(review, data, require_submitted)
    except ValueError:
        return
    raise AssertionError("accepted a review it should have rejected: " + why)


def check_validator(data):
    task = data["task"]
    blank = blank_review(data)
    assert validate_review(blank, data) == 0, "a blank review has no decisions"

    # The guard that matters most: a review drawn against a different draw must
    # not score against this one. D21 moved the frame once already.
    stale = json.loads(json.dumps(blank))
    stale["dataset_id"] = "0" * 64
    rejects(stale, data, "stale dataset_id")
    if data["sheet_id"]:
        stale = json.loads(json.dumps(blank))
        stale["sheet_id"] = "ffffffffffff"
        rejects(stale, data, "stale sheet_id")

    short = json.loads(json.dumps(blank))
    short["rows"] = short["rows"][:-1]
    rejects(short, data, "a dropped row")

    dupe = json.loads(json.dumps(blank))
    dupe["rows"][1] = json.loads(json.dumps(dupe["rows"][0]))
    rejects(dupe, data, "a duplicated row")

    # The literal source travels with the judgement, so editing the sentence out
    # from under a decision is detectable rather than silently rescored.
    tampered = json.loads(json.dumps(blank))
    field = "sentence" if task == "blind" else "candidate_entity"
    tampered["rows"][0]["source"][field] = "not what was shown"
    rejects(tampered, data, "an edited source record")

    extra = json.loads(json.dumps(blank))
    extra["rows"][0]["verdict"] = "yes"
    rejects(extra, data, "an unexpected field")

    if task == "blind":
        bad = json.loads(json.dumps(blank))
        bad["rows"][0]["status"] = "triples"
        bad["rows"][0]["triples"] = [{"subject": "Karna", "predicate": "isAlliedWith",
                                      "object": "Duryodhana", "unclear": False}]
        rejects(bad, data, "a predicate outside the annotator's fifteen")

        contradiction = json.loads(json.dumps(blank))
        contradiction["rows"][0]["status"] = "none"
        contradiction["rows"][0]["triples"] = [{"subject": "a", "predicate": "",
                                                "object": "b", "unclear": False}]
        rejects(contradiction, data, "no-relation carrying triples")

        # An empty slot is not a decision. Counting it would put unfinished work
        # into the denominator of the gate.
        half = json.loads(json.dumps(blank))
        half["rows"][0]["status"] = "triples"
        half["rows"][0]["triples"] = [{"subject": "Karna", "predicate": "slainBy",
                                       "object": "", "unclear": False}]
        assert validate_review(half, data) == 0, "a half-filled triple counts as complete"

        done = json.loads(json.dumps(blank))
        for r in done["rows"]:
            r["status"] = "none"
        assert validate_review(done, data) == len(done["rows"])
    else:
        bad = json.loads(json.dumps(blank))
        bad["rows"][0]["status"] = "probably"
        rejects(bad, data, "a status outside the five")
        done = json.loads(json.dumps(blank))
        for r in done["rows"]:
            r["status"] = "yes"
        assert validate_review(done, data) == len(done["rows"])

    # Submission is all-or-nothing, and needs a name and the attestation.
    for tweak in ({"submitted": True, "attestation": True, "annotator": "A"},
                  {"submitted": True, "attestation": False, "annotator": "A"},
                  {"submitted": True, "attestation": True, "annotator": "  "}):
        part = json.loads(json.dumps(blank))
        part.update(tweak)
        rejects(part, data, "submitted while incomplete or unattested")
    rejects(json.loads(json.dumps(blank)), data, "unsubmitted under require_submitted", True)

    full = json.loads(json.dumps(done))
    full.update({"submitted": True, "attestation": True, "annotator": "Test"})
    assert validate_review(full, data, True) == len(full["rows"])
    return full


def check_key_stays_closed(data, html):
    """The sheet's own columns reach the page and nothing else does.

    This is structural and never reads the key. If the key ever joined the page
    the whitelist would have to have been widened, and the whitelist is here.
    """
    allowed = set(COLUMNS[data["task"]]) | {"key"}
    if data["task"] == "entities":
        allowed |= {"sightings", "total_sightings"}
    for r in data["rows"]:
        assert set(r) == allowed, set(r) ^ allowed
    for leaked in ("stratum", "pipeline_asserted", "pipeline_proposed", KEY):
        assert leaked not in html, "the page names %s" % leaked
    assert KEY not in Path(ROOT / "src/annotation_data.py").read_text()
    assert not any(KEY in p for p in sum(SOURCES.values(), ()))


def check_embedding(html):
    """A sentence carrying a script tag must not end the data block early.

    The rows ship inline because `file://` blocks `fetch()`, so the escape in
    `annot_pages.safe_json` is the only thing between a `<` in Debroy's prose
    and a page that loads no data at all.
    """
    import annot_pages
    hostile = annot_pages.safe_json({"s": "</script><img onerror=1>"})
    assert "<" not in hostile, hostile
    block = html.split('type="application/json">', 1)[1].split("</script>", 1)[0]
    assert "<" not in block, "raw < inside the inline data block"


def check_storage_blocked(task):
    """localStorage can refuse. The draft must still export, and the page must
    say recovery is unavailable rather than imply it saved."""
    from playwright.sync_api import sync_playwright
    page_path = ROOT / "build" / ("annot_%s.html" % task)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
                                    args=["--no-sandbox"])
        ctx = browser.new_context(viewport={"width": 1440, "height": 1000},
                                  accept_downloads=True)
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.add_init_script("Storage.prototype.getItem = Storage.prototype.setItem ="
                             " function () {throw new Error('storage blocked')};")
        page.goto(page_path.as_uri())
        page.wait_for_selector("#workspace")
        if task == "blind":
            page.get_by_role("radio", name="No relation").check()
        else:
            page.get_by_role("radio", name="Yes", exact=True).first.check()
        page.wait_for_function("document.getElementById('progress').value >= 1")
        assert "unavailable" in page.locator("#save-label").inner_text().lower()
        page.locator("#annotator").fill("Test Annotator")
        with page.expect_download() as got:
            page.locator("#export").click()
        draft = json.loads(Path(got.value.path()).read_text())
        assert sum(r["status"] != "undecided" for r in draft["rows"]) == 1
        assert not errors, errors
        browser.close()


def check_import(full, task):
    """--import installs, refuses a second write, and keeps the first file."""
    out = ROOT / "annotations" / ("phase5_blind.json" if task == "blind" else "phase2_entities.json")
    existed = out.exists()
    before = out.read_bytes() if existed else None
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(full, f)
        path = f.name
    env = {"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"}
    run = lambda *a: subprocess.run([sys.executable, "src/annot_pages.py", "--import", path, *a],
                                    cwd=ROOT, capture_output=True, text=True, env=env)
    try:
        first = run()
        assert first.returncode == 0, first.stderr
        assert out.exists()
        second = run()
        assert second.returncode != 0 and "preserved" in second.stderr, second.stderr
        assert run("--replace").returncode == 0
    finally:
        Path(path).unlink()
        if existed:
            out.write_bytes(before)
        else:
            out.unlink(missing_ok=True)
            for stray in (ROOT / "annotations").glob("*.previous-*.json"):
                stray.unlink()


def check_browser(task, full):
    from playwright.sync_api import sync_playwright
    page_path = ROOT / "build" / ("annot_%s.html" % task)
    downloads = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
                                    args=["--no-sandbox"])
        ctx = browser.new_context(viewport={"width": 1440, "height": 1000},
                                  accept_downloads=True)
        page = ctx.new_page()
        errors, network = [], []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: network.append(r.url) if r.url.startswith("http") else None)
        page.goto(page_path.as_uri())
        page.wait_for_selector("#workspace")
        total = int(page.locator("#progress").get_attribute("max"))

        # Record one decision and confirm it counts.
        if task == "blind":
            page.get_by_role("radio", name="No relation").check()
        else:
            page.get_by_role("radio", name="Yes", exact=True).first.check()
        page.wait_for_function("document.getElementById('progress').value >= 1")
        page.locator("#annotator").fill("Test Annotator")

        # A draft must export without being complete; that is the whole point of
        # a draft, and losing a part-finished day is the failure being prevented.
        with page.expect_download() as got:
            page.locator("#export").click()
        draft = json.loads(Path(got.value.path()).read_text())
        downloads.append(draft)
        assert draft["submitted"] is False and len(draft["rows"]) == total
        assert sum(r["status"] != "undecided" for r in draft["rows"]) == 1
        validate_review(draft, load_dataset(task))

        # Submitting while unfinished must refuse rather than write a short file.
        page.locator("#submit").click()
        assert "unfinished" in page.locator("#message").inner_text().lower()
        assert not page.locator("#confirm").is_visible()

        # The draft survives a reload, which is the recovery promise.
        page.reload()
        page.wait_for_selector("#workspace")
        assert page.locator("#progress").get_attribute("value") == "1"
        assert "Recovered" in page.locator("#message").inner_text()

        # A review from another dataset is rejected and the draft is kept.
        alien = json.loads(json.dumps(draft))
        alien["dataset_id"] = "0" * 64
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(alien, f)
            alien_path = f.name
        page.locator("#file").set_input_files(alien_path)
        page.wait_for_function("document.getElementById('message').classList.contains('error')")
        assert "preserved" in page.locator("#message").inner_text()
        assert page.locator("#progress").get_attribute("value") == "1"
        Path(alien_path).unlink()

        # A complete review imports, and then submits with the attestation.
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(full, f)
            full_path = f.name
        page.locator("#file").set_input_files(full_path)
        page.locator("#confirm-action").click()
        page.wait_for_function("document.getElementById('progress').value == document.getElementById('progress').max")
        page.locator("#submit").click()
        assert page.locator("#attest-wrap").is_visible()
        assert page.locator("#confirm-action").is_disabled(), "attestation was not required"
        page.locator("#attest").check()
        with page.expect_download() as got:
            page.locator("#confirm-action").click()
        submitted = json.loads(Path(got.value.path()).read_text())
        assert submitted["submitted"] and submitted["attestation"]
        validate_review(submitted, load_dataset(task), True)
        Path(full_path).unlink()

        for width, height in ((390, 844), (1920, 1080)):
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(120)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), width
        assert not errors, errors
        assert not network, "the page reached the network: %s" % network
        browser.close()
    return downloads


def main():
    for task in SOURCES:
        data = load_dataset(task)
        html = (ROOT / "build" / ("annot_%s.html" % task)).read_text(encoding="utf-8")
        full = check_validator(data)
        check_key_stays_closed(data, html)
        check_embedding(html)
        check_import(full, task)
        check_browser(task, full)
        check_storage_blocked(task)
        print("%s: %d rows, validator and Chrome round trip pass" % (task, len(data["rows"])))
    print("annotation checks passed; the calibration key was not opened")


if __name__ == "__main__":
    main()
