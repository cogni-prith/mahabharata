"""Recover chapter boundaries from the parva PDFs.

Writes build/spine.json (chapter heads with character offsets into the
normalized text stream) and build/pdftext/<parva>.txt (that stream, cached so
Phase 1 can locate sentences without re-parsing the PDFs).

Text extraction mirrors preprocess_pdf.py: words mode, sorted by block/line/word,
footnote suffixes stripped. get_text() without "words" silently drops function
words on several of these PDFs, so do not change the mode.

Three head formats occur in this edition and all three are needed:

  A  "CHAPTER 861(1)"  uppercase body head. Most of parvas 6 to 18.
  B  "226(1)"          bare, no CHAPTER word. All of Sabha, Aranyaka, Virata,
                       most of Udyoga, and Bhishma 11 where the word is simply
                       missing from the source.
  C  a line containing only the chapter number, at font size 10.8 against body
                       text at 14.4. Adi only, and Adi uses nothing else.

Detection walks the expected sequence 1..N per parva and searches for the next
number specifically, with a cursor that only moves forward. Scanning for any
number instead would collect footnote markers, which are typographically
identical to format C heads once the text is flattened.
"""
import json, os, re, sys

import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(ROOT, "parva_pdfs")
OUT_DIR = os.path.join(ROOT, "build")
TXT_DIR = os.path.join(OUT_DIR, "pdftext")

# Chapter counts in the BORI critical edition. Used to validate recovery, never
# to fabricate a boundary: a chapter not found in the text stays missing.
BORI = {
    "01_Adi": 225, "02_Sabha": 72, "03_Aranyaka": 299, "04_Virata": 67,
    "05_Udyoga": 197, "06_Bhishma": 117, "07_Drona": 173, "08_Karna": 69,
    "09_Shalya": 64, "10_Souptika": 18, "11_Stri": 27, "12_Shanti": 353,
    "13_Anushasana": 154, "14_Ashvamedhika": 96, "15_Ashramavasika": 47,
    "16_Mausala": 9, "17_Mahaprasthanika": 3, "18_Svargarohana": 5,
}

_FOOTNOTE_SUFFIX = re.compile(r"(?<=[A-Za-z])\d+$")
LIST = re.compile(r"\bCHAPTER\s+(\d+)\s*\(\s*(\d+)\s*\)\s*:\s*(\d+)\s+shlokas", re.I)


def solo_numbers(page):
    """Number-only lines at less than body size, in reading order.

    Adi marks chapters this way and nothing else does. Body text here is 14.4pt
    and the chapter numbers are 10.8pt, which is the only reliable signal once
    the text is flattened, because an inline footnote marker is the same glyphs.
    """
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            txt = "".join(sp["text"] for sp in l["spans"]).strip()
            if re.fullmatch(r"\d{1,4}", txt) and max(sp["size"] for sp in l["spans"]) < 12:
                out.append(txt)
    return out


def page_text(page):
    words = page.get_text("words")
    words.sort(key=lambda w: (w[5], w[6], w[7]))
    out = []
    for w in words:
        cleaned = _FOOTNOTE_SUFFIX.sub("", w[4])
        if cleaned:
            out.append(cleaned)
    return " ".join(out)


def normalize(text):
    """The join key normalization. Must match preprocess_pdf.py."""
    text = re.sub(r"['’]s\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(w.lower() for w in text.split(" ") if not w.isdigit())


def head_patterns(rel, parva):
    """Patterns matching the head of chapter `rel`.

    The (?!\\s*:) guard on A and B rejects the front-matter list form
    "Chapter 1220(1): 52 shlokas", a table of contents entry rather than a
    boundary. Both live in the same text stream.

    C and C' are restricted to Adi. A standalone number before a speech opener
    is also exactly what a footnote marker looks like once the text is
    flattened, so applying C elsewhere invents boundaries: unrestricted, it
    fired 12 times in Bhishma and pushed the search cursor past 13 real heads.
    Adi is the only parva that uses the format, confirmed by layout: it has 225
    number-only lines at font size 10.8 against body text at 14.4, and every
    other parva has none.
    """
    pats = [
        ("A", re.compile(r"\bCHAPTER\s+(\d+)\s*\(\s*%d\s*\)(?!\s*:)" % rel, re.I)),
        ("B", re.compile(r"(?<!\w)(\d{1,4})\s*\(\s*%d\s*\)(?!\s*:)" % rel)),
    ]
    if parva == "01_Adi":
        pats.append(
            ("C", re.compile(r"[.!?’”]\s+(%d)\s+(?=[‘“A-Z])" % rel)))
        # C', the same where extraction fused a trailing footnote marker onto the
        # chapter number, turning 4 into 41. Confirmed in Adi chapter 4.
        pats.append(
            ("C'", re.compile(r"[.!?’”]\s+(%d)\d\s+(?=[‘“A-Z])" % rel)))
    return pats


def find_heads(raw, n_expected, parva):
    """Walk the expected chapter sequence, taking the first plausible match.

    Plausibility is a continuity constraint on the printed global number. The
    bare form "554 (136)" is indistinguishable from a numbered list entry, and
    Anushasana contains the thousand names of Shiva numbered exactly that way:
    "554 (136) Bhojana;555 (137) Bhokta". Those matched 23 times and Shanti 6
    more. Requiring the global number to continue the running sequence rejects
    them, because the real chapter 136 is global 1817, not 554.

    The window is wide because printed global numbers carry typos in this
    source, confirmed in Shalya, which prints 1126 where 1226 belongs. Ordinal
    position is the key, so a typo needs to be tolerated, not trusted. The
    comparison is non-decreasing rather than increasing for the same reason:
    Ashramavasika prints global 1977 for both chapter 45 and chapter 46.
    """
    heads, missing = [], []
    cursor, prev_global = 0, None

    def plausible(g):
        if prev_global is None:
            return True
        return prev_global <= g <= prev_global + 50

    for rel in range(1, n_expected + 1):
        hit = None
        for tag, pat in head_patterns(rel, parva):
            for m in pat.finditer(raw, cursor):
                if not plausible(int(m.group(1))):
                    continue
                if hit is None or m.start() < hit[1].start():
                    hit = (tag, m)
                break
        if hit is None:
            missing.append(rel)
            continue
        tag, m = hit
        prev_global = int(m.group(1))
        heads.append({
            "ordinal": len(heads) + 1,
            "printed_relative": rel,
            "printed_global": prev_global,
            "raw_offset": m.start(),
            "format": tag,
        })
        cursor = m.end()
    return heads, missing


def find_heads_layout(pages, page_solos, n_expected):
    """Adi. Heads come from layout; offsets come from locating the number token
    within the page that carried it."""
    heads, missing = [], []
    starts, acc = [], 0
    for p in pages:
        starts.append(acc)
        acc += len(p) + 1

    expected = 1
    for pno, solos in enumerate(page_solos):
        cursor = 0
        for txt in solos:
            val = int(txt)
            # Extraction fuses a trailing footnote marker onto the chapter
            # number on one page, turning 4 into 41. Accept only when the
            # expected number is a strict prefix, never as a general repair.
            if val != expected and str(val).startswith(str(expected)):
                val = expected
            m = re.compile(r"(?<!\d)%s(?!\d)" % txt).search(pages[pno], cursor)
            if m is None:
                continue
            cursor = m.end()
            if val != expected:
                continue
            heads.append({
                "ordinal": len(heads) + 1,
                "printed_relative": val,
                "printed_global": val,
                "raw_offset": starts[pno] + m.start(),
                "format": "C" if txt == str(val) else "C-fused",
            })
            expected += 1
    missing = [r for r in range(1, n_expected + 1)
               if r not in {h["printed_relative"] for h in heads}]
    return heads, missing


def build(pdf_path, n_expected, parva):
    doc = fitz.open(pdf_path)
    pages, page_solos = [], []
    for page in doc:
        pages.append(page_text(page))
        page_solos.append(solo_numbers(page) if parva == "01_Adi" else [])
    n_pages = len(pages)
    doc.close()
    raw = " ".join(pages)

    listed = {(int(m.group(1)), int(m.group(2))): int(m.group(3))
              for m in LIST.finditer(raw)}
    if parva == "01_Adi":
        heads, missing = find_heads_layout(pages, page_solos, n_expected)
    else:
        heads, missing = find_heads(raw, n_expected, parva)

    # Offsets into normalized space. normalize() only deletes characters, so the
    # normalized length of a prefix is the normalized offset of its end.
    for h in heads:
        h["norm_offset"] = len(normalize(raw[:h["raw_offset"]]))
        h["declared_shlokas"] = listed.get(
            (h["printed_global"], h["printed_relative"]))

    return normalize(raw), heads, missing, len(listed), n_pages


def main():
    os.makedirs(TXT_DIR, exist_ok=True)
    spine, tot_found, tot_exp = {}, 0, 0
    print("%-20s %6s %6s %6s  %s" % ("parva", "BORI", "found", "listed", "formats / missing"))
    for parva, n_exp in BORI.items():
        norm, heads, missing, n_listed, n_pages = build(
            os.path.join(PDF_DIR, parva + ".pdf"), n_exp, parva)
        with open(os.path.join(TXT_DIR, parva + ".txt"), "w", encoding="utf-8") as fh:
            fh.write(norm)
        fmts = {}
        for h in heads:
            fmts[h["format"]] = fmts.get(h["format"], 0) + 1
        spine[parva] = {
            "pages": n_pages, "norm_chars": len(norm), "bori_chapters": n_exp,
            "listed_chapters": n_listed, "found": len(heads),
            "missing_relative": missing, "formats": fmts, "chapters": heads,
        }
        tot_found += len(heads)
        tot_exp += n_exp
        note = ",".join("%s=%d" % kv for kv in sorted(fmts.items()))
        if missing:
            note += "  MISSING %d: %s" % (len(missing), missing[:8])
        print("%-20s %6d %6d %6d  %s" % (parva, n_exp, len(heads), n_listed, note))

    with open(os.path.join(OUT_DIR, "spine.json"), "w", encoding="utf-8") as fh:
        json.dump(spine, fh)
    print("\ntotal: %d of %d BORI chapters recovered (%.1f%%)"
          % (tot_found, tot_exp, 100.0 * tot_found / tot_exp))


if __name__ == "__main__":
    sys.exit(main())
