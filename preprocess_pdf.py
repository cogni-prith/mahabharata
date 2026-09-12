
"""
preprocess_pdf.py
==================
Extracts and cleans text from a PDF into LLM-ready verses/sentences —
fixing the exact extraction artifacts already fought in the validator's
own /extract-pdf endpoint, so the bulk-tagging pipeline starts from clean
input instead of inheriting the same garbage.

What it fixes:
  1. Line-wrap word-gluing ("of those" -> "ofthose"). Fixed by extracting
     each word as its own pre-segmented unit (fitz's "words" mode) and
     joining them with an explicit space ourselves, rather than asking
     any library to infer where spaces belong from character positions —
     that inference is exactly where pypdf/pdfplumber both failed
     identically on this PDF.
  2. Footnote digits glued onto the end of a word with no space
     ("Mahat11" -> "Mahat") — the classic copy-paste-flattens-a-
     superscript artifact.
  3. Footnote digits glued onto sentence-ending punctuation
     ("...of dharma.2 In gentle...") — left alone, this merges two real
     sentences into one when splitting.
  4. Splits strictly on periods only (not "!" or "?"), matching the
     validator's own splitting behavior, so verse boundaries stay
     predictable.

Usage:
    python preprocess_pdf.py TheMahabharata.pdf
    python preprocess_pdf.py TheMahabharata.pdf cleaned_verses.json
"""

import sys
import re
import json
import fitz  # PyMuPDF


# Same patterns already proven against this exact PDF in main.py's
# /predict and /extract-pdf endpoints.
_FOOTNOTE_SUFFIX_RE = re.compile(r"(?<=[A-Za-z])\d+$")
_FOOTNOTE_AFTER_PUNCT_RE = re.compile(r"([.!?])(\d+)")
# Matches both the straight apostrophe and the curly one PDFs often use.
_POSSESSIVE_RE = re.compile(r"['\u2019]s\b", re.IGNORECASE)


def extract_clean_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)

    all_words = []
    for page in doc:
        words_on_page = page.get_text("words")
        # (x0, y0, x1, y1, word, block_no, line_no, word_no) — sort into
        # proper reading order, then take each word as a discrete unit.
        words_on_page.sort(key=lambda w: (w[5], w[6], w[7]))
        for w in words_on_page:
            cleaned = _FOOTNOTE_SUFFIX_RE.sub("", w[4])
            if cleaned:
                all_words.append(cleaned)

    full_text = " ".join(all_words).strip()

    # Separate footnote digits still glued onto sentence-ending
    # punctuation, before that punctuation itself gets stripped below.
    full_text = _FOOTNOTE_AFTER_PUNCT_RE.sub(r"\1 \2", full_text)

    # Drop possessive 's entirely BEFORE the blanket punctuation strip —
    # otherwise stripping just the apostrophe leaves a dangling orphan
    # "s" word behind ("Krishna's" -> "Krishna s" rather than "Krishna").
    full_text = _POSSESSIVE_RE.sub("", full_text)

    # Strip every remaining punctuation mark EXCEPT the period (commas,
    # quotes, exclamation/question marks, semicolons, dashes, ...).
    # Replaced with a space rather than deleted outright — deleting a
    # dash/quote that sits directly between two words with no
    # surrounding space would glue them back together, which is exactly
    # the bug this whole pipeline exists to avoid.
    full_text = re.sub(r"[^\w\s.]", " ", full_text)
    full_text = re.sub(r"\s+", " ", full_text).strip()

    # Drop any standalone digit-only token — footnote markers freed up
    # by either step above.
    full_text = " ".join(w for w in full_text.split(" ") if not w.isdigit())

    return full_text


def split_into_verses(full_text: str) -> list[str]:
    return [chunk.strip() + "." for chunk in full_text.split(".") if chunk.strip()]


def main():
    if len(sys.argv) < 2:
        print("Usage: python preprocess_pdf.py <input.pdf> [output.json]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "cleaned_verses.json"

    print(f"Extracting and cleaning text from {pdf_path} ...")
    full_text = extract_clean_text(pdf_path)

    if not full_text:
        print("No extractable text found — this PDF may be scanned/image-only.")
        sys.exit(1)

    verses = split_into_verses(full_text)
    print(f"Found {len(verses)} verses/sentences.")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(verses, f, ensure_ascii=False, indent=2)

    print(f"Saved to {out_path}")
    print("\nFirst 3 verses, to sanity-check the cleaning:")
    for v in verses[:3]:
        print(f"  - {v}")


if __name__ == "__main__":
    main()
