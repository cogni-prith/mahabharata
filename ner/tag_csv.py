"""
tag_csv.py
==========
NER tagger for Mahabharata CSV files — powered by Gemini 2.0 Flash (free tier).

INPUT CSV columns : sentence_no, sentence, word, POS, tag
OUTPUT CSV columns: same, with tag column filled in

Only NON-O tags are written. O-tagged words get an empty cell,
keeping the output clean for human review.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SETUP (one-time)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Install the NEW library (google-genai, not google-generativeai):
       pip install google-genai

2. Get a FREE API key at: https://aistudio.google.com → "Get API key"

3. Set the key as an environment variable BEFORE running:
       Windows PowerShell : $env:GEMINI_API_KEY="AIza..."
       Mac / Linux        : export GEMINI_API_KEY="AIza..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE  (quote filenames that contain brackets!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python tag_csv.py "Mausala_Parva(16).csv" Mausala_tagged.csv

If interrupted, just re-run — it resumes from the checkpoint file.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import os
import csv
import json
import time
import re
from pathlib import Path
from collections import OrderedDict
from unittest import result

# ── Dependency check ───────────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit(
        "Missing dependency. Install it with:\n"
        "    pip install google-genai\n"
    )

# ── Configuration ──────────────────────────────────────────────────────────────
BATCH_SIZE  = 10    # sentences per API call
SLEEP_SECS  = 10  # ~13 RPM — safely under the 15 RPM free-tier limit
MAX_RETRIES = 5     # retry attempts per batch before giving up

VALID_TAGS = {
    "O",
    "B-PERSON",      "I-PERSON",
    "B-GPE",         "I-GPE",
    "B-DYNASTY",     "I-DYNASTY",
    "B-WEAPON",      "I-WEAPON",
    "B-CONCEPT",     "I-CONCEPT",
    "B-EVENT",       "I-EVENT",
    "B-SACRED_TEXT", "I-SACRED_TEXT",
    "B-NORP",        "I-NORP",
}

SYSTEM_INSTRUCTION = """\
You are a Named Entity Recognition (NER) tagger specialising in ancient Indian \
epic texts, specifically the Mahabharata.

Label each word with exactly one BIO tag from this list:
  B-PERSON / I-PERSON        — Characters, people, deities
                               (Arjuna, Krishna, Yudhishthira, Indra, Vyasa, Draupadi)
  B-GPE    / I-GPE           — Geographical / political places
                               (Dvaraka, Hastinapura, Kurukshetra, Panchala, Lanka)
  B-DYNASTY / I-DYNASTY      — Royal lineages, clans, tribes
                               (Pandava, Kaurava, Yadava, Vrishni, Kuru, Bharata, Andhakas)
  B-WEAPON / I-WEAPON        — Weapons, divine arms, missiles
                               (Gandiva, Sudarshana, brahmastra, Pinaka, Pashupatastra, Vajra)
  B-CONCEPT / I-CONCEPT      — Philosophical / abstract spiritual concepts
                               (dharma, karma, moksha, yoga, ahimsa, samsara, maya)
  B-EVENT  / I-EVENT         — Named battles, wars, ceremonies, major events
                               (Kurukshetra war, Rajasuya, Ashvamedha, Swayamvara, Pralaya)
  B-SACRED_TEXT / I-SACRED_TEXT — Holy texts, scriptures, hymns
                               (Vedas, Gita, Upanishads, Puranas, Mahabharata, Rigveda, Smriti)
  B-NORP   / I-NORP          — Nationalities, religious, caste or social groups
                               (Brahmin, Kshatriya, Vaishya, Sudra, Suta, Rishi, Muni)

Rules:
  • BIO notation: B- marks the FIRST word of an entity; I- marks every
    CONTINUATION word of the same entity span.
  • Non-entity words → tag O.
  • Return ONLY a raw JSON object — no markdown fences, no explanation, no extra text.
  • OMIT O-tagged words from the JSON entirely — only include non-O entries.
  • Word indices are 0-based integers (use string keys in JSON).

Output format:
{
  "<sentence_no>": { "<word_index>": "<TAG>", ... },
  ...
}

Example
───────
Input:
  Sentence 3: ["The", "mighty", "Arjuna", "picked", "up", "his", "Gandiva", "bow"]
  Sentence 4: ["Krishna", "and", "the", "Pandava", "army", "marched", "to", "Kurukshetra"]

Output:
{
  "3": {"2": "B-PERSON", "6": "B-WEAPON", "7": "I-WEAPON"},
  "4": {"0": "B-PERSON", "3": "B-DYNASTY", "4": "I-DYNASTY", "7": "B-GPE"}
}
"""
# ── Helpers ────────────────────────────────────────────────────────────────────

def build_user_prompt(batch):
    lines = []
    for sno, _text, words in batch:
        lines.append(f"Sentence {sno}: {json.dumps(words, ensure_ascii=False)}")
    return "\n".join(lines)


def call_gemini(client, batch):
    """
    Calls Gemini with one batch and returns a normalised dict:
      { sentence_no_str: { word_idx_str: tag_str } }
    Returns {} if all retries are exhausted.
    """
    prompt = build_user_prompt(batch)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0,
                ),
            )
            raw = response.text.strip()

            # Strip accidental markdown code fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```\s*$", "", raw)

            parsed = json.loads(raw)

            # Normalise: make all keys strings
            normalised = {}
            for sno_key, word_dict in parsed.items():
                normalised[str(sno_key)] = {
                    str(widx): tag
                    for widx, tag in word_dict.items()
                }
            return normalised

        except json.JSONDecodeError as exc:
            raw_snippet = raw[:300] if 'raw' in dir() else "(no response)"
            print(f"\n    [attempt {attempt}] JSON parse error: {exc}")
            print(f"    Response snippet: {raw_snippet}")

        except Exception as exc:
            msg = str(exc)
            print(f"\n    [attempt {attempt}] ERROR: {msg[:300]}")
            if "404" in msg or "NOT_FOUND" in msg:
                print("    Model not found — stopping immediately.")
                return {}
            elif "429" in msg or "quota" in msg.lower() or "resource_exhausted" in msg.lower():
                wait = 65 * attempt
                print(f"    Rate limit hit. Waiting {wait}s before retry {attempt}…")
                time.sleep(wait)
            else:
                print(f"    Waiting 10s before retry…")
                time.sleep(10)

        if attempt < MAX_RETRIES:
            time.sleep(6)

    print("\n    All retries failed — skipping batch (tags left blank).")
    return {}


def load_checkpoint(path: Path):
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("done", [])), data.get("tags", {})
    return set(), {}


def save_checkpoint(path: Path, done_set: set, tags: dict):
    path.write_text(
        json.dumps({"done": list(done_set), "tags": tags}, ensure_ascii=False),
        encoding="utf-8",
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print('Usage: python tag_csv.py "input.csv" output.csv')
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "GEMINI_API_KEY not set.\n"
            "Windows PowerShell : $env:GEMINI_API_KEY='AIza...'\n"
            "Mac / Linux        : export GEMINI_API_KEY='AIza...'\n"
        )

    input_path  = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    ckpt_path   = input_path.with_suffix(".checkpoint.json")

    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    # ── Load CSV ───────────────────────────────────────────────────────────────
    print(f"Loading {input_path.name} …")
    rows = []
    fieldnames = None

    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows.append({k: v.strip() for k, v in row.items()})

    sentences = OrderedDict()
    for row in rows:
        sno = row["sentence_no"]
        if sno not in sentences:
            sentences[sno] = {"sentence": row["sentence"], "words": []}
        sentences[sno]["words"].append(row["word"])

    print(f"  {len(rows):,} word rows | {len(sentences):,} sentences")

    # ── Checkpoint ─────────────────────────────────────────────────────────────
    done_set, all_tags = load_checkpoint(ckpt_path)
    if done_set:
        print(f"  Resuming — {len(done_set):,} sentences already tagged.")

    pending = [
        (sno, data["sentence"], data["words"])
        for sno, data in sentences.items()
        if sno not in done_set
    ]
    print(f"  Sentences still to tag: {len(pending):,}")

    # ── Tagging loop ───────────────────────────────────────────────────────────
    if pending:
        client = genai.Client(api_key=api_key)

        total_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE
        t_start = time.time()

        for b_start in range(0, len(pending), BATCH_SIZE):
            batch  = pending[b_start : b_start + BATCH_SIZE]
            b_num  = b_start // BATCH_SIZE + 1

            elapsed      = time.time() - t_start
            batches_done = b_num - 1
            if batches_done > 0:
                avg      = elapsed / batches_done
                eta_secs = avg * (total_batches - batches_done)
                eta_str  = f"  ETA ~{int(eta_secs // 60)}m {int(eta_secs % 60)}s"
            else:
                eta_str  = ""

            print(
                f"  Batch {b_num:>4}/{total_batches}"
                f"  [s{batch[0][0]}–s{batch[-1][0]}]"
                f"{eta_str} … ",
                end="", flush=True,
            )

            result = call_gemini(client, batch)

            if not result:
                print("Batch failed - will retry next run.")
                continue

            for sno, _text, words in batch:
                sentence_result = result.get(sno, {})
                sno_tags = all_tags.setdefault(sno, {})

                for widx, word in enumerate(words):
                    tag = sentence_result.get(str(widx), "")
                    if tag not in VALID_TAGS:
                        tag = ""
                    if tag == "O":
                        tag = ""
                    if tag:
                        sno_tags[str(widx)] = tag

                done_set.add(sno)

            print(f"done  ({len(done_set):,}/{len(sentences):,} sentences)")
            save_checkpoint(ckpt_path, done_set, all_tags)

            if b_start + BATCH_SIZE < len(pending):
                time.sleep(SLEEP_SECS)
    else:
        print("All sentences already tagged — writing output directly.")

    # ── Write output CSV ───────────────────────────────────────────────────────
    print(f"\nWriting {output_path.name} …")

    out_fieldnames = fieldnames[:]
    if "tag" not in out_fieldnames:
        out_fieldnames.append("tag")

    word_counters = {}
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()

        for row in rows:
            sno  = row["sentence_no"]
            widx = word_counters.get(sno, 0)
            word_counters[sno] = widx + 1

            out_row = dict(row)
            out_row["tag"] = all_tags.get(sno, {}).get(str(widx), "")
            writer.writerow(out_row)

    total_tags = sum(len(v) for v in all_tags.values())
    print(f"\n✓ Done!")
    print(f"  {total_tags:,} named-entity tags written")
    print(f"  Output    : {output_path}")
    print(f"  Checkpoint: {ckpt_path}  ← safe to delete once happy with results")


if __name__ == "__main__":
    main()
