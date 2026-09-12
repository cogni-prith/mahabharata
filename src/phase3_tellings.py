"""Phase 3. Speech frames and diegetic depth.

Debroy nests direct speech with alternating quote marks: the outer frame opens
with a single quote, the frame inside it with a double, and so on. A speech frame
opener is a name followed by a speech verb, "Vaishampayana said,".

Depth is computable from one sentence, because the typography re-prints the full
enclosing stack at the start of every speech line: "‘“The gods asked, ‘O god!" carries
two enclosing frames before the opener and opens a third. So the depth of a
telling is the number of open quote marks preceding its opener.

A running stack across sentences was tried first and does not work. Opens exceed
closes by design under this convention, 9,040 to 5,390 after possessives are
removed, because a continued speech re-opens at each line and only the last line
closes. The stack drifted to depth 75.

The apostrophe is both a closing quote and a possessive. Of 15,309 occurrences,
9,909 are "’s" possessives. They are stripped before any quote is counted.

Narratee comes from the vocative, "O Janamejaya!", which Debroy preserves. It is
recorded only when present, never inferred from who is listening.
"""
import collections, json, re, sys
import pandas as pd

BASE = "http://example.org/mbh/"

SPEECH_VERB = r"(?:said|spoke|replied|asked|answered|continued|enquired|responded)"
OPENER = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+%s\b" % SPEECH_VERB)
VOCATIVE = re.compile(r"\bO\s+([A-Z][a-z]+)\b")
OPEN_Q = "‘“"
CLOSE_Q = "’”"

# Openers that are not names. "Having" and "He" match the pattern but carry no
# referent, and a frame with no identified narrator is not asserted.
NOT_A_NAME = {"Having", "He", "She", "They", "Then", "Thus", "After", "When",
              "The", "This", "That", "It", "On", "At", "But", "And", "As", "If",
              "While", "Hearing", "Seeing", "Being", "So", "Now", "One", "All",
              "Some", "Both", "Who", "What", "Their", "His", "Her", "Its", "You",
              "We", "I", "There", "These", "Those", "Such", "Other", "Another"}


def iri(kind, *parts):
    tail = "/".join(re.sub(r"[^A-Za-z0-9]+", "_", str(p)).strip("_") for p in parts)
    return "<%s%s/%s>" % (BASE, kind, tail)


POSS = re.compile(r"([A-Za-z])’s\b")
POSS_PL = re.compile(r"s’(?=\s|$|[.,;:!?])")


def strip_possessives(text):
    """Blank out possessive apostrophes so they are not read as closing quotes.
    Length is preserved so character offsets stay valid."""
    text = POSS.sub(lambda m: m.group(1) + "\u0000s", text)
    return POSS_PL.sub("s\u0000", text)


# "O Indra among men" is a form of address meaning "foremost of men" and is not
# the god Indra. "O Gandhari's son" names the addressee's parent, not the
# addressee. Both were read as narratees and both were wrong.
EPITHET_HEAD = re.compile(r"\bO\s+[A-Z][a-z]+(?:\s+among\b|\u0000s\b)")


def narratee_of(text, start, narrator):
    """The vocative addressee of this speech frame, or None.

    Searched forward from the speech verb and stopped at the next open quote,
    because a vocative inside an embedded speech belongs to the embedded
    speaker. Taking the first vocative in the sentence attributed "O Damayanti"
    to the outer narrator Brihadashva when Parnada said it.
    """
    # The quote mark right after the speech verb opens this frame, so it is not
    # an embedded speech and the search starts inside it.
    head = text[start:start + 4]
    for i, ch in enumerate(head):
        if ch in OPEN_Q:
            start += i + 1
            break
    stop = min((text.find(c, start) for c in OPEN_Q if text.find(c, start) >= 0),
               default=len(text))
    window = text[start:stop]
    for m in VOCATIVE.finditer(window):
        if EPITHET_HEAD.match(window, m.start()):
            continue
        if m.group(1) == narrator:
            continue          # self-address; the vocative is the other turn's
        return m.group(1)
    return None


def extract(df):
    """Depth is the count of open quote marks preceding the speech opener."""
    tellings, stats = [], collections.Counter()
    df = df.sort_values(["parva", "sent_id"])
    for r in df.itertuples():
        text = strip_possessives(r.sentence)
        for m in OPENER.finditer(text):
            if m.group(1).split()[0] in NOT_A_NAME:
                stats["opener_not_a_name"] += 1
                continue
            prefix = text[:m.start()]
            # A line often begins by closing the frames the previous line left
            # open, "’”’ CHAPTER 885(25) ‘“Arjuna said". Those leading closes
            # return to an outer level rather than nesting below zero, so they
            # are not counted. Everything from the first open mark onward is.
            fo = min((prefix.find(c) for c in OPEN_Q if c in prefix), default=-1)
            body = prefix if fo < 0 else prefix[fo:]
            depth = sum(body.count(c) for c in OPEN_Q) - sum(body.count(c) for c in CLOSE_Q)
            if depth < 0:
                stats["negative_depth_clamped"] += 1
                depth = 0
            if depth > 6:
                stats["depth_over_6"] += 1
            voc = narratee_of(text, m.end(), m.group(1))
            tellings.append({
                "parva": r.parva,
                "adhyaya_ordinal": None if pd.isna(r.adhyaya_ordinal) else int(r.adhyaya_ordinal),
                "sent_id": int(r.sent_id),
                "narrator": m.group(1),
                "narratee": voc,
                "depth": int(depth),
                "cited": bool(r.chapter_cited),
            })
            stats["tellings"] += 1
    return tellings, stats


def write_ttl(tellings, path):
    """One named graph per telling span, as a TriG-style quad file.

    A telling is the epistemic wrapper: propositions voiced inside it are claims
    by its narrator, not assertions by the text. Phase 4 and 5 attach triples to
    these graphs.
    """
    lines = ["@prefix mbh: <%s> ." % BASE,
             "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
             "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .", ""]
    for i, t in enumerate(tellings):
        g = iri("g/telling", t["parva"], t["sent_id"])
        lines.append("%s a mbh:Telling ;" % g)
        lines.append("  mbh:narrator %s ;" % iri("e", t["narrator"]))
        if t["narratee"]:
            lines.append("  mbh:narratee %s ;" % iri("e", t["narratee"]))
        lines.append('  mbh:diegeticDepth "%d"^^xsd:integer ;' % t["depth"])
        if t["adhyaya_ordinal"] is not None:
            lines.append("  mbh:inPassage %s ;" % iri("p", t["parva"], t["adhyaya_ordinal"]))
        lines.append('  mbh:citedAtChapterLevel "%s"^^xsd:boolean .'
                     % ("true" if t["cited"] else "false"))
        lines.append("")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    men = pd.read_parquet("build/mentions.parquet")
    sents = men.drop_duplicates(subset=["parva", "sent_id"])[
        ["parva", "sent_id", "adhyaya_ordinal", "chapter_cited", "sentence"]]
    tellings, stats = extract(sents)
    write_ttl(tellings, "out/tellings.ttl")
    pd.DataFrame(tellings).to_parquet("build/tellings.parquet")
    by_depth = collections.Counter(t["depth"] for t in tellings)
    top = collections.Counter(t["narrator"] for t in tellings)
    with open("build/phase3_report.md", "w", encoding="utf-8") as f:
        f.write("# Phase 3 run report\n\nGenerated by `src/phase3_tellings.py`.\n\n")
        f.write("- Sentences scanned: %d\n- Tellings extracted: %d\n"
                "- Openers rejected as not a name: %d\n- Negative depths clamped: %d\n"
                "- Depths above 6: %d\n\n"
                % (len(sents), len(tellings), stats["opener_not_a_name"],
                   stats["negative_depth_clamped"], stats["depth_over_6"]))
        f.write("## Diegetic depth distribution\n\n| depth | tellings |\n|---|---|\n")
        for d in sorted(by_depth):
            f.write("| %d | %d |\n" % (d, by_depth[d]))
        f.write("\n## Narrators, top 25\n\n| narrator | tellings |\n|---|---|\n")
        for n, c in top.most_common(25):
            f.write("| %s | %d |\n" % (n, c))
        f.write("\n## Narratee recorded\n\n- With a vocative narratee: %d\n- Without: %d\n"
                % (sum(1 for t in tellings if t["narratee"]),
                   sum(1 for t in tellings if not t["narratee"])))
    print("tellings=%d depths=%s narrators=%d" % (len(tellings), dict(by_depth), len(top)))
    print("stats=%s" % dict(stats))


if __name__ == "__main__":
    main()
