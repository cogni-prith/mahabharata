"""Phonetic key for romanised Sanskrit, Bengali and Hindi names.

The chat layer already reaches "Arjun" from Arjuna by edit distance. Edit
distance cannot reach "Orjun", "Korno", "Bheem" or "Durjodhon", because those
are not typos. They are how the same name is spelled by someone who speaks
Bengali or Hindi, and they sit 2 to 3 edits away, which is where unrelated
names also sit. Measured with fuzz.WRatio over the entity labels:

    orjun      -> Arjuna 73        (below CUTOFF 85, nothing found)
    korno      -> Krodhavashas 68  (Karna not in the top three)
    sokuni     -> Kuni 90          (a wrong name, above the cutoff)
    vishma     -> Havishmat 90     (a wrong name, above the cutoff)

This module turns every one of those spellings into the same key as the name
they mean: indic_key("Orjun") == indic_key("Arjuna") == "arjun".

It is a key for retrieval and nothing else. Two names sharing a key are not
asserted to be one person. Yudhisthira and Yudhishthira share a key and stay
separate entities, exactly as they do under the fuzzy match.

out/indic-phonetic.js is the same rule list for the Lineage mode, which runs
from file:// with no server. src/test_phonetic.py checks the two agree.
"""
import re
import unicodedata

# Order matters. Each rule sees the output of the one above it.
RULES = [
    (r"[^a-z ]", ""),                         # letters only; diacritics already stripped
    (r"ksh|kss|x", "ks"),                     # Ikshvaku / Ixvaku
    (r"chh|ch", "c"),                         # Chitrangada / Citrangada
    (r"sh", "s"),                             # Shakuni / Sakuni, Bhishma / Bhisma
    (r"ph|f", "p"),
    (r"([bcdgjkpt])h", r"\1"),                # aspiration: Bhima / Bima, Dhrita / Drita
    (r"[vw]", "b"),                           # Bengali: Vidura / Bidur, Vishma / Bhishma
    (r"z", "j"),
    (r"q", "k"),
    (r"(?<=[^aeiou ])y\b", "i"),              # Gandhary / Gandhari
    (r"^y|(?<= )y", "j"),                     # Bengali: Yudhishthira / Judhisthir
    (r"(?<=[^aeiou ])y", "j"),                # Abhimanyu / Obhimonyu, Duryodhana / Durjodhon
    (r"(?<=[^aeiou ])ri(?=[^aeiou ])", "r"),  # vocalic r: Krishna / Krsna (from Kṛṣṇa)
    (r"ee|ii|ie|ey", "i"),                    # Bheem / Bhima
    (r"oo|uu", "u"),                          # Arjoon
    (r"au|ou|ow", "o"),                       # Draupadi / Droupadi
    (r"ai|oi|ei", "e"),
    (r"aa", "a"),                             # Paandu / Pandu
    (r"o", "a"),                              # Bengali o for Sanskrit a: Orjun, Korno
    (r"(.)\1+", r"\1"),                       # doubles: Ashwatthama / Ashwathama
    (r"a\b", ""),                             # final schwa: Arjuna / Arjun, Korno / Karn
]
_COMPILED = [(re.compile(p), r) for p, r in RULES]


def strip_diacritics(text):
    """Kṛṣṇa -> Krsna. IAST typed into the box should not miss."""
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


def indic_key(name):
    s = strip_diacritics(str(name)).lower().replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    for pat, rep in _COMPILED:
        s = pat.sub(rep, s)
    return re.sub(r"\s+", " ", s).strip()


# Words the phonetic key must never be applied to. "part" keys to "part", which
# is Partha's key; "born" keys like Barna, "many" like Mani, "your" like Jara.
# Exact matching runs first, so a guarded word that really is a name still
# resolves when spelled the way the corpus spells it. The guard only stops the
# phonetic step from turning an English word into a person.
#
# CORPUS_WORDS is derived, not chosen: every word of four letters or more that
# the translation uses in lowercase at least five times (so it is a word, not a
# name) and whose key collides with a Person, Place or lineage label. A few are
# OCR fragments of the source text; they are kept because the derivation keeps
# them. src/test_phonetic.py re-derives the list from out/semantic.trig and
# fails if a new label makes it stale. QUESTION_WORDS covers modern words a
# question uses that a translation never does.
CORPUS_WORDS = frozenset("""
    afab agneya away baab baad baba babb bada badd bath bbaaad bbab bbad
    bbed beda bhrit bind boar boat boats born both call camp darbha dies
    fail feet fell forth four fourth goad harsh hour jest kara maid main
    many marries monkey month part path pour rakshas rich road roam sabha
    saddhyas sand sang sank seed shall shook shot show song sorrow sorry
    soul sound south sudha summon swan tail tamas vadha vain wash wind
    without woman wood worn worth your
""".split())
QUESTION_WORDS = frozenset("""
    data host info list relation related relationship explain story summary
    details describe killed father mother brother sister wife husband
    what which where when whom whose there their they them were have with
    from that this these those into upon also only like most more much some
    such than then very will would could should tell about give does name
    know mean meant
""".split())
GUARD = CORPUS_WORDS | QUESTION_WORDS
MIN_LENGTH = 4


def phonetic_key(token):
    """The key to look a typed token up by, or None if it must not be tried.

    Short tokens are refused because three letters carry too little to tell
    names apart: "son" keys like Shona and "war" like Vara.
    """
    t = strip_diacritics(str(token)).lower().strip()
    if len(t) < MIN_LENGTH or t in GUARD:
        return None
    key = indic_key(t)
    return key if len(key) >= 3 else None


def derive_corpus_words(passage_texts, labels, min_count=5):
    """Rebuild CORPUS_WORDS. Used by the test, not at runtime."""
    from collections import Counter
    counts = Counter()
    for text in passage_texts:
        counts.update(re.findall(r"(?<![A-Za-z])[a-z]{%d,}(?![A-Za-z])" % MIN_LENGTH, text))
    keys = {indic_key(l) for l in labels}
    lower = {l.lower() for l in labels}
    return {w for w, n in counts.items()
            if n >= min_count and w not in lower and indic_key(w) in keys}


if __name__ == "__main__":
    import sys
    for w in sys.argv[1:] or ["Arjuna", "Orjun", "Korno", "Kṛṣṇa", "Judhisthir"]:
        print(w, "->", indic_key(w))
