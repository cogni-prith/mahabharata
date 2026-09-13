"""One definition of the entity vocabulary, shared by Phase 2 and Phase 4.

There were two. Phase 2 declared entities in `out/entities.ttl` from PERSON
surfaces seen at least ten times, which is the threshold its apposition evidence
needs. Phase 4 linked relations to any PERSON surface of the right shape. The
graph therefore referred to 251 entity nodes that were never declared, because
the evidence threshold for a merge had leaked into the question of whether an
entity exists at all. It does not: an entity exists if the annotation labelled a
surface as one. Thresholds belong where evidence is weighed.
"""
import re
import pandas as pd

# Capitalised words the annotation labelled PERSON that are not names. Sentence
# initial pronouns, determiners and honorifics, which the BIO layer tags on shape.
NOT_A_NAME = {"You", "He", "She", "They", "We", "It", "Thus", "Then", "Having",
              "There", "Here", "This", "That", "These", "Those", "But", "And",
              "His", "Her", "Their", "Your", "Who", "What", "When", "Where",
              "Thou", "Thee", "Thy", "Yet", "Now", "One", "Some", "All"}

SHAPE = re.compile(r"[A-Z][a-z]{2,}")


def vocabulary(mentions="build/mentions.parquet"):
    """(persons, places, mention counts, mentions frame) from the mention layer.

    A surface takes the label it carries most often, so a name tagged PERSON in
    900 places and GPE in two stays a person.
    """
    df = pd.read_parquet(mentions) if isinstance(mentions, str) else mentions
    s = df.surface.str.strip()
    lab = df.groupby(s).label.agg(lambda x: x.value_counts().idxmax())
    counts = s.value_counts()
    persons = {n for n, l in lab.items()
               if l == "PERSON" and SHAPE.fullmatch(n) and n not in NOT_A_NAME}
    places = {n for n, l in lab.items()
              if l in ("GPE", "LOC") and SHAPE.fullmatch(n)}
    return persons, places, counts, df


if __name__ == "__main__":
    p, pl, c, _ = vocabulary()
    assert "Arjuna" in p and "Hastinapura" in pl or "Kashi" in pl, "vocabulary looks wrong"
    assert not (p & NOT_A_NAME), "stoplist not applied"
    print("persons %d places %d, Arjuna %d mentions" % (len(p), len(pl), c["Arjuna"]))
