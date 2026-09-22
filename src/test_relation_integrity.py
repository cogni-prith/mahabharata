"""Regression checks for entity attachment and typed kinship, not an evaluation set."""
import spacy
from phase4_relations import ent_of, extract_sentence


def main():
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    persons = {"Karna", "Arjuna", "Bhishma", "Shantanu", "Radha", "Kunti", "Vasava", "Sanjaya", "Duryodhana", "Nakula", "Vrisha", "Krishna"}

    def extract(text, coref=None):
        out = []
        r = {"parva": "test", "sent_id": 0, "adhyaya_ordinal": None, "chapter_cited": False, "depth": 0}
        extract_sentence(nlp(text), r, persons, {"Chedi", "Madra"}, out, coref)
        return out

    text = "I, Vasava's son, spoke to Karna."
    doc = nlp(text)
    assert ent_of(doc[0], persons, set(), [(0, len(text), "Sanjaya", 3)])[0] is None, "First person must not inherit an enclosing cluster"
    text = "His chariot approached Karna."
    doc = nlp(text)
    assert ent_of(doc[0], persons, set(), [(0, 11, "Sanjaya", 3)])[0] is None, "A possessor is not the root of the resolved phrase"
    text = "He spoke to Karna."
    assert ent_of(nlp(text)[0], persons, set(), [(0, 2, "Sanjaya", 3)])[0] == "Sanjaya"
    assert ent_of(nlp(text)[0], persons, set(), [(0, 2, "Sanjaya", 3), (0, 2, "Duryodhana", 3)])[0] is None
    rows = extract("I had Bhishma, Shantanu's son, Karna, and Arjuna as protectors.")
    assert not any(r["subject"] == "Karna" and r["object"] == "Shantanu" for r in rows)
    rows = extract("Arjuna is Vasava's son.")
    assert any(r["predicate"] == "hasDivineProgenitor" and r["subject"] == "Arjuna" for r in rows)
    assert not extract("Arjuna is not Vasava's son.")
    assert not extract("Is Arjuna Vasava's son?")
    rows = extract("Karna is Radha's foster son.")
    assert any(r["predicate"] == "hasFosterParent" for r in rows)
    rows = extract("Arjuna is Kunti's biological son.")
    assert any(r["predicate"] == "hasBiologicalParent" for r in rows)
    rows = extract("Karna, the son of Radha, spoke.")
    assert rows and all(r["predicate"] == "bornTo" for r in rows), "Generic son must not assert biological parentage"
    assert not any(r["predicate"].endswith("hasBiologicalParent") for r in extract("Karna is Kunti's son.")), \
        "and must not acquire one through the possessive form either"
    assert not extract("He was slain by Nakula.", [(0, 2, "Karna", 3)]), "A model-resolved pronoun cannot establish a death"
    text = "Vrisha destroyed it."
    assert not extract(text, [(text.index("it"), text.index("it")+2, "Karna", 3)]), "Destroyed equipment is not a slain person"
    assert not extract("Arjuna destroyed Karna."), "Destroy does not establish a literal death"
    assert any(r["subject"] == "Karna" and r["object"] == "Arjuna" and r["predicate"] == "slainBy"
               for r in extract("Karna was killed by Arjuna."))
    for text in ("Karna attacked him.", "Karna defeated him.", "He was defeated by Karna."):
        start = text.index("him") if "him" in text else 0
        end = start + (3 if "him" in text else 2)
        assert not extract(text, [(start, end, "Arjuna", 3)]), "An unverified antecedent must not become a conflict fact"
    assert any(r["predicate"] == "foughtWith" for r in extract("Karna attacked Arjuna."))
    assert any(r["predicate"] == "defeatedBy" for r in extract("Arjuna was defeated by Karna."))
    assert not extract("Releasing arrows towards Bhishma, Krishna\u2019s son roared loudly."), "A new clause's subject is not an epithet of the previous object"
    for text in ("On hearing the words of Bhishma, the immensely valorous king of Chedi desired to fight.",
                 "In that encounter for the sake of Bhishma, the valiant king of Madra was enraged."):
        assert not any(r["predicate"] == "rules" for r in extract(text)), text
    for text in ("Bhishma, Shantanu's son, roared.", "I spoke to Bhishma, Shantanu's son."):
        assert any(r["subject"] == "Bhishma" and r["object"] == "Shantanu" for r in extract(text)), text
    print("Relation integrity checks passed: coreference boundaries, ambiguity, list attachment, kinship types")


if __name__ == "__main__":
    main()
