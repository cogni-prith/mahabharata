"""The guard that refuses "X's son" as a coreference argument.

Run: .venv/bin/python src/test_possessed_kin.py
"""
import spacy
from phase4_relations import possessed_kin

nlp = spacy.load("en_core_web_sm", disable=["ner"])


def head(text, word):
    return next(t for t in nlp(text) if t.lower_ == word)


def main():
    # The sentence that produced (Abhimanyu, slainBy, Drona).
    assert possessed_kin(head("Abhimanyu is then killed by Duhshasana's son.", "son"))
    assert possessed_kin(head("The son of Drona rushed at him.", "son"))
    assert possessed_kin(head("His brother struck Karna.", "brother"))
    # A kinship noun with no possessor still names one person through coref:
    # "the son said" in a passage about one man is not a relational description.
    assert not possessed_kin(head("The son said this to Bhishma.", "son"))
    # Not a kinship noun at all.
    assert not possessed_kin(head("Drona's charioteer fell.", "charioteer"))
    print("ok")


if __name__ == "__main__":
    main()
