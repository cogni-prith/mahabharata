"""Grounded chat over the asserted graph and the passage index.

Retrieval is hybrid on purpose. The graph half answers structured questions
precisely and cannot answer anything it never extracted; the passage half
answers anything phrased loosely and cannot be precise. Each covers the
other's blind spot, and both are handed to the model as context it must
answer from.

Three rules this file exists to keep:

An appellation is never a resolution. `build/appellations.csv` holds 246
candidate links over 121 appellations and not one of them is merged, because
the evidence does not settle them. Achyuta has five candidates and the best
scores 0.25. So a query naming an epithet may retrieve the candidate's
records, and the answer must say candidate.

Every fact carries its citation or says it has none. A line with no
`(parvan, adhyaya)` prints `uncited`, which is the honest state, not a gap to
fill in.

The remote call is opt-in. Without `GROQ_API_KEY` this prints the retrieved
context and stops, which is also the way to inspect what the model would have
been told. Only the question and that context leave the machine.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_entities import norm_surface
from indic_phonetic import indic_key, phonetic_key

ROOT = Path(__file__).resolve().parents[1]
# The reviewed set, not the raw extractor output. `src/review_viewer_graph.py`
# re-runs every relation record through the current extractor guards and
# withholds what no longer passes. The raw file still carries rows those guards
# were written to reject, because it was produced before them: it says Karna was
# slain by Nakula, from "He was slain by the great-souled Nakula", where "He" is
# a pronoun that coreference attached to Karna. Reading the reviewed file is
# what makes those guards reach an answer.
RELATIONS = ROOT / "build/reviewed_relations.parquet"
RAW_RELATIONS = ROOT / "build/relations.parquet"
APPELLATIONS = ROOT / "build/appellations.csv"
ENTITIES = ROOT / "out/entities.ttl"
MENTIONS = ROOT / "build/mentions.parquet"

ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
# Chosen from what the account actually serves, read off /v1/models rather
# than assumed: the largest plain chat model on offer, with a context window
# far wider than any block this builds. Not groq/compound, which carries its
# own web search: a grounded answer must not have a route back to the open
# internet. `--model` overrides this, and /v1/models lists the alternatives.
MODEL = "openai/gpt-oss-120b"
# Cloudflare rejects urllib's default agent with error 1010 before the request
# reaches the API, which looks like an auth failure and is not one.
AGENT = "mahabharata-kg-chat/1.0"
# Below this an edit-distance hit is noise. "Arjun" scores 90 against Arjuna,
# "Yudhistira" 96 against Yudhishthira; unrelated names sit in the fifties.
CUTOFF = 85
# A name has to be one the corpus actually uses. The declared entity set
# includes annotation artifacts: "Tell" is a PERSON with 2 mentions and
# "Yudhishtira" is a one-off misspelling of a name appearing 2,063 times.
# Without a floor, "tell me about Yudhistira" resolves to both of them. Ten is
# the threshold phase2_entities already applies to its own name list. Genuine
# names below it stay reachable through passage retrieval, which is the half
# of this that does not need a name at all.
MIN_MENTIONS = 10
# Spelling variants are separate entities in this graph and nothing merges
# them, so a misspelling can legitimately land near several of them at once.
# Returning all of them retrieves both sets of records; returning one would be
# picking a referent, which is the choice this project does not make.
FUZZY_KEEP = 3
# Edit distance cannot reach a name spelled the way a Bengali or Hindi speaker
# writes it. "Orjun" scores 73 against Arjuna and "Korno" does not have Karna in
# its top three, while "sokuni" scores 90 against Kuni and "vishma" 90 against
# Havishmat, both wrong. src/indic_phonetic.py reduces every such spelling to one
# key, so the phonetic step runs before the fuzzy one. A shared key is a lead
# for retrieval, never a merge: Yudhisthira and Yudhishthira share one and come
# back as two entities, exactly as they do under the fuzzy match.
#
# When the key itself is not an exact hit, it is compared by plain edit ratio as
# a last resort, after the WRatio pass has found nothing. Plain ratio, not
# WRatio: WRatio rewards substrings, which is how "bhim" lands on Abhimanyu.
PHONETIC_CUTOFF = 88
# How many epithets of a resolved entity to pull records from. Sorted by
# apposition confidence, so the best-evidenced come first. Bounded because the
# facts they bring are leads, not the answer, and must not crowd out the
# records filed under the name actually asked about.
EPITHET_KEEP = 6
# Patterns whose `evidence` column is a description of how the row was derived
# rather than a quotation from the text. `vocative` rows come from the Phase 3
# telling layer and all 459 of them carry the same string. Printed as evidence
# it reads like a source quote, costs tokens on every request, and is not one.
UNQUOTED_PATTERNS = {"vocative"}
# Words in the question that suggest which predicates answer it. This orders
# the budget, it never filters: every predicate still gets its turn, the
# relevant ones just get it first. Prompting alone did not hold. Told to lead
# with parentage, the model still answered "no father is recorded" with six
# bornTo rows sitting in the context, because they were below thirty speaksTo
# rows. Position in the block decides what gets read.
PREDICATE_HINTS = {
    "bornTo": ("father", "mother", "parent", "parents", "born", "son", "daughter",
               "child", "children", "sire", "descend", "lineage", "family"),
    "hasDivineProgenitor": ("father", "divine", "god", "progenitor", "born", "parent"),
    "slainBy": ("kill", "killed", "kills", "slay", "slain", "slew", "death", "died",
                "die", "murder", "destroyed"),
    "foughtWith": ("fight", "fought", "battle", "war", "against", "opponent", "duel"),
    "defeatedBy": ("defeat", "defeated", "beat", "lost", "won", "vanquish"),
    "instructedBy": ("teach", "taught", "teacher", "learn", "guru", "student",
                     "preceptor", "instruct", "train"),
    "hasSpouse": ("wife", "husband", "married", "marry", "spouse", "wed"),
    "cursedBy": ("curse", "cursed", "curses"),
    "rules": ("rule", "ruled", "king", "kingdom", "reign", "throne", "ruler"),
    "speaksTo": ("said", "say", "spoke", "speak", "told", "address", "talk"),
}
# Trimmed from 40 once the budget became question-aware. The free tier allows
# 200,000 tokens a day and a question was costing 7,000 of them, which is 28
# questions. With the rows that answer the question at the top of the block,
# the tail was padding the bill rather than the answer.
MAX_FACTS = 28
MAX_PASSAGES = 5
HISTORY_TURNS = 6
# Evidence quotes are the bulk of the block. The first rows are the ones the
# answer is built from, so they keep their quote and the tail keeps its
# citation, which is what provenance actually requires.
QUOTED_FACTS = 12
MAX_NOTES = 8
MAX_SIBLINGS = 12
# Plain-language mode, the default for the explorer and the CLI. An appellation
# resolves to its single best-supported candidate (Partha to Arjuna), and the
# answer is written for a general reader. This is a deliberate exception to the
# rule against guessing an entity link, taken on instruction for readability.
# MBH_STRICT_NAMES=1 restores strict mode. The Retriever default stays strict,
# so src/test_chat.py keeps testing the rule it was written for.
SIMPLE = os.environ.get("MBH_STRICT_NAMES") != "1"
SIBLING_HINTS = ("brother", "sister", "sibling")
# A family question: words whose stems mark kinship. Plain-language mode then
# adds a FAMILY block reaching two steps out from the people named.
KIN_HINTS = ("father", "mother", "parent", "son", "daughter", "child", "brother", "sister",
             "sibling", "grand", "uncle", "aunt", "cousin", "nephew", "niece", "wife",
             "husband", "spouse", "marri", "born", "divine", "relat", "family", "lineage",
             "descend", "ancestor", "progenitor")
# Gods of the curated family table's cosmic clan. A row there with one of them
# as the parent is a divine parent, as Dharma for Yudhishthira and Vayu for Bhima.
DIVINE = {"indra", "surya", "vayu", "dharma", "ashwins", "ashvins", "agni", "varuna", "yama"}
MAX_FAMILY = 40
LINEAGE = "annotations/lineage/mahabharata_lineage.csv"
# Popular spellings the corpus does not use. Most popular forms only drop the
# final a (Arjun, Karn, Bhishm) and resolve() handles those by rule; these are
# the ones a rule cannot reach. Values are corpus spellings, lower case.
SPELLINGS = {
    "bheem": "bhima", "bhimsen": "bhimasena", "bheemsen": "bhimasena",
    "draupadi": "droupadi", "dropadi": "droupadi", "sahdev": "sahadeva",
    "ashwatthama": "ashvatthama", "ashwathama": "ashvatthama", "ashvathama": "ashvatthama",
    "dronacharya": "drona", "kripacharya": "kripa",
    "balram": "baladeva", "balaram": "baladeva", "balarama": "baladeva",
    "kansa": "kamsa", "eklavya": "ekalavya", "parikshita": "parikshit",
    "dushasana": "duhshasana", "dushasan": "duhshasana",
    "yudhishtir": "yudhishthira", "yudhishtira": "yudhishthira",
}
# Words of the question itself, never names. Loose matching turned "and" into
# Bandi, Chandra and Dandadhara.
NOT_NAMES = {
    "who", "whom", "whose", "what", "which", "when", "where", "why", "how", "was",
    "were", "are", "the", "and", "did", "does", "his", "her", "their", "same",
    "person", "people", "brother", "sister", "father", "mother", "son", "sons",
    "daughter", "wife", "husband", "killed", "kill", "teacher", "about", "tell",
    "with", "from", "that", "this", "name", "names", "between", "related", "also",
}
SHORT = 6       # tokens this long or shorter are matched on plain ratio, not WRatio

SYSTEM = """You answer questions about the Mahabharata from the context below and from nothing else.

The user may write with spelling or grammar errors. Correct them silently and answer the intended question. Never remark on the errors.

Answer the question first, in one sentence, before any detail.

Rival statements are not equals. Weigh them by `support`, how many separate places state it, then by confidence. Lead with the best-supported answer and say plainly that it is the best supported. Mention weaker rivals after it, as weaker, with their numbers. Do not present a flat list of candidates as though the corpus cannot decide when the numbers clearly favour one.

Where a section of names that may be the same person is given, use it before you count rivals. Several rival answers may be one person under several epithets, and saying "five killers" when four of the names carry evidence pointing at one man is wrong. Say which names may co-refer and with what confidence, and then give the count of distinct people the evidence actually supports.

Rules you may not break:
- Use only the context. If it does not answer the question, say what the corpus does hold on that subject before saying what it lacks. "No father is recorded, though Kunti is recorded as a parent" is useful; "the context does not contain that" alone is not.
- Cite the parva and chapter for every claim you make, exactly as the context gives it. If a fact is marked uncited, say it is uncited.
- An appellation candidate is not an identity. If the context lists a name under UNRESOLVED NAMES, say in your FIRST sentence that the corpus does not resolve it, before naming any candidate. Do not open by stating who it refers to and qualify it afterwards. Give the candidate, its confidence and how many rivals it has. Never write that the two are the same person, that one "is" the other, or that a name "refers to" or "is used for" someone.
- Facts marked predicted or proposed are model output, not statements the text makes. Label them that way.
- The text does not settle every question. Where the context is thin or contradictory, say that instead of choosing.

Answer in at most 150 words and at most six bullets. Never more.

Select, do not enumerate. The context is everything retrieved about the names in the question, not a list of things to recite. Most of it will be irrelevant to what was asked. A dump of every triple is a failure even when every line in it is true.

For "who is X" or "tell me about X", describe the person in prose from the source passages, and use at most two or three relations that actually characterise them, such as parentage, rule, or a decisive act. Never list who they spoke to. A high support count on speaksTo means the corpus records a lot of dialogue, not that the dialogue defines them.

For a question about one specific relation, answer that relation and stop.

`bornTo` is a generic parent and does not encode mother or father; the graph stores kinship primitives and never guessed which. Asked for someone's father or mother, list every recorded parent, name the predicate's limit in one clause, and let the reader draw the conclusion. Never answer "no father is recorded" while a `bornTo` row for that person is sitting in the context unmentioned. `hasDivineProgenitor` is a separate and weaker thing: report it as a divine progenitor, never as the father.

Write short prose with short bullet lists where a list helps. Use "- " for bullets and **bold** for a name you are listing. No tables, no headings, no horizontal rules. Put each citation in brackets right after the claim it supports, like (08_Karna 69).

Context:
{context}"""

SYSTEM_SIMPLE = """You answer questions about the Mahabharata for a general reader who knows nothing about it. Use only the context below. Never add anything from your own knowledge, even when you are sure of it.

The user may write with spelling or grammar errors. Correct them silently and answer the intended question.

How to write:
- Plain, everyday English. Put the direct answer in the first sentence, then at most two more short sentences.
- If a list helps, use at most five short "- " bullets. No tables, no headings.
- Never mention confidence, support, scores, sightings, candidates, patterns, "the corpus", "the graph", "the context" or "the records". Never write relation names such as bornTo or slainBy; say "parent", "was killed by", "fought", and so on.
- At most 80 words, not counting the Source line.

Names:
- Under NAMES, every name in the question is matched to one person. Treat it as that person. Write the person as the question named them. Only when the question used another name for them, and only the first time, add the common name in brackets: asked about Dhananjaya, write "Dhananjaya (Arjuna)"; asked about Arjuna, write just "Arjuna". Never say a name is unresolved, uncertain or ambiguous.
- If a NAMES line says the name is also used almost as often for others, you must say so once, as a short clause, the first time you use the name: "Partha (here Yudhishthira; the name is also used for Arjuna and Bhima)". Then answer for the matched person only.
- A name listed as another name for someone is that person. Treat facts recorded under it as facts about them.

Facts:
- Give the answer stated most often (highest support). Mention a rival only if it is stated as often.
- A parent fact tagged [X is the mother] or [X is the father] tells you which. Use it plainly: "Arjuna's mother is Kunti and his father is Pandu". A parent with no tag is only known to be a parent: say "parent" for that one, unless the evidence quote itself says father or mother. Asked for a mother or father that no tag or quote identifies, name the recorded parents and say the text here does not mark which is which. Never decide it from how often a parent is mentioned.
- SIBLINGS lists people who share a recorded parent. Asked about brothers or sisters, name every person in SIBLINGS, and call them brothers or sisters plainly.
- For any family question, answer from FAMILY first. It already reaches two steps out: a person's parents and grandparents, and a person's children with each child's other parents, including divine fathers. A maternal grandparent is the mother's parent; a paternal grandparent is the father's parent. If FAMILY does not mark the side, say "grandparent through" that parent. Names in FAMILY are already merged, so never list one person twice under two names.
- A link from the "curated family table" is from the project's hand-made family table and cites no passage. Use it, and mark it once in plain words, such as "(from the family table)". Put "family table" in the Source line for it.
- KINSHIP PATH is the shortest chain of family links between two people. Explain it in plain words, step by step, and name the relationship it adds up to only when the chain makes it plain (a parent's sibling's child is a cousin; a spouse's sibling is an in-law).
- Facts marked predicted or proposed are a model's guesses; if you use one, say "possibly".
- If the context does not answer the question, say so in one sentence and give the closest thing it does say.
- The facts record only these kinds of relation: parent, divine parent, spouse, ruled, fought, was killed by, was defeated by, was cursed by, was taught by, spoke to, and siblings worked out from parents. Asked about any other kind (ally, friend, enemy, uncle, grandson, charioteer, and so on), say in your first sentence that this kind of relation is not recorded, then give the closest facts under their real meaning. Never present "spoke to" as alliance, friendship or loyalty.

End with one line starting "Source:" listing up to three chapter references exactly as the context gives them, like 08_Karna 69. Write "uncited" for a fact with no chapter. If the answer is that something is not recorded, cite the facts you gave instead; leave the Source line out only if you gave none.

Context:
{context}"""


def entity_labels(path=ENTITIES):
    """Person and Place labels from the asserted graph.

    Read off the Turtle by hand rather than through rdflib: the file is
    generated by string concatenation in phase2_entities.write_entities_ttl
    and every label sits on one line, so parsing 400 KB through a graph
    library buys nothing here.
    """
    labels = set()
    for line in Path(path).read_text().splitlines():
        if "mbh:Person" not in line and "mbh:Place" not in line:
            continue
        _, _, rest = line.partition('rdfs:label "')
        label, _, _ = rest.partition('"')
        if label:
            labels.add(label)
    return labels


class Retriever:
    """Everything the chat layer knows, loaded once.

    The passage index is optional. It costs a sentence-transformer load and a
    34 MB read, so a caller that only wants graph facts, and every test in
    src/test_chat.py, can skip it.
    """
    # Strict mode unless __init__ says otherwise; src/test_phonetic.py builds
    # its stand-in without calling __init__.
    top_alias = False
    alias_by_norm = {}

    def __init__(self, root=ROOT, passages=True, index=None, top_alias=False):
        root = Path(root)
        self.top_alias = top_alias
        reviewed = root / "build/reviewed_relations.parquet"
        self.reviewed = reviewed.exists()
        raw = root / "build/relations.parquet"
        if self.reviewed or raw.exists():
            self.rel = pd.read_parquet(reviewed if self.reviewed else raw)
        else:
            # Phase 4 has not run here (coreference needs a GPU). Serve passages
            # with no graph facts rather than refuse to start; nothing is invented.
            print("No relations parquet in build/: Ask answers from passages only.", flush=True)
            self.rel = pd.DataFrame(columns=["subject", "predicate", "object", "status", "confidence",
                                             "pattern", "parva", "adhyaya_ordinal", "cited", "evidence"])
        if self.reviewed:
            self.rel = self.rel[self.rel.status == "asserted"]
        self.app = pd.read_csv(root / "build/appellations.csv")
        graph_names = set(self.rel.subject) | set(self.rel.object)
        self.names = sorted(graph_names | entity_labels(root / "out/entities.ttl"))
        counts = pd.read_parquet(root / "build/mentions.parquet",
                                 columns=["surface"]).surface.value_counts()
        self.counts = {n: int(counts.get(n, 0)) for n in self.names}
        self.by_norm = {}
        for n in self.names:
            if self.counts[n] < MIN_MENTIONS:
                continue
            # On a normalized collision keep the spelling the corpus prefers.
            cur = self.by_norm.get(norm_surface(n))
            if cur is None or self.counts[n] > self.counts[cur]:
                self.by_norm[norm_surface(n)] = n
        self.appellations = {}
        for a in self.app.appellation.unique():
            self.appellations.setdefault(norm_surface(a), a)
        # Phonetic keys over the same floored name list, so the mention floor
        # that keeps "Tell" out of exact and fuzzy matching keeps it out here
        # too. Within a key, the spelling the corpus uses most comes first.
        self.by_key = {}
        for n in self.by_norm.values():
            self.by_key.setdefault(indic_key(n), []).append(n)
        for names in self.by_key.values():
            names.sort(key=lambda n: -self.counts[n])
        self.appellations_by_key = {}
        for a in self.appellations.values():
            self.appellations_by_key.setdefault(indic_key(a), []).append(a)
        # Plain-language mode: each name's single best person. Phase 2b's
        # mention scores decide it, as total confidence over every mention it
        # won outside the reject band, which weighs how often and how surely.
        # The appellation table only fills names Phase 2b never scored; its
        # top candidates include known errors (Kartavirya to Arjuna).
        ranked = self.app.sort_values(["confidence", "evidence_count"],
                                      ascending=False, na_position="last")
        self.top_of = ranked.drop_duplicates("appellation").set_index(
            "appellation").candidate_entity.to_dict()
        self.also_of = {}
        self.alias_cites = {}
        links = root / "build/er/mention_links.csv"
        if top_alias and links.exists():
            ml = pd.read_csv(links, usecols=["surface", "candidate", "confidence", "action",
                                             "parva", "adhyaya_ordinal", "sources"])
            # A link proposed by spelling likeness alone is not evidence of
            # identity: it mapped Shura, Kunti's father, onto the sage Shukra.
            # Seed, apposition and identity links remain.
            mass = (ml[(ml.action != "reject") & (ml.sources != "spelling")]
                    .groupby(["surface", "candidate"])
                    .confidence.sum().sort_values(ascending=False))
            for surface, group in mass.groupby(level=0, sort=False):
                cands = group.droplevel(0)
                self.top_of[surface] = cands.index[0]
                # A runner-up within 10% is a near tie, and the answer says so.
                self.also_of[surface] = [c for c, v in cands.iloc[1:].items()
                                         if v >= 0.9 * cands.iloc[0]]
            # Where the text itself uses the name for that person: the most
            # confident accepted mentions, so a "same person?" answer has a
            # chapter to cite instead of "uncited".
            acc = ml[ml.action.isin(["auto_accept", "accept_sample_verify"])
                     & ml.adhyaya_ordinal.notna()].sort_values("confidence", ascending=False)
            for (surface, cand), g in acc.groupby(["surface", "candidate"], sort=False):
                if self.top_of.get(surface) == cand and surface != cand:
                    cites = [citation(p, a, True) for p, a in zip(g.parva, g.adhyaya_ordinal)]
                    self.alias_cites[surface] = list(dict.fromkeys(cites))[:2]
        self.alias_by_norm = {norm_surface(s): s for s, c in self.top_of.items() if c != s}
        # Mother or father for a parent, from Phase 4s GLiREL counts, where
        # they are lopsided enough to call. Plain-language mode only; strict
        # mode keeps bornTo as the generic parent it is. Absent file, no roles.
        roles = root / "build/parent_roles.csv"
        self.parent_roles = {}
        if top_alias and roles.exists():
            pr = pd.read_csv(roles)
            self.parent_roles = {p: r for p, r in zip(pr.parent, pr.role) if r in ("mother", "father")}
        self._moved_init_tail(root, passages, index)
        self.kin = self._build_kin(root) if top_alias else ({}, {}, {})

    def canon(self, name):
        """A name as the graph spells the person: Bhima -> Bhimasena, Draupadi -> Droupadi.

        A bracketed name is a different person by the curated table's own
        convention ("Kunti (ancestor)", "Karna (Kaurava)") and is left as
        written, unless the bracket names the same person as the name before
        it: "Pritha (Kunti)" is Kunti.
        """
        if "(" in name:
            base, _, note = name.partition("(")
            note = note.rstrip(") ").strip()
            same = self.canon(note) if note and "(" not in note else None
            if same and same in self.by_norm.values() and self.canon(base) == same:
                return same
            return name.strip()
        n = norm_surface(name).strip()
        n = SPELLINGS.get(n, n)
        if n in self.alias_by_norm:
            return self.top_of[self.alias_by_norm[n]]
        return self.by_norm.get(n, name.strip())

    def _build_kin(self, root):
        """Parent, child and spouse links, from the graph and the curated table.

        Graph rows keep their citation. A row whose end is a leftover epithet
        node ("Kounteya", "Dharmaraja") is skipped: Phase 4f already folded the
        confident mentions into their person, so what is left under the epithet
        is exactly the doubtful part, and "Kounteya" would hand Bhima's divine
        father to Yudhishthira. A bare title on one row ("Ajamidha") is skipped
        as well. The curated table is hand-made and uncited; its links are
        labelled so, and never override a link the text states.
        """
        up, down, spouse = {}, {}, {}
        kinds = {"bornTo": "parent", "hasDivineProgenitor": "divine"}
        titles = set(self.appellations.values())
        curated = []
        path = root / LINEAGE
        if path.exists():
            for r in pd.read_csv(path).itertuples():
                par, child = self.canon(str(r.Parent)), self.canon(str(r.Child))
                if par != child:
                    curated.append((par, child))
        curated_parents = {}
        for par, child in curated:
            curated_parents.setdefault(child, set()).add(par)
        rel = self.rel[self.rel.predicate.isin(list(kinds) + ["hasSpouse"])]
        rel = rel.sort_values("cited", ascending=False)
        for (s, p, o), g in rel.groupby(["subject", "predicate", "object"], sort=False):
            if any(norm_surface(n) in self.alias_by_norm for n in (s, o)):
                continue
            # A bare title on one row ("Ajamidha"). A real person who also sits in
            # the appellation table, as Bhimasena does, is kept when the curated
            # table knows them.
            known = set(curated_parents) | {q for ps in curated_parents.values() for q in ps}
            if len(g) < 2 and any(n in titles and n not in known for n in (s, o)):
                continue
            # A parent stated once that the curated table contradicts is most
            # likely a name clash: "Nakula, Drona's son", "Jarasandha's son,
            # Sahadeva", and Vyasa read as Krishna in "Krishna, Satyavati's son".
            if (p == "bornTo" and len(g) < 2 and s in curated_parents
                    and o not in curated_parents[s]):
                continue
            r0 = g.iloc[0]
            link = {"source": "text", "cite": citation(r0.parva, r0.adhyaya_ordinal, r0.cited),
                    "support": len(g)}
            if p == "hasSpouse":
                spouse.setdefault(s, {}).setdefault(o, link)
                spouse.setdefault(o, {}).setdefault(s, link)
            else:
                up.setdefault(s, {}).setdefault(o, dict(link, kind=kinds[p]))
                down.setdefault(o, {}).setdefault(s, dict(link, kind=kinds[p]))
        for par, child in curated:
            link = {"source": "curated", "cite": "family table", "support": 0,
                    "kind": "divine" if par.lower() in DIVINE else "parent"}
            up.setdefault(child, {}).setdefault(par, link)
            down.setdefault(par, {}).setdefault(child, link)
        return up, down, spouse

    def _moved_init_tail(self, root, passages, index):
        # The reverse index: given an entity, which appellations name it. Asked
        # for Arjuna's father the graph looked empty, and the father was there
        # the whole time under his epithets. bornTo Pandu is stated seven times
        # of Dhananjaya, once of Savyasachi and once of Phalguna, and
        # hasDivineProgenitor Indra of Dhananjaya. Retrieval only ever looked
        # up the literal string typed, so none of it was reachable.
        self.epithets_of = {}
        evidenced = self.app[(self.app.status != "prior_only")
                             & self.app.confidence.notna()]
        for _, r in evidenced.sort_values("confidence", ascending=False).iterrows():
            self.epithets_of.setdefault(r.candidate_entity, []).append(
                {"appellation": r.appellation, "confidence": float(r.confidence),
                 "evidence_count": int(r.evidence_count), "status": r.status,
                 "basis": "apposition"})
        # The corpus prior, used to widen retrieval and for nothing else.
        # Apposition alone does not reach Dhananjaya from Arjuna: its best
        # apposition candidate is Partha, which is another epithet of the same
        # man. The prior names Arjuna directly, and it is right here for the
        # same reason it is right in 24 other rows, where the apposition
        # candidate is a sibling epithet rather than the canonical name.
        #
        # This is the prior CLAUDE.md calls weak and never truth, so it is
        # fenced. It only ever widens what is retrieved, never asserts. It is
        # labelled `basis: prior` and carries no confidence, because it has
        # none to carry. And it is gated on the appellation having real
        # apposition evidence for something, which is what keeps the known
        # failure out: Bharadvaja is prior_only and stays excluded.
        for _, r in evidenced.drop_duplicates("appellation").iterrows():
            if pd.isna(r.prior_says) or r.prior_says == r.appellation:
                continue
            bucket = self.epithets_of.setdefault(r.prior_says, [])
            if not any(e["appellation"] == r.appellation for e in bucket):
                bucket.append({"appellation": r.appellation, "confidence": None,
                               "evidence_count": 0, "status": "corpus_prior",
                               "basis": "prior"})
        # The explorer service already holds a SearchIndex, and it owns a
        # sentence encoder and 34 MB of vectors. Share it rather than paying
        # for a second copy inside the same process.
        self.index = index
        if passages and index is None:
            from semantic_search import SearchIndex
            self.index = SearchIndex(root)

    def resolve(self, query):
        """Names the query might be about, each labelled with how it was found.

        Returns a list of dicts. `resolved` is False for anything reached
        through an appellation, and stays False however confident the
        candidate looks.
        """
        tokens = []
        for t in norm_surface(query).split():
            # norm_surface keeps punctuation, so "Vrikodara?" missed the exact
            # and alias lookups and fell through to a fuzzy match on a node
            # with no facts. A typed question nearly always ends in one.
            t = t.strip("?!.,;:\"'()[]")
            if len(t) <= 2 or t in NOT_NAMES:
                continue
            known = (t in self.by_norm or t in self.appellations
                     or (self.top_alias and t in self.alias_by_norm))
            # The phonetic step below reaches most popular spellings (Bheem,
            # Arjun) and says it did. The table and the dropped-a rule are for
            # the ones it cannot reach (Balram, Karn).
            key = None if known else phonetic_key(t)
            if not known and not (key and self.by_key.get(key)):
                t = SPELLINGS.get(t, t)
                known = (t in self.by_norm or t in self.appellations
                         or (self.top_alias and t in self.alias_by_norm))
                # The popular form often drops the final a: Arjun, Karn, Bhishm.
                if not known and (t + "a" in self.by_norm or t + "a" in self.appellations
                                  or (self.top_alias and t + "a" in self.alias_by_norm)):
                    t += "a"
            tokens.append(t)
            # norm_surface drops "Arjuna's" but not "arjunas", and a dropped
            # apostrophe is the commonest thing in a typed question. The bare
            # stem is worth trying because "arjunas" scores 83 against Arjuna,
            # just under the cutoff, while "arjuna" is exact.
            if t.endswith("s") and len(t) > 4:
                tokens.append(t[:-1])
        found, seen, aliased = [], set(), set()

        def add(rec):
            key = (rec["name"], rec["via"])
            if key not in seen:
                seen.add(key)
                found.append(rec)

        for tok in tokens:
            # Plain-language mode: a name whose best person is someone else
            # ("Partha") answers as that person, not as the leftover graph node.
            if self.top_alias and tok in self.alias_by_norm:
                # Consumed even when add() drops it as a repeat: "vrikodara and
                # bheem" names Bhimasena twice, and the second must not fall
                # through to the fuzzy pass and land on King Bhima of Vidarbha.
                aliased.add(tok)
                s = self.alias_by_norm[tok]
                add({"name": self.top_of[s], "via": "alias", "resolved": True,
                     "score": 100.0, "surface": tok, "appellation": s,
                     "also": self.also_of.get(s, []), "cites": self.alias_cites.get(s, [])})
                continue
            if tok in self.by_norm:
                add({"name": self.by_norm[tok], "via": "exact", "resolved": True,
                     "score": 100.0, "surface": tok})
            if tok in self.appellations:
                self._expand(add, self.appellations[tok], 100.0)

        matched = {r["surface"] for r in found}
        # Sound before spelling. Runs only on tokens that matched nothing
        # exactly, and phonetic_key refuses short tokens and English words
        # whose key happens to collide with a name ("part" and Partha).
        for tok in tokens:
            if tok in matched:
                continue
            key = phonetic_key(tok)
            if not key:
                continue
            names = self.by_key.get(key, [])[:FUZZY_KEEP]
            for name in names:
                add({"name": name, "via": "phonetic", "resolved": True,
                     "score": 100.0, "surface": tok, "key": key})
            if not names:
                for a in self.appellations_by_key.get(key, [])[:1]:
                    self._expand(add, a, 100.0)
                    names = [a]
            if names:
                matched.add(tok)

        for tok in tokens:
            if tok in matched or tok in aliased:
                continue
            # WRatio scores partial overlaps highly, so a short word matched any
            # longer name containing it: "karn" scored 90 on Kumbhakarna and
            # "bhim" on Abhimanyu. Short tokens are compared whole.
            scorer = fuzz.ratio if len(tok) <= SHORT else fuzz.WRatio
            hits = process.extract(tok, self.by_norm.keys(), scorer=scorer,
                                   score_cutoff=CUTOFF, limit=FUZZY_KEEP)
            if hits:
                for key, score, _ in hits:
                    add({"name": self.by_norm[key], "via": "fuzzy", "resolved": True,
                         "score": float(score), "surface": tok})
                continue
            hit = process.extractOne(tok, self.appellations.keys(), scorer=scorer,
                                     score_cutoff=CUTOFF)
            if hit:
                self._expand(add, self.appellations[hit[0]], float(hit[1]))
                continue
            # Last resort: a typo on top of a spelling variant, "Joydrath" for
            # Jayadratha. Compared key to key, so the variant part costs nothing.
            key = phonetic_key(tok)
            if key:
                for k, score, _ in process.extract(key, self.by_key.keys(), scorer=fuzz.ratio,
                                                   score_cutoff=PHONETIC_CUTOFF,
                                                   limit=FUZZY_KEEP):
                    for name in self.by_key[k][:FUZZY_KEEP]:
                        add({"name": name, "via": "phonetic", "resolved": True,
                             "score": float(score), "surface": tok, "key": k})

        # Now the other direction. Every name resolved above may also be
        # written under epithets, and the records filed under those epithets
        # are unreachable otherwise. These arrive resolved=False and every
        # fact they bring is marked with the epithet it came through, because
        # the apposition evidence is a lead and not an identity.
        if self.top_alias:
            # Plain-language mode reads the graph Phase 4f folded, where every
            # confidently linked epithet mention already sits under its person.
            # A leftover epithet node holds exactly the mentions that were not
            # confident, so pulling it in would attach the doubtful records:
            # "Kounteya" brings Bhima's divine father to Yudhishthira.
            return found
        for r in [x for x in found if x["resolved"]]:
            # Budget the two bases separately. Sorted together, the apposition
            # entries fill every slot and the prior-based ones never get in,
            # which is the whole reason Dhananjaya stayed invisible.
            pool = self.epithets_of.get(r["name"], [])
            chosen = ([e for e in pool if e["basis"] == "apposition"][:EPITHET_KEEP]
                      + [e for e in pool if e["basis"] == "prior"][:EPITHET_KEEP])
            for row in chosen:
                add({"name": row["appellation"], "via": "epithet", "resolved": False,
                     "score": r["score"], "surface": r["surface"],
                     "appellation": row["appellation"], "of": r["name"],
                     "confidence": row["confidence"], "basis": row["basis"],
                     "evidence_count": row["evidence_count"], "status": row["status"],
                     "candidates": int((self.app.appellation == row["appellation"]).sum())})
        return found

    def _expand(self, add, appellation, match_score):
        rows = self.app[self.app.appellation == appellation]
        if self.top_alias:
            add({"name": self.top_of[appellation], "via": "alias", "resolved": True,
                 "score": match_score, "surface": norm_surface(appellation),
                 "appellation": appellation, "also": self.also_of.get(appellation, [])})
            return
        for _, r in rows.sort_values("confidence", ascending=False, na_position="last").iterrows():
            # Eleven rows are prior_only: the corpus prior proposes the link
            # and no apposition sighting supports it, so there is no
            # confidence to report and NaN is what pandas puts there. A null
            # confidence must stay null rather than become a number.
            conf = None if pd.isna(r.confidence) else float(r.confidence)
            add({"name": r.candidate_entity, "via": "appellation", "resolved": False,
                 "score": match_score, "surface": norm_surface(appellation),
                 "appellation": appellation, "confidence": conf,
                 "evidence_count": int(r.evidence_count or 0), "status": r.status,
                 "candidates": len(rows)})

    def graph_facts(self, resolutions, limit=MAX_FACTS, query=""):
        """Every asserted relation touching a resolved name.

        No predicate filter. Filtering by a predicate guessed from the question
        is what makes a graph-backed bot answer "I don't know" to a fact it
        holds: ask who Karna's father is and a bornTo filter finds nothing,
        because the extraction recorded the edge in the other direction.
        """
        by_name = {}
        for r in resolutions:
            by_name.setdefault(r["name"], r)
        if not by_name:
            return []
        names = set(by_name)
        hit = self.rel[self.rel.subject.isin(names) | self.rel.object.isin(names)]
        support = (self.rel.groupby(["subject", "predicate", "object"])
                   .size().to_dict())
        # Best-supported first within each predicate. Everything in bornTo
        # carries confidence 0.85, so without this the order inside a group is
        # arbitrary and a trim drops rows by luck: (Dhananjaya, bornTo, Pandu)
        # at support 6 fell out of the budget while single-citation rows stayed.
        hit = hit.assign(_support=[support.get(k, 1) for k in
                                   zip(hit.subject, hit.predicate, hit.object)])
        hit = hit.sort_values(["cited", "_support", "confidence"], ascending=False)
        # Collapse repeats of the same triple before spending the budget. The
        # corpus states (Karna, slainBy, Arjuna) in two places and
        # (Karna, slainBy, Nakula) in one, and that difference is the whole
        # answer to "who killed Karna". Listed as separate rows it is invisible
        # and the two look like rival claims of equal weight.
        hit = hit.drop_duplicates(["subject", "predicate", "object", "parva", "adhyaya_ordinal"])
        facts = []
        unique = hit.drop_duplicates(["subject", "predicate", "object"])
        for _, row in self._round_robin(unique, limit, relevant(query)):
            touched = [n for n in (row.subject, row.object) if n in by_name]
            via = [by_name[n] for n in touched]
            facts.append({
                "subject": row.subject, "predicate": row.predicate, "object": row.object,
                "confidence": float(row.confidence), "pattern": row.pattern,
                # An uncited row has no chapter, and pandas spells that NaN.
                # NaN is not JSON, so the absence has to be a real null before
                # it reaches a caller that serializes. The null stays null.
                "parva": row.parva,
                "adhyaya": None if pd.isna(row.adhyaya_ordinal) else int(row.adhyaya_ordinal),
                "cited": bool(row.cited),
                "support": int(support.get((row.subject, row.predicate, row.object), 1)),
                "may_be": [],
                "citations": self._citations(hit, row),
                "evidence": row.evidence,
                "quoted": row.pattern not in UNQUOTED_PATTERNS,
                "unresolved_via": [v["appellation"] for v in via if not v["resolved"]],
                "parent_role": (self.parent_roles.get(row.object)
                                if row.predicate == "bornTo" else None),
            })
        return facts

    def siblings(self, resolutions, query=""):
        """People who share a recorded parent with a person the question named.

        The graph has no sibling relation, so "who was Partha's brother" finds
        nothing directly. Two people born to the same recorded parent are
        siblings; that is derived here and labelled as derived. Only built when
        the question asks about siblings, and only in plain-language mode.
        """
        words = norm_surface(query).split()
        if not self.top_alias or not any(w.startswith(h) for w in words for h in SIBLING_HINTS):
            return []
        people = {}
        for r in resolutions:
            if r["resolved"]:
                people.setdefault(r.get("of") or r["name"], set()).add(r["name"])
        born = self.rel[self.rel.predicate == "bornTo"]
        out = []
        for person, names in people.items():
            # A parent stated in two or more places, where there is one. A
            # single stray bornTo row is how an extraction error would make a
            # stranger into a sibling.
            counts = born[born.subject.isin(names)].object.value_counts()
            parents = set(counts[counts >= 2].index) or set(counts.index)
            # Cited rows first, so a pair stated both with and without a
            # chapter is cited: Karna to Kunti has one of each.
            rows = born[born.object.isin(parents) & ~born.subject.isin(names)].sort_values(
                "cited", ascending=False)
            stated = rows.subject.value_counts()
            seen = {}
            for _, row in rows.iterrows():
                sib = self.top_of.get(row.subject, row.subject)
                # A leftover epithet node ("Ajamidha", a descendant of Ajamidha)
                # is a title, not a person. A name in the appellation table on a
                # single parent row is taken as one; Bhimasena, also in that
                # table, has six rows and stays. Nakula has one and is no title.
                title = sib in self.appellations.values() and stated.get(row.subject, 0) < 2
                if sib in names or sib == person or title:
                    continue
                rec = seen.setdefault(sib, {"person": person, "sibling": sib,
                                            "parents": [], "citation": citation(
                                                row.parva, row.adhyaya_ordinal, row.cited)})
                if row.object not in rec["parents"]:
                    rec["parents"].append(row.object)
            out += sorted(seen.values(), key=lambda s: -len(s["parents"]))
        return out[:MAX_SIBLINGS]

    def role_of(self, parent, link):
        if link["kind"] == "divine":
            return "divine father"
        return self.parent_roles.get(parent, "parent")

    @staticmethod
    def source_of(link):
        return "text %s" % link["cite"] if link["source"] == "text" else "curated family table"

    def family(self, resolutions, query=""):
        """Kin two steps out from the people a family question names.

        Retrieval otherwise reaches only facts that touch a named person, so
        "Dhananjaya's maternal grandfather" stopped at Kunti and "Pandu's sons
        and their divine fathers" stopped at the sons. This walks one more step:
        parents and their parents, children and their other parents, spouses.
        """
        words = norm_surface(query).split()
        if not self.top_alias or not any(w.startswith(h) for w in words for h in KIN_HINTS):
            return {"lines": [], "path": []}
        people = []
        for r in resolutions:
            if r["resolved"] and r["via"] != "epithet":
                n = self.canon(r["name"])
                if n not in people:
                    people.append(n)
        up, down, spouse = self.kin

        def fmt(name, link, role=None):
            return "%s (%s%s)" % (name, role + "; " if role else "", self.source_of(link))

        lines = []
        for p in people[:3]:
            parents = up.get(p, {})
            if parents:
                lines.append("%s's parents: %s" % (p, ", ".join(
                    fmt(q, l, self.role_of(q, l)) for q, l in parents.items())))
            for q, l in parents.items():
                grand = up.get(q, {})
                if l["kind"] == "divine" or not grand:
                    continue
                side = {"mother": "maternal", "father": "paternal"}.get(self.parent_roles.get(q))
                label = "%s grandparents" % side if side else "grandparents through %s" % q
                lines.append("%s's %s, the parents of %s: %s" % (p, label, q, ", ".join(
                    fmt(g, gl, self.role_of(g, gl)) for g, gl in grand.items())))
            kids = list(down.get(p, {}).items())[:15]
            if kids:
                lines.append("%s's children: %s" % (p, ", ".join(
                    fmt(c, l, "as divine father" if l["kind"] == "divine" else None) for c, l in kids)))
                for c, _ in kids:
                    others = [(q, l) for q, l in up.get(c, {}).items() if q != p]
                    if others:
                        lines.append("  %s's other parents: %s" % (c, ", ".join(
                            fmt(q, l, self.role_of(q, l)) for q, l in others)))
            if spouse.get(p):
                lines.append("%s's spouses: %s" % (p, ", ".join(
                    fmt(s, l) for s, l in spouse[p].items())))
        path = self.kin_path(people[0], people[1]) if len(people) >= 2 else []
        return {"lines": lines[:MAX_FAMILY], "path": path}

    def kin_path(self, a, b, depth=5):
        """The shortest chain of parent, child and spouse links from a to b.

        Divine parentage is left out: Indra or Surya alone would connect half
        the epic, which is a true statement and no answer to "how are they related".
        """
        up, down, spouse = self.kin

        def steps(n):
            for q, l in up.get(n, {}).items():
                if l["kind"] != "divine":
                    yield q, "%s is %s's %s" % (q, n, self.role_of(q, l)), l
            for c, l in down.get(n, {}).items():
                if l["kind"] != "divine":
                    yield c, "%s is %s's child" % (c, n), l
            for s, l in spouse.get(n, {}).items():
                yield s, "%s is %s's spouse" % (s, n), l

        prev, frontier = {a: None}, [a]
        for _ in range(depth):
            nxt = []
            for n in frontier:
                for m, text, l in steps(n):
                    if m not in prev:
                        prev[m] = (n, "%s (%s)" % (text, self.source_of(l)))
                        nxt.append(m)
            if b in prev:
                break
            frontier = nxt
        if b not in prev:
            return []
        chain, n = [], b
        while prev[n]:
            n, text = prev[n]
            chain.append(text)
        return [("%s and %s" % (a, b), list(reversed(chain)))]

    @staticmethod
    def _citations(hit, row):
        """Every distinct chapter that states this triple, best first.

        How many separate places the corpus says a thing is the only weight
        this graph offers for choosing between rival statements.
        """
        same = hit[(hit.subject == row.subject) & (hit.predicate == row.predicate)
                   & (hit.object == row.object)]
        out = []
        for _, r in same.iterrows():
            c = citation(r.parva, r.adhyaya_ordinal, r.cited)
            if c not in out:
                out.append(c)
        return out[:6]

    def coreference_notes(self, facts):
        """Where two names in these facts are linked by apposition evidence.

        Asked who killed Karna, the graph answers Arjuna, Kiriti, Savyasachi
        and Vijaya, which reads as four killers and is really one name and
        three epithets. `build/appellations.csv` holds that evidence and no
        query path has ever shown it, so the answer looked like a contradiction
        the corpus does not actually contain.

        This reports the evidence. It does not merge anything, and the
        confidences are mostly low for good reason: Kounteya is any son of
        Kunti and Partha is used of Yudhishthira nearly as often as of Arjuna.
        A number near 1 is a strong hint; 0.25 is a hint that it is worth
        reading the passage.
        """
        present = {f[k] for f in facts for k in ("subject", "object")}
        notes = []
        for name in sorted(present):
            rows = self.app[self.app.appellation == name]
            for _, r in rows.iterrows():
                # prior_only rows carry no apposition sighting at all; the
                # corpus prior alone proposes them. That prior is the thing
                # CLAUDE.md calls a weak prior and never truth, and
                # build/canonical_id_audit.md traces the 295-row Bharadvaja to
                # Drona collapse to it. Bharadvaja to Drona is in this very
                # list. Reporting it here as evidence that two names may be
                # one person would launder the known failure into an answer.
                if r.status == "prior_only" or pd.isna(r.confidence):
                    continue
                if r.candidate_entity in present and r.candidate_entity != name:
                    notes.append({"appellation": name, "candidate": r.candidate_entity,
                                  "confidence": float(r.confidence),
                                  "evidence_count": int(r.evidence_count),
                                  "candidates": len(rows)})
        return sorted(notes, key=lambda n: -n["confidence"])

    @staticmethod
    def _round_robin(hit, limit, first=()):
        """Spend the budget across predicates, best first within each.

        Taking the top rows outright does not work. speaksTo is 1,247 of the
        2,800 assertions and carries the highest confidence in the set, so it
        fills the block on its own: ask who killed Karna and every line that
        comes back is somebody addressing him, while (Karna, slainBy, Arjuna)
        sits below the cut. Filtering by a predicate guessed from the question
        would break the opposite way. Giving every predicate a turn costs
        nothing and answers both shapes of question.
        """
        groups = {p: list(g.iterrows()) for p, g in hit.groupby("predicate", sort=False)}

        def rotate(keys, budget, out):
            keys = sorted(keys, key=lambda k: -len(groups[k]))
            while budget > 0 and any(groups[k] for k in keys):
                for k in keys:
                    if groups[k]:
                        out.append(groups[k].pop(0))
                        budget -= 1
                        if budget <= 0:
                            break
            return out

        out = []
        # Hinted predicates take the first half of the budget among themselves.
        # Ordering alone was not enough: round-robin interleaves, so "who was
        # the father of Arjuna" got two parentage rows and then thirty rows of
        # who spoke to whom. Giving the hinted group a share puts the answer
        # where the model reads it. The second half rotates over everything,
        # hinted predicates included, so nothing is filtered out.
        hinted = [k for k in groups if k in first]
        if hinted:
            rotate(hinted, limit // 2, out)
        rotate(list(groups), limit - len(out), out)
        return out[:limit]

    def passages(self, query, limit=MAX_PASSAGES):
        if self.index is None:
            return {"passages": [], "status": "unavailable"}
        return self.index.search(query, limit=limit)


def citation(parva, adhyaya, cited):
    if not cited or adhyaya is None or pd.isna(adhyaya):
        return "uncited"
    return "%s %d" % (parva, int(adhyaya))


def relevant(query):
    """Predicates whose hint words appear in the question."""
    words = set(norm_surface(query).split())
    return {p for p, hints in PREDICATE_HINTS.items()
            if any(h in words or any(w.startswith(h) for w in words) for h in hints)}


def build_context(resolutions, facts, retrieval, notes=(), siblings=(), family=None):
    """The context block, and the only thing besides the question that is sent.

    Every fact line ends in a citation or the word uncited, so the model cannot
    quote one without quoting its provenance too.
    """
    out = []
    unresolved = [r for r in resolutions if not r["resolved"]]
    if unresolved:
        out.append("UNRESOLVED NAMES AND EPITHETS. Candidate referents, not identities. "
                   "The corpus does not settle these and the graph merges none of them. "
                   "Where the question named a person, records filed under their epithets "
                   "are retrieved too, because the corpus files a great deal under an "
                   "epithet rather than the common name: Arjuna's parentage is stated far "
                   "more often of Dhananjaya than of Arjuna. Use those records, and say "
                   "which name each was recorded under.")
        for r in unresolved:
            if r.get("basis") == "prior":
                out.append('  "%s" is recorded elsewhere in the corpus as an epithet of %s. '
                           'That link comes from the corpus name prior, which has no '
                           'confidence attached and is weak evidence. Records filed under '
                           '"%s" are shown so the answer can reach them; report them as '
                           'recorded under that name, not as settled fact about %s.'
                           % (r["appellation"], r["of"], r["appellation"], r["of"]))
            elif r.get("basis") == "apposition":
                out.append('  "%s" may be an epithet of %s, confidence %.2f from %d '
                           'apposition sightings. Records filed under "%s" are shown so the '
                           'answer can reach them, and remain records about that name.'
                           % (r["appellation"], r["of"], r["confidence"],
                              r["evidence_count"], r["appellation"]))
            elif r["confidence"] is None:
                # No apposition sighting supports this at all. Say that, rather
                # than leaving a blank where a number belongs.
                out.append('  "%s" -> candidate %s, NO apposition evidence, proposed only by '
                           'the corpus prior, which this project treats as weak and often '
                           'wrong. Treat as a lead, not as evidence. 1 of %d candidates'
                           % (r["appellation"], r["name"], r["candidates"]))
            else:
                out.append('  "%s" -> candidate %s, confidence %.2f from %d apposition sightings, '
                           'status %s, 1 of %d candidates'
                           % (r["appellation"], r["name"], r["confidence"],
                              r["evidence_count"], r["status"], r["candidates"]))
        out.append("")

    resolved = [r for r in resolutions if r["resolved"]]
    if resolved:
        out.append("NAMES." if any(r["via"] in ("alias", "epithet") for r in resolved)
                   else "RESOLVED NAMES.")
        variants = {}
        for r in resolved:
            variants.setdefault(r["surface"], []).append(r)
        for surface, group in variants.items():
            for r in group:
                if r["via"] == "alias":
                    line = '  "%s" in the question means %s.' % (r["appellation"], r["name"])
                    if r.get("cites"):
                        line += " The text calls %s by this name in %s." % (
                            r["name"], ", ".join(r["cites"]))
                    if r.get("also"):
                        line += (' The name is also used, almost as often, for %s. The first '
                                 'time you use it, write exactly: "%s (here %s; the name is '
                                 'also used for %s)".' % (
                                     " and ".join(r["also"]), r["appellation"], r["name"],
                                     " and ".join(r["also"])))
                    out.append(line)
                    continue
                if r["via"] == "epithet":
                    out.append("  %s is another name for %s." % (r["name"], r["of"]))
                    continue
                note = ("exact match" if r["via"] == "exact"
                        else "sound match on %r, score %.0f" % (surface, r["score"])
                        if r["via"] == "phonetic"
                        else "spelling match on %r, score %.0f" % (surface, r["score"]))
                out.append("  %s (%s)" % (r["name"], note))
            if len([r for r in group if r["via"] in ("exact", "fuzzy", "phonetic")]) > 1:
                out.append("    %r matched %d spellings. The graph holds them as "
                           "separate entities and does not state they are the same "
                           "person. Report what each one's records say; do not merge them."
                           % (surface, len(group)))
        out.append("")

    if notes:
        out.append("NAMES IN THESE FACTS THAT MAY BE THE SAME PERSON. Apposition evidence "
                   "from the corpus, not a merge. Several names below are epithets, so a "
                   "list of rival answers may be one answer under several names. Report "
                   "this as evidence with its confidence, never as settled identity.")
        for n in notes[:MAX_NOTES]:
            out.append("  %s may be %s, confidence %.2f from %d sightings, 1 of %d candidates"
                       % (n["appellation"], n["candidate"], n["confidence"],
                          n["evidence_count"], n["candidates"]))
        out.append("")

    if family and family["lines"]:
        out.append("FAMILY, up to two steps from the people in the question. Names are "
                   "already merged to one person each. A 'text' link cites a passage; a "
                   "'curated family table' link comes from the project's hand-made family "
                   "table and cites no passage.")
        out += ["  " + line for line in family["lines"]]
        out.append("")
    if family and family["path"]:
        for pair, chain in family["path"]:
            out.append("KINSHIP PATH between %s, shortest chain of family links:" % pair)
            out += ["  %d. %s" % (i + 1, step) for i, step in enumerate(chain)]
        out.append("")

    if siblings:
        out.append("SIBLINGS, worked out from a shared recorded parent. The graph has no "
                   "sibling relation of its own.")
        for s in siblings:
            out.append("  %s and %s share the parent %s, %s" % (
                s["person"], s["sibling"], " and ".join(s["parents"]), s["citation"]))
        out.append("")

    out.append("ASSERTED FACTS from the extracted graph. `support` is how many separate "
               "places in the corpus state that triple, and is the main weight for "
               "choosing between rival statements.")
    if not facts:
        out.append("  none. The graph holds no relation for these names.")
    for i, f in enumerate(facts):
        line = "  (%s, %s, %s) support %d, confidence %.2f, pattern %s, %s" % (
            f["subject"], f["predicate"], f["object"], f["support"], f["confidence"],
            f["pattern"], ", ".join(f["citations"][:3]))
        if f["unresolved_via"]:
            line += " [reached only through the unresolved name %s]" % ", ".join(f["unresolved_via"])
        if f.get("parent_role"):
            line += " [%s is the %s]" % (f["object"], f["parent_role"])
        # The hint goes on the line, not only in the block above it. A model
        # reading "slainBy Kiriti" will not go back and cross-reference a
        # section it has already passed, and that is exactly the row where
        # missing it turns one killer into five.
        same = [n for n in notes if n["appellation"] in (f["subject"], f["object"])]
        for n in same[:2]:
            line += " [%s may be %s, confidence %.2f]" % (
                n["appellation"], n["candidate"], n["confidence"])
        out.append(line)
        if f["evidence"] and f["quoted"] and i < QUOTED_FACTS:
            out.append('      evidence: "%s"' % f["evidence"])
    out.append("")

    passages = retrieval.get("passages", [])
    out.append("SOURCE PASSAGES retrieved by similarity. Relevance is not evidence of a fact.")
    if retrieval.get("status") == "unavailable":
        out.append("  the passage index is not loaded.")
    elif retrieval.get("status") == "weak_match":
        out.append("  weak match. Nothing in the corpus scored well against this question.")
    for p in passages:
        out.append("  [%s] %s" % (citation(p["parva"], p.get("chapter"), p["cited"]), p["text"]))
    return "\n".join(out)


def retry_after(detail, default=6.0):
    match = re.search(r"try again in ([\d.]+)s", detail)
    return min(float(match.group(1)) + 0.5, 30.0) if match else default


def call_groq(messages, key, model=MODEL, endpoint=ENDPOINT, timeout=90, retries=2):
    body = json.dumps({"model": model, "messages": messages}).encode()
    req = urllib.request.Request(endpoint, data=body, method="POST", headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json",
        "User-Agent": AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:400]
        if e.code == 403 and "1010" in detail:
            detail = "blocked by Cloudflare before reaching the API, not an auth failure"
        # The free tier allows 8,000 tokens a minute and a grounded context
        # block is a few thousand of them, so two questions in quick
        # succession hit this normally rather than exceptionally.
        if e.code == 429 and retries > 0:
            time.sleep(retry_after(detail))
            return call_groq(messages, key, model, endpoint, timeout, retries - 1)
        raise RuntimeError("Groq returned %d: %s" % (e.code, detail))
    except urllib.error.URLError as e:
        raise RuntimeError("Could not reach Groq: %s" % e.reason)
    except TimeoutError:
        # A slow reply raises TimeoutError from the read, not URLError, and it
        # reached the service as a bare 500 "Local request failed". Said plainly
        # instead. Not retried: a second wait would hold the page for minutes.
        raise RuntimeError("Groq took longer than %d s to answer. Try asking again in a "
                           "moment." % timeout)
    message = payload["choices"][0]["message"]
    # Reasoning models put the answer in content and their scratch work in a
    # separate field. Only the answer is shown; the scratch work is not an
    # answer and is not evidence.
    return (message.get("content") or "").strip() or "The model returned no answer text."


def retrieve(retriever, query):
    """Everything the model will be shown, as data rather than as a block.

    The CLI wants the text; the explorer panel wants the parts, so it can draw
    a citation next to each fact and mark the unresolved ones itself instead of
    trusting prose to have said so.
    """
    resolutions = retriever.resolve(query)
    facts = retriever.graph_facts(resolutions, query=query)
    # Plain-language mode has already chosen one person per name, so the
    # may-be-the-same-person evidence would only reintroduce the hedging.
    notes = [] if retriever.top_alias else retriever.coreference_notes(facts)
    for f in facts:
        f["may_be"] = [n for n in notes if n["appellation"] in (f["subject"], f["object"])][:2]
    siblings = retriever.siblings(resolutions, query)
    family = retriever.family(resolutions, query) if retriever.top_alias else {"lines": [], "path": []}
    found = retriever.passages(query)
    return {
        "query": query,
        "resolutions": resolutions,
        "facts": facts,
        "coreference": notes,
        "siblings": siblings,
        "family": family,
        "passages": found.get("passages", []),
        "retrieval_status": found.get("status"),
        "notice": found.get("notice", ""),
        "context": build_context(resolutions, facts, found, notes, siblings, family),
    }


def name_note(resolutions):
    """One plain line for a name that is close to a tie, put above the answer.

    The model drops this caveat about half the time when only asked to, so it
    is added here, where it cannot be dropped. Only near ties get it.
    """
    lines = ['"%s" is taken here to mean %s; the name is also used for %s.'
             % (r["appellation"], r["name"], " and ".join(r["also"]))
             for r in resolutions if r.get("via") == "alias" and r.get("also")]
    return " ".join(lines)


def with_note(answer_text, resolutions):
    note = name_note(resolutions)
    return "%s\n\n%s" % (note, answer_text) if note and answer_text else answer_text


def generate(query, context, history=(), key=None, model=MODEL):
    system = SYSTEM_SIMPLE if SIMPLE else SYSTEM
    messages = [{"role": "system", "content": system.format(context=context)}]
    messages += list(history)[-HISTORY_TURNS:]
    messages.append({"role": "user", "content": query})
    return call_groq(messages, key, model)


def answer(retriever, query, history=(), key=None, model=MODEL):
    """Retrieve, build the context, and either call the model or return the context."""
    found = retrieve(retriever, query)
    context = found["context"]
    if not key:
        return context, context
    return with_note(generate(query, context, history, key, model), found["resolutions"]), context


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--once", help="answer one question and exit")
    ap.add_argument("--no-llm", action="store_true",
                    help="print the retrieved context instead of calling the model")
    ap.add_argument("--no-passages", action="store_true",
                    help="graph facts only; skips loading the sentence encoder")
    ap.add_argument("--model", default=MODEL,
                    help="Groq model id; see https://api.groq.com/openai/v1/models")
    args = ap.parse_args(argv)

    key = None if args.no_llm else os.environ.get("GROQ_API_KEY")
    if not key and not args.no_llm:
        print("GROQ_API_KEY is not set. Printing retrieved context instead of an answer.\n",
              file=sys.stderr)
    r = Retriever(passages=not args.no_passages, top_alias=SIMPLE)

    if args.once:
        try:
            print(answer(r, args.once, key=key, model=args.model)[0])
        except (ValueError, RuntimeError) as e:
            print(e, file=sys.stderr)
            return 1
        return 0

    history = []
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q in {"exit", "quit"}:
            return 0
        try:
            reply, _ = answer(r, q, history, key, args.model)
        except (ValueError, RuntimeError) as e:
            print("  %s" % e)
            continue
        print("\n" + reply)
        if key:
            history += [{"role": "user", "content": q},
                        {"role": "assistant", "content": reply}]


if __name__ == "__main__":
    sys.exit(main())
