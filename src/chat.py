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

    def __init__(self, root=ROOT, passages=True, index=None):
        root = Path(root)
        reviewed = root / "build/reviewed_relations.parquet"
        self.reviewed = reviewed.exists()
        self.rel = pd.read_parquet(reviewed if self.reviewed
                                   else root / "build/relations.parquet")
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
            if len(t) <= 2:
                continue
            tokens.append(t)
            # norm_surface drops "Arjuna's" but not "arjunas", and a dropped
            # apostrophe is the commonest thing in a typed question. The bare
            # stem is worth trying because "arjunas" scores 83 against Arjuna,
            # just under the cutoff, while "arjuna" is exact.
            if t.endswith("s") and len(t) > 4:
                tokens.append(t[:-1])
        found, seen = [], set()

        def add(rec):
            key = (rec["name"], rec["via"])
            if key not in seen:
                seen.add(key)
                found.append(rec)

        for tok in tokens:
            if tok in self.by_norm:
                add({"name": self.by_norm[tok], "via": "exact", "resolved": True,
                     "score": 100.0, "surface": tok})
            if tok in self.appellations:
                self._expand(add, self.appellations[tok], 100.0)

        matched = {r["surface"] for r in found}
        for tok in tokens:
            if tok in matched:
                continue
            hits = process.extract(tok, self.by_norm.keys(), scorer=fuzz.WRatio,
                                   score_cutoff=CUTOFF, limit=FUZZY_KEEP)
            if hits:
                for key, score, _ in hits:
                    add({"name": self.by_norm[key], "via": "fuzzy", "resolved": True,
                         "score": float(score), "surface": tok})
                continue
            hit = process.extractOne(tok, self.appellations.keys(), scorer=fuzz.WRatio,
                                     score_cutoff=CUTOFF)
            if hit:
                self._expand(add, self.appellations[hit[0]], float(hit[1]))

        # Now the other direction. Every name resolved above may also be
        # written under epithets, and the records filed under those epithets
        # are unreachable otherwise. These arrive resolved=False and every
        # fact they bring is marked with the epithet it came through, because
        # the apposition evidence is a lead and not an identity.
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
            })
        return facts

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


def build_context(resolutions, facts, retrieval, notes=()):
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
        out.append("RESOLVED NAMES.")
        variants = {}
        for r in resolved:
            variants.setdefault(r["surface"], []).append(r)
        for surface, group in variants.items():
            for r in group:
                note = ("exact match" if r["via"] == "exact"
                        else "spelling match on %r, score %.0f" % (surface, r["score"]))
                out.append("  %s (%s)" % (r["name"], note))
            if len(group) > 1:
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


def call_groq(messages, key, model=MODEL, endpoint=ENDPOINT, timeout=60, retries=2):
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
    notes = retriever.coreference_notes(facts)
    for f in facts:
        f["may_be"] = [n for n in notes if n["appellation"] in (f["subject"], f["object"])][:2]
    found = retriever.passages(query)
    return {
        "query": query,
        "resolutions": resolutions,
        "facts": facts,
        "coreference": notes,
        "passages": found.get("passages", []),
        "retrieval_status": found.get("status"),
        "notice": found.get("notice", ""),
        "context": build_context(resolutions, facts, found, notes),
    }


def generate(query, context, history=(), key=None, model=MODEL):
    messages = [{"role": "system", "content": SYSTEM.format(context=context)}]
    messages += list(history)[-HISTORY_TURNS:]
    messages.append({"role": "user", "content": query})
    return call_groq(messages, key, model)


def answer(retriever, query, history=(), key=None, model=MODEL):
    """Retrieve, build the context, and either call the model or return the context."""
    found = retrieve(retriever, query)
    context = found["context"]
    if not key:
        return context, context
    return generate(query, context, history, key, model), context


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
    r = Retriever(passages=not args.no_passages)

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
