"""Phase 2b. Confidence-aware entity resolution.

Implements the method in "Entity Normalization in a Mahabharata Knowledge Graph"
(the confidence-score design document) on top of the Phase 1 mention layer and
the Phase 2 appellation table. Every PERSON mention gets a candidate set, five
features per candidate, and a confidence:

    confidence = 0.30 A + 0.25 C + 0.20 R + 0.15 S + 0.10 E

  A  alias match. 1.0 if a dictionary links the surface to the candidate, 0.7 for
     an approximate spelling only, as the document specifies.
  C  context. Cosine between the mention's sentence and the candidate's profile
     (sentences where the candidate is named unambiguously), with the mention
     masked in both so the name cannot vote for itself. Raw cosines from a
     sentence encoder sit in a narrow band, so C is the cosine's percentile among
     the self-similarities of unambiguous mentions: 0.5 is a typical correct match.
  R  relationship consistency. Of the unambiguous names in the surrounding
     sentences, the share that are graph neighbours of the candidate
     (co-mention graph plus out/relations.ttl). 0.5 when no such name exists,
     because no evidence is not evidence against.
  S  source. The most reliable source proposing the link, on the document's scale:
     1.0 the text itself (identity, or Phase 2 asserted merge), 0.8 the seed
     alias list, 0.6 automatic extraction (apposition-only, corpus prior, spelling).
  E  entity type. 1 if the candidate is mostly labelled PERSON, else 0.

Bands follow the document: >= 0.90 auto-accept, 0.75 to 0.89 accept with sample
verification, 0.50 to 0.74 human review, below 0.50 reject. One addition: where
the runner-up is within MARGIN of the winner, the mention goes to human review
whatever its score. "Kounteya" can be any of four sons of Kunti, and a close
second is exactly the case the hard rule against guessing a link is about.

Nothing here writes to out/. Every link is a proposal in build/er/, and a merge
into the graph would need the decision recorded in docs/DECISIONS.md first.

Gold standard: if annotations/er_gold.csv exists (mention_id, correct_entity),
the script also reports precision, recall and F1 per band, calibration (ECE and a
reliability table), and learns the weights with logistic regression, grouped by
surface so no alias is in both train and test.

Run: python src/phase2b_confidence.py [--encoder tfidf|MODEL_NAME]
"""
import argparse, collections, json, os, re, sys, unicodedata

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_entities import TITLES, norm_surface  # noqa: E402
from vocab import NOT_A_NAME  # noqa: E402

W = {"A": 0.30, "C": 0.25, "R": 0.20, "S": 0.15, "E": 0.10}
BANDS = [(0.90, "very_high", "auto_accept"), (0.75, "high", "accept_sample_verify"),
         (0.50, "medium", "human_review"), (0.0, "low", "reject")]
MARGIN = 0.10         # runner-up this close sends the mention to review
FUZZY_MIN = 90        # rapidfuzz ratio for a spelling variant
FUZZY_DOMINANCE = 3   # the variant must be this many times more frequent
PROFILE_K = 300       # sentences per candidate profile
MIN_COUNT = 3         # a surface seen fewer times is not a canonical entity
WINDOW = 2            # sentences either side for R
SEED = 20260930
OUT = "build/er"
GOLD = "annotations/er_gold.csv"

# Seed alias list, from general knowledge of the epic, in this translation's
# spelling. It is a secondary source (S = 0.8) and needs a human pass: it says
# which bearers an epithet CAN have, never which one a passage means. "self"
# keeps the surface as its own candidate, for names another person also bears
# (Bhima the king of Vidarbha, Vasudeva the father of Krishna).
SEED_ALIASES = {
    "Partha": ["Arjuna", "Yudhishthira", "Bhimasena"],
    "Parthas": ["Arjuna", "Yudhishthira", "Bhimasena"],
    "Kounteya": ["Arjuna", "Yudhishthira", "Bhimasena", "Karna"],
    "Kunti son": ["Arjuna", "Yudhishthira", "Bhimasena", "Karna"],
    "Dhananjaya": ["Arjuna"], "Phalguna": ["Arjuna"], "Savyasachi": ["Arjuna"],
    "Bibhatsu": ["Arjuna"], "Kiriti": ["Arjuna"], "Jishnu": ["Arjuna"],
    "Gudakesha": ["Arjuna"], "Vijaya": ["Arjuna", "self"], "Nara": ["Arjuna", "self"],
    "Dharmaraja": ["Yudhishthira", "Yama"], "Ajatashatru": ["Yudhishthira"],
    "Kouravya": ["Yudhishthira", "Dhritarashtra", "Duryodhana", "Bhishma"],
    "Bhima": ["Bhimasena", "self"], "Vrikodara": ["Bhimasena"],
    "Vasudeva": ["Krishna", "self"], "Keshava": ["Krishna"], "Janardana": ["Krishna"],
    "Madhava": ["Krishna", "self"], "Govinda": ["Krishna"], "Madhusudana": ["Krishna"],
    "Hrishikesha": ["Krishna"], "Achyuta": ["Krishna", "Vishnu"],
    "Varshneya": ["Krishna", "Satyaki", "self"], "Dasharha": ["Krishna", "self"],
    "Pundarikaksha": ["Krishna", "Vishnu"], "Shouri": ["Krishna"],
    "Hari": ["Krishna", "Vishnu"], "Narayana": ["Vishnu", "Krishna"],
    "Purushottama": ["Krishna", "Vishnu"],
    "Krishna": ["self", "Droupadi", "Vyasa"],
    "Krishna Dvaipayana": ["Vyasa"], "Dvaipayana": ["Vyasa"],
    "Radheya": ["Karna"], "Vaikartana": ["Karna"], "Suyodhana": ["Duryodhana"],
    "Bharadvaja": ["Drona", "self"], "Bharadvaja son": ["Drona"],
    "Drona son": ["Ashvatthama"], "Drouni": ["Ashvatthama"],
    "Gangeya": ["Bhishma"], "Devavrata": ["Bhishma"], "Shantanu son": ["Bhishma"],
    "Yuyudhana": ["Satyaki"], "Shini descendant": ["Satyaki"], "Shaineya": ["Satyaki"],
    "Panchali": ["Droupadi"], "Yajnaseni": ["Droupadi"], "Draupadi": ["Droupadi"],
    "Shakra": ["Indra"], "Vasava": ["Indra"], "Shatakratu": ["Indra"],
    "Purandara": ["Indra"], "Maghavan": ["Indra"],
    "Parshata": ["Dhrishtadyumna", "Drupada"], "Saindhava": ["Jayadratha"],
    "Soubala": ["Shakuni"], "Subhadra son": ["Abhimanyu"],
    "Goutama": ["Kripa", "self"], "Hardikya": ["Kritavarma"], "Kshatta": ["Vidura"],
    "Pritha": ["Kunti"], "Mahadeva": ["Shiva"], "Maheshvara": ["Shiva"],
    "Shankara": ["Shiva"], "Sharva": ["Shiva"], "Hara": ["Shiva"], "Ishana": ["Shiva"],
    "Sthanu": ["Shiva"], "Bhava": ["Shiva"], "Rudra": ["Shiva", "self"],
    "Ushanas": ["Shukra"], "Vaishravana": ["Kubera"], "Svayambhu": ["Brahma"],
    "Prajapati": ["Brahma", "self"], "Kumara": ["Skanda", "self"],
    "Bhargava": ["Parashurama", "Shukra", "self"], "Rama": ["self", "Parashurama", "Balarama"],
    "Souti": ["Ugrashrava"],
}
# Where a "self" reading of a seed alias can apply, as (parva, first adhyaya,
# last adhyaya). Outside these chapters the name is only its seed candidates.
# King Bhima of Vidarbha is the Nala tale; elsewhere "Bhima" is the Pandava, and
# the bare name, which can never have a context profile, won C = 0.5 by default
# on 1,007 of 1,517 mentions. Same standing as the seed list: needs a human pass.
SELF_SCOPE = {"Bhima": [("03_Aranyaka", 50, 78)]}
S_SOURCE = {"identity": 1.0, "phase2_asserted": 1.0, "seed": 0.8,
            "phase2_apposition": 0.6, "prior": 0.6, "spelling": 0.6}


def fold(s):
    """Orthographic key: the Phase 2 normalization plus diacritics, so the
    document's Partha with a macron and the translation's Partha meet."""
    s = unicodedata.normalize("NFKD", str(s))
    return norm_surface("".join(c for c in s if not unicodedata.combining(c)))


def clean_surface(s):
    toks = [t for t in str(s).split() if t not in TITLES]
    return " ".join(toks)


def band(c):
    for lo, name, action in BANDS:
        if c >= lo:
            return name, action


# ---------------------------------------------------------------- candidates

def load_mentions():
    df = pd.read_parquet("build/mentions.parquet")
    df = df[df.label == "PERSON"].copy()
    df["clean"] = df.surface.map(clean_surface)
    df = df[df.clean.str.fullmatch(r"[A-Z][a-z]{2,}( [A-Za-z][a-z]+)*")
            & ~df.clean.isin(NOT_A_NAME)]
    df["key"] = df.clean.map(fold)
    return df.reset_index(drop=True)


def all_labels():
    """Majority label per folded surface over every mention, not only PERSON."""
    df = pd.read_parquet("build/mentions.parquet", columns=["surface", "label"])
    df["key"] = df.surface.map(clean_surface).map(fold)
    return df.groupby("key").label.agg(lambda x: x.value_counts().idxmax()).to_dict()


def build_dictionary(men):
    """(folded alias -> {folded candidate: [sources]}, display names, canonical set)."""
    counts = men.key.value_counts()
    display = men.groupby("key").clean.agg(lambda x: x.value_counts().idxmax()).to_dict()
    links = collections.defaultdict(lambda: collections.defaultdict(set))
    no_self = set()

    for alias, cands in SEED_ALIASES.items():
        a = fold(alias)
        for c in cands:
            links[a][a if c == "self" else fold(c)].add("seed")
        if "self" not in cands:
            no_self.add(a)

    # Spelling variants and drop-cap truncations ("udhishthira", "hishma"). The
    # minority form stops being an entity of its own and gets A = 0.7.
    from rapidfuzz import fuzz, process
    frequent = [k for k, n in counts.items() if n >= MIN_COUNT and " " not in k]
    for k in frequent:
        if len(k) < 5:
            continue
        for t, score, _ in process.extract(k, frequent, scorer=fuzz.ratio,
                                           score_cutoff=FUZZY_MIN, limit=4):
            if t == k or t == k + "s" or k == t + "s":
                continue
            if counts[t] >= FUZZY_DOMINANCE * counts[k]:
                links[k][t].add("spelling")
                no_self.add(k)

    # Phase 2. An asserted merge (apposition and corpus prior agree) and a prior
    # rewrite each propose a candidate. Apposition evidence alone only supports a
    # candidate something else proposed: adjacency also catches lists of names,
    # and Phase 2's direction rule reads "Partha Arjuna" as Arjuna being an
    # epithet of Partha, which made every plain "Arjuna" a tie.
    if os.path.exists("build/phase2_resolution.json"):
        res = json.load(open("build/phase2_resolution.json", encoding="utf-8"))
        for dep, r in res.items():
            d = fold(dep)
            if r.get("prior"):
                links[d][fold(r["prior"])].add("prior")
            for c in r["candidates"]:
                if not c["entity"]:
                    continue
                e = fold(c["entity"])
                if r["status"] == "asserted":
                    links[d][e].add("phase2_asserted")
                elif r["status"] == "prior_only":
                    links[d][e].add("prior")
                elif e in links.get(d, {}):
                    links[d][e].add("phase2_apposition")
            if r["status"] == "asserted":
                no_self.add(d)

    # A candidate the seed list itself resolves to the alias, or to another of
    # its candidates, is that person under a second name, not a rival. The corpus
    # prior calls the Pandava "Bhima" while the seed makes Bhima an epithet of
    # Bhimasena, so each proposed the other: neither was ever unambiguous, neither
    # got a context profile, every mention of either tied at C = 0.5, and the free
    # 0.5 then outscored Arjuna's real C on "Partha". The link folds into the
    # candidate it names. A seed link is never folded: the seed is the specific word.
    seed_to = {fold(a): {fold(c) for c in cs if c != "self"} for a, cs in SEED_ALIASES.items()}
    for a in list(links):
        for c in list(links[a]):
            if c == a or "seed" in links[a][c]:
                continue
            same = seed_to.get(c, set()) & (set(links[a]) | {a})
            if same:
                for d in same - {a}:
                    links[a][d] |= links[a][c]
                del links[a][c]

    canonical ={k for k, n in counts.items() if n >= MIN_COUNT and k not in no_self}
    for a in list(links):
        for c in list(links[a]):
            if c != a and c not in counts:          # candidate never named in this text
                del links[a][c]
    return links, display, canonical


def candidates_for(key, links, canonical):
    cands = {}
    if key in canonical:
        # A name the seed list says someone else also bears ("Bhima") is not
        # identity evidence for the unmerged bearer. It enters at the seed's level.
        cands[key] = {"seed"} if "seed" in links.get(key, {}).get(key, ()) else {"identity"}
    for c, src in links.get(key, {}).items():
        if c == key and key not in canonical:
            continue
        cands.setdefault(c, set()).update(src)
    # A compound whose every token points at one candidate names it outright:
    # "Dharmaraja Yudhishthira". "Drona son" does not, and its tokens disagree.
    toks = key.split()
    if not cands and len(toks) > 1:
        sets = [set(candidates_for(t, links, canonical)) for t in toks]
        common = set.intersection(*sets) if all(sets) else set()
        for c in common:
            cands[c] = {"phase2_apposition"}
    return cands


# ---------------------------------------------------------------- features

def mask(sentence, surfaces):
    for s in sorted(surfaces, key=len, reverse=True):
        sentence = re.sub(r"\b%s\b" % re.escape(s), "someone", sentence)
    return sentence


class Encoder:
    def __init__(self, name):
        self.name = name
        if name == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = None
            self.TfidfVectorizer = TfidfVectorizer
        else:
            from sentence_transformers import SentenceTransformer
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            print("encoder %s on %s" % (name, dev))
            self.model = SentenceTransformer(name, device=dev)

    def fit(self, texts):
        if self.name == "tfidf":
            self.vec = self.TfidfVectorizer(min_df=2, sublinear_tf=True,
                                            stop_words="english").fit(texts)

    def __call__(self, texts):
        if self.name == "tfidf":
            from sklearn.preprocessing import normalize
            return normalize(self.vec.transform(texts)).astype(np.float32)
        return self.model.encode(list(texts), batch_size=256, normalize_embeddings=True,
                                 show_progress_bar=True, convert_to_numpy=True)


def dense(x):
    return x.toarray() if hasattr(x, "toarray") else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    men = load_mentions()
    labels = all_labels()
    links, display, canonical = build_dictionary(men)
    men["cands"] = [candidates_for(k, links, canonical) for k in men.key]
    scope = {fold(a): spans for a, spans in SELF_SCOPE.items()}

    def in_scope(r):
        spans = scope.get(r.key)
        if not spans or r.key not in r.cands or len(r.cands) == 1:
            return r.cands
        if any(r.parva == p and lo <= r.adhyaya_ordinal <= hi for p, lo, hi in spans):
            return r.cands
        return {c: s for c, s in r.cands.items() if c != r.key}
    men["cands"] = [in_scope(r) for r in men.itertuples()]
    men = men[men.cands.map(len) > 0].reset_index(drop=True)
    men["n_cands"] = men.cands.map(len)
    print("PERSON mentions with candidates: %d, canonical entities: %d, aliases: %d"
          % (len(men), len(canonical), sum(1 for k in links if k not in canonical)))

    # Unambiguous mentions: the name is its own and only candidate.
    unamb = men[(men.n_cands == 1) & men.apply(lambda r: r.key in r.cands, axis=1)]

    # --- C: profiles and mention contexts
    enc = Encoder(args.encoder)
    prof_rows = []
    for k, g in unamb.groupby("key"):
        take = g.sample(min(len(g), PROFILE_K), random_state=SEED)
        prof_rows += [(k, mask(s, [c])) for s, c in zip(take.sentence, take.clean)]
    prof = pd.DataFrame(prof_rows, columns=["key", "text"])
    men["ctx"] = [mask(s, [c]) for s, c in zip(men.sentence, men.clean)]
    enc.fit(pd.concat([prof.text, men.ctx]).tolist())
    pv = dense(enc(prof.text.tolist()))
    centroid = {}
    for k, idx in prof.groupby("key").indices.items():
        v = pv[idx].mean(axis=0)
        centroid[k] = v / (np.linalg.norm(v) or 1.0)
    mv = dense(enc(men.ctx.tolist()))

    def cos(i, k):
        return float(mv[i] @ centroid[k]) if k in centroid else np.nan

    # Calibrate C against the self-similarity of unambiguous mentions.
    self_sims = np.array(sorted(cos(i, k) for i, k in zip(unamb.index, unamb.key)
                                if k in centroid))
    to_c = lambda x: 0.5 if np.isnan(x) else float(np.searchsorted(self_sims, x) / len(self_sims))

    # --- R: neighbour graph from unambiguous co-mentions and the asserted graph
    sent_names = unamb.groupby(["parva", "sent_id"]).key.agg(set)
    co = collections.Counter()
    for ns in sent_names:
        for a in ns:
            for b in ns:
                if a < b:
                    co[(a, b)] += 1
    nbr = collections.defaultdict(set)
    for (a, b), n in co.items():
        if n >= 2:
            nbr[a].add(b)
            nbr[b].add(a)
    triples = read_relations("out/relations.ttl")
    for s, _, o, _ in triples:
        nbr[fold(s)].add(fold(o))
        nbr[fold(o)].add(fold(s))
    sn_index = sent_names.to_dict()

    def context_names(parva, sid):
        out = set()
        for d in range(-WINDOW, WINDOW + 1):
            out |= sn_index.get((parva, sid + d), set())
        return out

    # --- score every (mention, candidate)
    rows = []
    for i, r in enumerate(men.itertuples()):
        near = context_names(r.parva, r.sent_id) - set(r.cands) - {r.key}
        for c, src in r.cands.items():
            A = 0.7 if src == {"spelling"} else 1.0
            C = to_c(cos(i, c))
            R = len(near & nbr[c]) / len(near) if near else 0.5
            S = max(S_SOURCE[s] for s in src)
            E = 1.0 if labels.get(c) == "PERSON" else 0.0
            conf = W["A"] * A + W["C"] * C + W["R"] * R + W["S"] * S + W["E"] * E
            rows.append((r.mention_id, c, A, C, R, S, E, conf, "+".join(sorted(src))))
    pairs = pd.DataFrame(rows, columns=["mention_id", "candidate", "A", "C", "R", "S", "E",
                                        "confidence", "sources"])
    pairs["candidate"] = pairs.candidate.map(lambda k: display.get(k, k))
    links_df = decide(pairs, men, display)
    page = write_outputs(pairs, links_df, men, display, triples, rng)
    gold = None
    if os.path.exists(GOLD):
        gold = evaluate(pairs, links_df)
    else:
        print("%s not found: no precision, recall, calibration or learned weights yet."
              " Fill %s/gold_template.csv and install it there." % (GOLD, OUT))
    write_page(pairs, links_df, page, gold)


def decide(pairs, men, display):
    pairs = pairs.sort_values(["mention_id", "confidence"], ascending=[True, False])
    top = pairs.groupby("mention_id").head(2)
    first = top.groupby("mention_id").first()
    second = top.groupby("mention_id").nth(1).set_index("mention_id")
    out = men.set_index("mention_id")[["parva", "sent_id", "adhyaya_ordinal", "tier",
                                        "surface", "sentence", "n_cands"]].join(first)
    out["runner_up"] = second.candidate
    out["margin"] = (out.confidence - second.confidence).fillna(1.0)
    b = out.confidence.map(band)
    out["band"] = [x[0] for x in b]
    out["action"] = [x[1] for x in b]
    close = (out.margin < MARGIN) & out.action.isin(["auto_accept", "accept_sample_verify"])
    out.loc[close, "action"] = "human_review"
    out["close_runner_up"] = close
    return out.reset_index()


# ---------------------------------------------------------------- graph effect

_TRIPLE = re.compile(r"mbh:subject <[^>]*/e/([^>]+)> ; mbh:predicate mbh:(\w+) ; "
                     r"mbh:object <[^>]*/e/([^>]+)>.*?mbh:inPassage <[^>]*/p/([^>]+)>")


def read_relations(path):
    if not os.path.exists(path):
        return []
    return [(s.replace("_", " "), p, o.replace("_", " "), psg)
            for s, p, o, psg in _TRIPLE.findall(open(path, encoding="utf-8").read())]


def gate(triples):
    t = {(s, p, o) for s, p, o in triples}
    deg = collections.Counter()
    for s, _, o in t:
        deg[s] += 1
        deg[o] += 1
    share = collections.Counter(p for _, p, _ in t)
    top, n = share.most_common(1)[0] if share else ("-", 0)
    top_ent = deg.most_common(1)[0] if deg else ("-", 0)
    return {"triples": len(t), "entities": len(deg),
            "degree_ge_5": sum(1 for v in deg.values() if v >= 5),
            "max_degree": "%s %d" % top_ent,
            "mean_degree": round(sum(deg.values()) / max(len(deg), 1), 2),
            "top_relation": top, "top_share": round(n / max(len(t), 1), 4)}


ACCEPTED = ("auto_accept", "accept_sample_verify")


def endpoint_map(links_df):
    """A function (name, parva, sent_id, chapter) -> entity, from accepted links only.

    Most specific first: the accepted link of that name in that sentence, then
    the chapter where every accepted mention of the name agrees on one entity,
    then corpus-wide, but only for a name with a single candidate everywhere:
    fifteen accepted "Parshata" mentions agreeing on Dhrishtadyumna say nothing
    about the 149 that went to review and may be Drupada. Anything else keeps
    the name it was written with."""
    lf = links_df.assign(s=links_df.surface.map(clean_surface))
    acc = lf[lf.action.isin(ACCEPTED)]
    acc = acc.assign(ch=acc.parva + "/" + acc.adhyaya_ordinal.astype("Int64").astype(str))
    unique = lambda g: {k: next(iter(v)) for k, v in g.candidate.agg(set).items() if len(v) == 1}
    by_sent = unique(acc.groupby(["s", "parva", "sent_id"]))
    by_ch = unique(acc.groupby(["s", "ch"]))
    single = lf.groupby("s").n_cands.max() == 1
    glob = unique(acc[acc.s.map(single)].groupby("s"))

    def m(name, parva=None, sent_id=None, chapter=None):
        if sent_id is not None and (name, parva, sent_id) in by_sent:
            return by_sent[(name, parva, sent_id)]
        return by_ch.get((name, chapter), glob.get(name, name))
    return m


def remap(triples, links_df):
    """Relation endpoints of out/relations.ttl rewritten through accepted links.
    A ttl triple carries a chapter and no sentence, so it maps at chapter level."""
    m = endpoint_map(links_df)
    out = []
    for s, p, o, psg in triples:
        s2, o2 = m(s, chapter=psg), m(o, chapter=psg)
        if s2 != o2:
            out.append((s2, p, o2))
    return out


# ---------------------------------------------------------------- outputs

def write_outputs(pairs, links_df, men, display, triples, rng):
    pairs.to_csv(os.path.join(OUT, "candidate_features.csv"), index=False)
    cols = ["mention_id", "parva", "sent_id", "adhyaya_ordinal", "surface", "candidate", "confidence",
            "band", "action", "runner_up", "margin", "A", "C", "R", "S", "E", "sources",
            "n_cands", "sentence"]
    links_df[cols].to_csv(os.path.join(OUT, "mention_links.csv"), index=False)
    links_df[links_df.action == "human_review"][cols].to_csv(
        os.path.join(OUT, "review_queue.csv"), index=False)

    # Canonical store and alias table, from accepted links only.
    acc = links_df[links_df.action.isin(ACCEPTED)]
    ents = acc.candidate.value_counts()
    pid = {e: "P%04d" % (i + 1) for i, e in enumerate(ents.index)}
    alias = (acc.assign(alias=acc.surface.map(clean_surface))
             .groupby(["candidate", "alias"])
             .agg(mentions=("mention_id", "size"), mean_confidence=("confidence", "mean"),
                  sources=("sources", lambda x: ";".join(sorted(set("+".join(x).split("+"))))))
             .reset_index())
    alias.insert(0, "person_id", alias.candidate.map(pid))
    alias = alias.rename(columns={"candidate": "canonical_name"})
    alias["alias_type"] = ""       # patronymic, epithet, spelling: a human fills these
    alias["language"] = ""
    alias.sort_values(["person_id", "mentions"], ascending=[True, False]).to_csv(
        os.path.join(OUT, "alias_table.csv"), index=False)
    store = (alias.groupby(["person_id", "canonical_name"])
             .apply(lambda g: pd.Series({
                 "mentions": g.mentions.sum(),
                 "aliases": "; ".join(a for a in g.alias if a != g.name[1])}),
                    include_groups=False)
             .reset_index().sort_values("person_id"))
    store.to_csv(os.path.join(OUT, "canonical_entities.csv"), index=False)

    write_gold_template(links_df, pairs, rng)

    before, after = gate(triples and [t[:3] for t in triples]), gate(remap(triples, links_df))
    by_band = links_df.band.value_counts().reindex([b[1] for b in BANDS]).fillna(0).astype(int)
    by_action = links_df.action.value_counts()
    amb = links_df[links_df.n_cands > 1]
    L = ["# Phase 2b. Confidence-aware entity resolution", "",
         "Generated by `src/phase2b_confidence.py`. Proposals only: nothing here is"
         " written to `out/`, and the weights are the design document's, not learned,"
         " unless the gold-standard section below says otherwise.", "",
         "Confidence is not accuracy. Until `%s` exists these scores are uncalibrated." % GOLD,
         "", "## Mentions", "",
         "- PERSON mentions scored: **%d**" % len(links_df),
         "- with more than one candidate: **%d**" % len(amb),
         "- runner-up within %.2f, sent to review whatever the score: **%d**"
         % (MARGIN, int(links_df.close_runner_up.sum())), "",
         "| band | mentions |", "|---|---|"] + \
        ["| %s | %d |" % (k, v) for k, v in by_band.items()] + \
        ["", "| action | mentions |", "|---|---|"] + \
        ["| %s | %d |" % (k, v) for k, v in by_action.items()] + \
        ["", "## Ambiguous surfaces, how their mentions split", "",
         "| surface | mentions | accepted as | to review |", "|---|---|---|---|"]
    ambiguous = []
    for s, g in amb.groupby(amb.surface.map(clean_surface)):
        if len(g) < 20:
            continue
        a = g[g.action.isin(ACCEPTED)].candidate.value_counts()
        ambiguous.append({"surface": s, "mentions": len(g),
                          "accepted": {k: int(v) for k, v in a.items()},
                          "review": int((g.action == "human_review").sum())})
        L.append("| %s | %d | %s | %d |" % (s, len(g), ", ".join("%s %d" % kv for kv in a.items())
                                             or "-", (g.action == "human_review").sum()))
    L += ["", "## Effect on the Phase 6 entry gate", "",
          "`out/relations.ttl` with its endpoints rewritten through accepted links only.", "",
          "| measure | before | after | required |", "|---|---|---|---|",
          "| distinct triples | %d | %d | >= 10,000 |" % (before["triples"], after["triples"]),
          "| entities | %d | %d | |" % (before["entities"], after["entities"]),
          "| entities of degree >= 5 | %d | %d | >= 500 |" % (before["degree_ge_5"],
                                                             after["degree_ge_5"]),
          "| largest relation share | %.1f%% %s | %.1f%% %s | <= 40%% |"
          % (100 * before["top_share"], before["top_relation"],
             100 * after["top_share"], after["top_relation"]),
          "| mean degree | %.2f | %.2f | |" % (before["mean_degree"], after["mean_degree"]),
          "| highest degree | %s | %s | |" % (before["max_degree"], after["max_degree"]), "",
          "Read these with care. `out/relations.ttl` is not the parquet the gate reads, so"
          " the before column is this file's own count, not the README's. Normalization"
          " merges nodes: Shakra, Vasava and Maghavan become Indra. A triple written twice"
          " under two names becomes one, a triple between two names of one person"
          " becomes a self-loop and is dropped, and several nodes of degree 5 become one."
          " So the triple count and the degree-5 count can fall while each character's"
          " node gets deeper. Normalization cannot add a triple; the 10,000 and the 40%"
          " are extraction problems.", ""]
    open(os.path.join(OUT, "er_report.md"), "w", encoding="utf-8").write("\n".join(L))
    json.dump({"before": before, "after": after, "bands": by_band.to_dict(),
               "actions": by_action.to_dict(), "weights": W},
              open(os.path.join(OUT, "er_summary.json"), "w"), indent=1)
    print("\n".join(L))
    return {"before": before, "after": after, "ambiguous": ambiguous, "alias": alias,
            "summary": {"scored": len(links_df), "multi": len(amb),
                        "close": int(links_df.close_runner_up.sum()),
                        "bands": {k: int(v) for k, v in by_band.items()},
                        "actions": {k: int(v) for k, v in by_action.items()}}}


def write_gold_template(links_df, pairs, rng, n=1000, per_surface=15):
    """1,000 mentions for a human to link, as the document suggests. Stratified by
    band, at most 15 per surface so "Partha" cannot fill it. The system's answer
    is kept out of the sheet so the annotator is not anchored on it; it sits in
    gold_key.csv, which should not be opened before the sheet is done."""
    cand_lists = pairs.groupby("mention_id").candidate.agg(lambda x: "; ".join(sorted(x)))
    df = links_df.assign(s=links_df.surface.map(clean_surface))
    df = df.sample(frac=1, random_state=SEED).groupby("s").head(per_surface)
    take = df.groupby("band").head(n // len(BANDS))
    if len(take) < n:
        rest = df.drop(take.index)
        take = pd.concat([take, rest.sample(min(len(rest), n - len(take)), random_state=SEED)])
    take = take.sample(frac=1, random_state=SEED)
    sheet = pd.DataFrame({
        "mention_id": take.mention_id, "parva": take.parva, "adhyaya": take.adhyaya_ordinal,
        "surface": take.surface,
        "sentence": [s.replace(m, "[[" + m + "]]", 1) for s, m in zip(take.sentence, take.surface)],
        "candidates": take.mention_id.map(cand_lists),
        "correct_entity": ""})
    sheet.to_csv(os.path.join(OUT, "gold_template.csv"), index=False)
    take[["mention_id", "candidate", "confidence", "band", "action"]].to_csv(
        os.path.join(OUT, "gold_key.csv"), index=False)


# ---------------------------------------------------------------- gold standard

def evaluate(pairs, links_df):
    """P, R, F1 and calibration from annotations/er_gold.csv, then learned weights.

    correct_entity holds a candidate name, OTHER for a person outside the set, or
    NOT_A_PERSON. UNSURE and blank rows are dropped. A mention counts as predicted
    when the system accepts it; TP when the accepted entity is the gold one."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    gold = pd.read_csv(GOLD, dtype=str).dropna(subset=["correct_entity"])
    gold["correct_entity"] = gold.correct_entity.str.strip()
    gold = gold[~gold.correct_entity.str.upper().isin(["", "UNSURE"])]
    g = links_df.merge(gold[["mention_id", "correct_entity"]], on="mention_id")
    has_gold_entity = ~g.correct_entity.str.upper().isin(["OTHER", "NOT_A_PERSON"])
    data = {"n": len(g), "thresholds": [], "calibration": [], "coef": None}
    L = ["", "## Gold standard (%d mentions from %s)" % (len(g), GOLD), "",
         "| accept at | predicted | TP | FP | FN | precision | recall | F1 |",
         "|---|---|---|---|---|---|---|---|"]
    for t in (0.90, 0.75, 0.50):
        pred = g.confidence >= t
        tp = int((pred & (g.candidate == g.correct_entity)).sum())
        fp = int(pred.sum()) - tp
        fn = int(has_gold_entity.sum()) - tp
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        L.append("| %.2f | %d | %d | %d | %d | %.3f | %.3f | %.3f |"
                 % (t, pred.sum(), tp, fp, fn, p, r, f1))
        data["thresholds"].append([t, int(pred.sum()), tp, fp, fn, p, r, f1])
    y = (g.candidate == g.correct_entity).astype(int).values
    lines, cal = reliability("design weights", g.confidence.values, y)
    L += lines
    data["calibration"].append(cal)

    # Learned weights: one row per (mention, candidate), grouped by surface.
    pg = pairs.merge(gold[["mention_id", "correct_entity"]], on="mention_id")
    pg["y"] = (pg.candidate == pg.correct_entity).astype(int)
    pg["group"] = pg.mention_id.map(links_df.set_index("mention_id").surface.map(clean_surface))
    X = pg[list(W)].values
    oof = np.zeros(len(pg))
    folds = min(5, pg.group.nunique())
    if folds >= 2 and pg.y.nunique() == 2:
        for tr, te in GroupKFold(folds).split(X, pg.y, pg.group):
            if pg.y.iloc[tr].nunique() < 2:
                continue
            oof[te] = LogisticRegression(max_iter=1000).fit(X[tr], pg.y.iloc[tr]).predict_proba(X[te])[:, 1]
        full = LogisticRegression(max_iter=1000).fit(X, pg.y)
        coef = full.coef_[0]
        pg["p"] = oof
        top = pg.sort_values("p", ascending=False).groupby("mention_id").head(1)
        lines, cal = reliability("learned weights, out of fold", top.p.values, top.y.values)
        L += lines
        data["calibration"].append(cal)
        data["coef"] = []
        L += ["", "Learned coefficients (logistic regression, all gold rows):", "",
              "| feature | coefficient | share of positive weight | design weight |",
              "|---|---|---|---|"]
        pos = np.clip(coef, 0, None).sum() or 1.0
        for f, c in zip(W, coef):
            L.append("| %s | %.3f | %.2f | %.2f |" % (f, c, max(c, 0) / pos, W[f]))
            data["coef"].append([f, float(c), float(max(c, 0) / pos), W[f]])
        pairs["p_learned"] = full.predict_proba(pairs[list(W)].values)[:, 1]
        pairs.to_csv(os.path.join(OUT, "candidate_features_learned.csv"), index=False)
    else:
        L.append("\nToo few labelled surfaces or classes to learn weights.")
    open(os.path.join(OUT, "er_report.md"), "a", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    return data


def reliability(name, p, y, bins=10):
    """(report lines, the same numbers as data for the page)."""
    if len(p) == 0:
        return ["", "No rows for %s." % name], {"name": name, "ece": 0.0, "brier": 0.0, "rows": []}
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
              for b in range(bins) if (idx == b).any())
    L = ["", "Calibration, %s: ECE %.3f, Brier %.3f" % (name, ece, np.mean((p - y) ** 2)), "",
         "| confidence | mentions | mean confidence | share correct |", "|---|---|---|---|"]
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            L.append("| %.1f to %.1f | %d | %.3f | %.3f |"
                     % (edges[b], edges[b + 1], m.sum(), p[m].mean(), y[m].mean()))
            rows.append(["%.1f to %.1f" % (edges[b], edges[b + 1]), int(m.sum()),
                         float(p[m].mean()), float(y[m].mean())])
    return L, {"name": name, "ece": float(ece), "brier": float(np.mean((p - y) ** 2)), "rows": rows}


# ---------------------------------------------------------------- page

def write_page(pairs, links_df, page, gold):
    """build/er/er.html, er-data.js and explorer.css, the way out/index.html is
    built: a fixed page from src/er_page.html and its data as a .js file that
    assigns one global, so the folder opens from file:// with no server."""
    import shutil
    here = os.path.dirname(os.path.abspath(__file__))
    cands = {}
    for mid, g in pairs.sort_values("confidence", ascending=False).groupby("mention_id", sort=False):
        cands[mid] = [[c, round(f, 3), round(a, 3), round(cc, 3), round(r, 3), round(s, 3),
                       round(e, 3), src]
                      for c, f, a, cc, r, s, e, src in zip(g.candidate, g.confidence, g.A, g.C,
                                                           g.R, g.S, g.E, g.sources)]
    actions = [b[2] for b in BANDS]
    sentences, sidx, mentions = [], {}, []
    for r in links_df.itertuples():
        if r.sentence not in sidx:
            sidx[r.sentence] = len(sentences)
            sentences.append(r.sentence)
        ad = None if pd.isna(r.adhyaya_ordinal) else int(r.adhyaya_ordinal)
        mentions.append([r.mention_id, r.surface, r.parva, ad, sidx[r.sentence],
                         actions.index(r.action), cands[r.mention_id]])
    entities = []
    for (pid, name), g in page["alias"].groupby(["person_id", "canonical_name"]):
        g = g.sort_values("mentions", ascending=False)
        entities.append({"id": pid, "name": name, "mentions": int(g.mentions.sum()),
                         "aliases": [[a, int(n), round(float(c), 3), s] for a, n, c, s in
                                     zip(g.alias, g.mentions, g.mean_confidence, g.sources)]})
    entities.sort(key=lambda e: e["id"])
    data = {"weights": W, "bands": [[lo, n, a] for lo, n, a in BANDS], "actions": actions,
            "margin": MARGIN, "summary": page["summary"],
            "gate": {"before": page["before"], "after": page["after"]},
            "ambiguous": page["ambiguous"], "entities": entities,
            "sentences": sentences, "mentions": mentions, "gold": gold}
    with open(os.path.join(OUT, "er-data.js"), "w", encoding="utf-8") as f:
        f.write("window.KGER=")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    shutil.copy(os.path.join(here, "er_page.html"), os.path.join(OUT, "er.html"))
    if os.path.exists("out/explorer.css"):
        shutil.copy("out/explorer.css", os.path.join(OUT, "explorer.css"))
    print("page: %s/er.html with er-data.js, %.1f MB, %d mentions, %d characters"
          % (OUT, os.path.getsize(os.path.join(OUT, "er-data.js")) / 1e6,
             len(mentions), len(entities)))
    write_standalone(here)


def write_standalone(here):
    """build/er/mahabharata_er.html: the same page as one file, stylesheet and data
    inlined, for when the three files would be separated (a single download from
    Kaggle's Output tab, an email attachment). "</" inside the data is escaped so
    a sentence can never close the script element early."""
    html = open(os.path.join(here, "er_page.html"), encoding="utf-8").read()
    css = (open("out/explorer.css", encoding="utf-8").read()
           if os.path.exists("out/explorer.css") else "")
    data = open(os.path.join(OUT, "er-data.js"), encoding="utf-8").read().replace("</", "<\\/")
    link = '<link rel="stylesheet" href="explorer.css">'
    script = '<script defer src="er-data.js"></script>'
    assert html.count(link) == 1 and html.count(script) == 1, "er_page.html changed shape"
    html = html.replace(link, "<style>\n" + css + "\n</style>")
    html = html.replace(script, "<script>\n" + data + "</script>")
    path = os.path.join(OUT, "mahabharata_er.html")
    open(path, "w", encoding="utf-8").write(html)
    print("single-file page: %s, %.1f MB" % (path, os.path.getsize(path) / 1e6))


if __name__ == "__main__":
    main()
