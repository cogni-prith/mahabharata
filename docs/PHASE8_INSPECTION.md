# Phase 8, hand inspection of the top 20

The gate for Phase 8 asks whether the entities at the head of the divergence
list are interesting or artifacts of low degree. Neither, as it turns out. The
list is an artifact of something else, and the artifact is the finding.

Read `build/divergence.md` alongside this. Everything here is below the Phase 6
entry gate: the model was trained on 1,771 triples against a stated floor of
10,000, on instruction, so none of it is a claim about the Mahabharata.

## The ranking does not discriminate

616 of the 683 scored entities share nothing at all between their top-10 graph
neighbourhood and their top-10 embedding neighbourhood. Four reach three shared
out of ten and one reaches six. Ranking by lowest overlap therefore orders a tie,
and the secondary sort by degree means the printed top 20 is just the twenty
highest-degree entities in the graph: Drona 60, Bhishma 45, Indra 38, Narada 25,
Vasudeva 25, and so on down to Brihaspati 14.

Low degree is not the confound here. The confound is the opposite one. The
highest-degree entities in the graph are exactly the ones the embedding disagrees
with most, because they are the ones with enough edges for the disagreement to
be visible.

## What the two neighbourhoods are actually measuring

Reading the pairs by hand, the disagreement is systematic and has two causes.

**The embedding groups appellations of the same person.** Yudhishthira's cosine
neighbours include Dharmaraja, Yudhisthira (a spelling variant carried as a
separate entity) and Kouravya. Vasudeva's include Madhusudana, Achyuta, Govinda,
Janardana and Dasharha. Indra's include Vasava, Maghavan and Shatakratu. Shakra's
include Maghavan, Matali and Puruhuta. These are the same figures under other
names, unmerged because Phase 2 left all 246 appellations at status `ambiguous`.
The embedding finds them without being told, because an epithet stands in the
same triples as its bearer and RotatE places entities with the same structural
role together. They can never appear in the graph neighbourhood, because the
graph has no edge between a name and another name for the same person. Every one
of those slots is a guaranteed miss.

That is a real and usable result about the method, not about the epic: cosine
similarity in this embedding is a candidate generator for the alias merge that
Phase 2 could not complete. It has now been checked against
`build/appellations.csv` by `src/phase8b_alias.py`, and the check is in
`build/alias_candidates.md`.

The check, since a dozen rows read by hand is a hypothesis and not a result. 185
of the 246 appellation rows have both names embedded. There is no gold alias
mapping, so this is not precision; it is agreement between apposition read off
the prose in Phase 2 and cosine distance in the embedding, which are independent.
An apposition-evidenced candidate lands in the cosine top 10 for 15.7% of pairs
against 1.5% by chance, top 25 for 30.8% against 3.7%, with a median rank of 93
out of 683 and 75 on the pairs with two or more sightings. Where each signal
names one best candidate, they agree for 71 of 94 appellations, 75.5%; restricted
to the 40 appellations where more than one candidate exists and the pick is a
real choice, 42.5%.

So the signal is real and far above chance, and it is nowhere near strong enough
to merge on. It is a generator. The shortlist it produces is 17 pairs, and by eye
most of them are right: Dharmaraja/Yudhishthira, Janardana/Krishna,
Saindhava/Jayadratha, Hardikya/Kritavarma, Partha/Arjuna, Gangeya/Bhishma,
Devavrata/Bhishma, Yajnasena/Drupada. One of the seventeen carries no apposition
span at all, only the weak `canonical_id` prior, and it is printed as uncited
rather than given an address. Richika lands on Bhargava, which is every descendant
of Bhrigu and not one man, so the list still reproduces the conflation failure
mode Phase 2 measured. Each row now prints its passage underneath so the ruling
is a reading rather than a hunt. Nothing was merged.

**Graph adjacency here mostly means opposition.** `foughtWith`, `slainBy` and
`defeatedBy` are 35% of the asserted triples, and all three join enemies. So the
graph neighbourhood of a warrior is a list of the people who fought him, while
the embedding neighbourhood is a list of the people who fought the same people he
did, which is his own side. The two lists are close to complementary by
construction.

Karna is the clearest case and it runs against the hypothesis this project set
out to test. Degree 53, two shared of ten.

- Graph: Bhima, Bhimasena, Duryodhana, Shalya, Arjuna, Ghatotkacha, Nakula,
  Partha, Vrishasena, Dhrishtadyumna. Seven of the ten are Pandavas or Pandava
  allies, because Karna's asserted edges are overwhelmingly combat.
- Embedding: Duryodhana, Shalya, Jayadratha, Bhishma, Shrutayudha, Shikhandi,
  Ashvatthama, Satyaki, Drona, Vaidehi. Seven of the ten are Kauravas or fight on
  that side; only Shikhandi and Satyaki are Pandava.

The stated aim was Karna's vector sitting near the Pandava cluster although the
symbolic edges place him with the Kauravas. What the trained model shows is the
exact inverse: the symbolic edges place him among Pandavas, because the edges are
fights, and the vector places him among Kauravas, because his fights are the
Kauravas' fights. The hypothesis as written assumed a graph built from alliance
and kinship. This graph is built from combat and speech, and the sign of the
expected effect flips with it.

Duryodhana is the exception that shows the same mechanism from the other side.
He is now the highest-overlap entity in the whole list, six of ten shared at
degree 48: graph gives Dhritarashtra, Bhimasena, Karna, Bhima, Drona, Radheya,
Yudhishthira, Shakuni, Shalya, Bhishma; the embedding gives Karna, Ashvatthama,
Kripa, Shakuni, Duhshasana, Shrutayu, Shalya, Drona, Bhishma, Dhritarashtra. The
overlap is high because his asserted edges are not mostly combat. He talks to his
own side, so for him adjacency and similarity point the same way.

## Verdict against the gate

- The ranked list is produced, in `build/divergence.md`, with the full
  distribution printed so the tie is visible rather than hidden by the sort.
- The top 20 are not interesting as individual entities and are not artifacts of
  low degree. They are the highest-degree entities, surfaced by a tie at zero
  overlap that has two identifiable causes: unmerged appellations, which the
  graph cannot express, and a predicate mix in which adjacency means enmity.
- Two things must happen before this analysis can say anything about the epic.
  The appellation merge has to be resolved, since a visible share of the cosine
  slots is currently spent on aliases of the entity itself. And the gate has to pass, since a model
  fitted to 1,771 triples at MRR 0.15 is not a reliable source of neighbourhoods.

Neither is work that can be done by rerunning a script.
