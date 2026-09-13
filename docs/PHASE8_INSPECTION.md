# Phase 8, hand inspection of the top 20

The gate for Phase 8 asks whether the entities at the head of the divergence
list are interesting or artifacts of low degree. Neither, as it turns out. The
list is an artifact of something else, and the artifact is the finding.

Read `build/divergence.md` alongside this. Everything here is below the Phase 6
entry gate: the model was trained on 1,845 triples against a stated floor of
10,000, on instruction, so none of it is a claim about the Mahabharata.

## The ranking does not discriminate

618 of the 688 scored entities share nothing at all between their top-10 graph
neighbourhood and their top-10 embedding neighbourhood. Only two entities reach
three or four shared neighbours out of ten. Ranking by lowest overlap therefore
orders a tie, and the secondary sort by degree means the printed top 20 is just
the twenty highest-degree entities in the graph: Yudhishthira 66, Drona 62,
Indra 39, Dhrishtadyumna 26, Vasudeva 26, and so on down to Bhagadatta 15.

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
Phase 2 could not complete. It should be checked against `build/appellations.csv`
before it is trusted, and that check has not been run.

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
- Embedding: Shalya, Gangeya, Jayadratha, Duryodhana, Radheya, Duhshasana,
  Bhishma, Satyaki, Shatanika, Suyodhana. Seven of the ten are Kauravas, one is
  Karna's own patronymic Radheya, and one is Duryodhana under a second name.

The stated aim was Karna's vector sitting near the Pandava cluster although the
symbolic edges place him with the Kauravas. What the trained model shows is the
exact inverse: the symbolic edges place him among Pandavas, because the edges are
fights, and the vector places him among Kauravas, because his fights are the
Kauravas' fights. The hypothesis as written assumed a graph built from alliance
and kinship. This graph is built from combat and speech, and the sign of the
expected effect flips with it.

Duryodhana, degree 52, two shared: graph gives Dhritarashtra, Bhimasena, Karna,
Bhima, Drona, Radheya, Yudhishthira, Satyaki, Shakuni, Shalya; the embedding
gives Shalya, Duhshasana, Suyodhana (his own second name), Karna, Ashvatthama,
Bhishma, Kouravya, Radheya, Partha, Alambusa. Same pattern, one side each.

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
  fitted to 1,845 triples at MRR 0.14 is not a reliable source of neighbourhoods.

Neither is work that can be done by rerunning a script.
