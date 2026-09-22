// Run the explorer's own query and scoring engine against the shipped vectors.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {create} = require('../out/kg-core.js');
global.window = {};
require('../out/projection.js');
require('../out/evidence.js');
const c = create(window.KG, window.KGTeaching || [], window.KGReview), K = c.data;
const toy = create({entities:['A','B'], relations:['rotation'], dim:2,
  emb:[1,0,0,1], rel_emb:[0,1], edges:[]});
assert(toy.score(0,0,1) === 0, 'Complex rotation should map A exactly onto B');
assert.equal(toy.score(1,0,0), -2, 'Inverse query must preserve relation direction');
assert.deepEqual(toy.predict(0,'rotation').map(x=>x[0]), [1,0]);
assert.equal(toy.parse('A - ? + B'), null);
const L = ['# Phase 7 explorer checks', '',
  '**Below the Phase 6 entry gate.** The asserted graph contains '+K.n_asserted+
  ' triples; '+K.n_trained+' remain after symmetric deduplication.',
  'These checks verify execution, not historical truth or predictive accuracy.', '',
  'The tests import `out/kg-core.js`, the same code used by `out/index.html`.',
  'RotatE: '+c.D/2+' complex dimensions; seed '+K.train_seed+'. UMAP seed '+K.umap_seed+'.', '',
  '## Natural-language queries', '', '| Query | Result |', '|---|---|'];
const cases = [
  ['Who fought Bhishma?', 'relation', 'Bhishma'],
  ['Who was Karna killed by?', 'relation', 'Karna'],
  ['Who is Arjuna’s father?', 'relation', 'Arjuna'],
  ["Who are Dhritarashtra's children?", 'relation', 'Dhritarashtra'],
  ['Who does Krishna speak to?', 'relation', 'Krishna'],
  ['Who studied under Drona?', 'relation', 'Drona'],
  ['Who rules Madra?', 'relation', 'Madra'],
  ['Who is like Karna?', 'nearest', 'Karna'],
  ['Who is Draupadi?', 'nearest', 'Droupadi'],
  ['Who killed Abhimanyu?', 'contested', 'Abhimanyu'],
  ["Who are Bhishma's enemies?", 'unavailable', 'Bhishma'],
  ['What is the meaning of dharma?', 'error', null]
];
for (const [q, kind, name] of cases) {
  const r = c.execute(q);
  assert.equal(r.kind,kind,q);
  if(name) assert.equal(K.entities[r.focus],name,q);
  L.push('| '+q+' | '+kind+(name?' / '+name:'')+' |');
}
assert.equal(c.parse('? speaksTo Krishna').direction,'o');
assert.equal(c.parse('Duryodhana foughtWith ?').direction,'s');
assert.equal(c.parse('? speaksTo ?'),null);
const teaching = c.execute('Who taught Karna?');
assert(teaching.passages?.some(p => p.teacher === 'Bhargava' && p.student === 'Karna' && p.chapter === 24),
  'Missing graph links must not hide the explicit Bhargava / Karna passage');
assert.equal(teaching.edges.length, 0, 'Source retrieval must not insert an edge into the trained graph');
for (const q of ["Who was Karna taught by?", "Who is Karna's teacher?"])
  assert.equal(c.execute(q).passages.length, teaching.passages.length, q);
assert(c.execute('Who studied under Bhargava?').passages.some(p=>p.student==='Karna'));
const connection=c.execute('How is Karna related to Arjuna?');
assert.equal(connection.kind,'path');
assert(connection.sharedParentage?.some(p=>p.parent==='Kunti'), 'Related-to questions should expose the shared Kunti passages');
assert(c.parentageOf(c.resolve('Arjuna')).some(p=>p.object==='Kunti'), 'Parentage must remain available in the normal profile');
assert(!c.parentageOf(c.resolve('Bhishma')).some(p=>p.object==='Krishna'), 'Rejected archive candidates must not return as source parentage');
for (const [relation,object] of [['bornTo','Krishna'],['rules','Chedi'],['rules','Madra']])
  assert(!K.edges.some(e=>K.entities[e[0]]==='Bhishma' && K.relations[e[1]]===relation && K.entities[e[2]]===object), 'Introductory-clause attachment: '+relation+' '+object);
const slayers = c.execute('who killed Karna').edges.map(i=>K.entities[K.edges[i][2]]);
assert(!slayers.includes('Nakula')&&!slayers.includes('Vrisha'), 'Pronoun errors must not survive as named killers');
assert.deepEqual(c.displayEdges(c.execute('who killed Karna').edges,c.resolve('Karna')).map(i=>K.entities[K.edges[i][2]]), ['Arjuna']);
assert(K.review.death_accounts?.Karna?.appellations.includes('Kiriti'));
assert(connection.path.length > 0);
assert.equal(connection.path[0].from,c.resolve('Karna'));
assert.equal(connection.path.at(-1).to,c.resolve('Arjuna'));
assert(connection.path.every(s=>K.edges[s.edge].includes(s.from) && K.edges[s.edge].includes(s.to)));
assert.equal(c.execute('How is Karna related to Karna?').path.length,0);
assert.equal(c.execute('who is allied with Duryodhana').kind,'unavailable');
for (const [child,parent] of [['Sanjaya','Vasava'],['Keshava','Vasava'],['Karna','Dhritarashtra'],['Karna','Duryodhana'],['Karna','Shantanu']])
  assert(!K.edges.some(e=>K.entities[e[0]]===child&&K.entities[e[2]]===parent&&/Parent|Progenitor/.test(K.relations[e[1]])), child+' must not be asserted as the child of '+parent);
assert(c.execute("Who is Karna's mother?").parentage.some(r=>r.object==='Radha'&&r.parentage_unspecified));
assert(K.review.notes.Vasava.text.includes('Indra'));
assert.equal(typeof K.review.stats.model_graph_matches_review,'boolean');
assert.equal(c.canPredict('hasDivineProgenitor'),window.KG.relations.includes('hasDivineProgenitor'));
assert(K.edges.some(e=>K.relations[e[1]]==='hasDivineProgenitor'));
const graphOnly = create({entities:['A','B'], relations:['rotation'], dim:2,
  emb:[1,0,0,1], rel_emb:[0,1], edges:[]}, [],
  {base_entities:['A','B'],base_relations:['rotation'],relations:['rotation','hasDivineProgenitor'],
   edges:[[0,1,1]],withheld:[],stats:{}});
assert(!graphOnly.canPredict('hasDivineProgenitor'));
assert(!graphOnly.data.review.stats.model_graph_matches_review);
assert.throws(()=>graphOnly.predict(0,'hasDivineProgenitor'));
assert.throws(()=>create(window.KG,[],{...window.KGReview,base_entities:['wrong index']}));
const named = c.resolve('Karna');
const forward = c.predict(named,'foughtWith');
assert.equal(forward.length,K.entities.length,'Score every candidate');
assert(forward.every((r,i)=>Number.isFinite(r[1])&&(i===0||r[1]<=forward[i-1][1])));
const reversed = c.predict(named,'speaksTo','o');
assert.equal(reversed.length,K.entities.length);
assert.equal(reversed[0][1],c.score(reversed[0][0],K.relations.indexOf('speaksTo'),named));
L.push('', '## Analogy ranks', '', '| Expression | Duryodhana rank |', '|---|---|');
for (const q of ['Arjuna - Kunti + Gandhari','Arjuna - Pandu + Dhritarashtra','Yudhishthira - Kunti + Gandhari']) {
  const r=c.execute(q);assert.equal(r.kind,'analogy');assert.equal(r.ranking.length,K.entities.length-3);
  L.push('| '+q+' | '+(r.ranking.findIndex(x=>K.entities[x[0]]==='Duryodhana')+1)+' |');
}
L.push('', 'The analogy target is not forced into the results. Its scientific acceptance gate remains unmet.', '',
  '## Graph integrity recheck', '',
  K.edges.length+' displayed links; '+K.review.stats.withheld_instances+' withheld source records.',
  'Displayed links '+(K.review.stats.model_graph_matches_review?'match':'differ from')+' the bundled model graph.',
  'Generic parentage remains a source statement with unspecified kind, not a biological assertion.', '',
  '## Browser verification', '',
  'Run `.venv/bin/python src/test_viewer_browser.py` for offline Chrome rendering,',
  'canvas pixels, point selection, orbit, dimension switching, queries, evidence,',
  'export, and mobile/tablet/desktop layout. Screenshots and its report go to',
  '`build/viewer_checks/`. This Node check does not substitute for a browser test.', '');
fs.writeFileSync(path.join(__dirname,'../out/phase7_check.md'),L.join('\n'));
console.log('Explorer core checks passed: complex rotation, both prediction directions, 12 language cases, 3 analogies.');
