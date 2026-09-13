// Phase 7 gate. The three query modes, run against the same data file the page
// loads and with the same scoring code, so the gate is checked rather than
// asserted. Run: node src/phase7_check.js  (writes out/phase7_check.md)
//
// Below the Phase 6 entry gate. This checks that the modes execute and return
// ranked output on real data. It does not claim the rankings are correct.
const fs = require('fs');
global.window = {};
require('../out/projection.js');
const K = window.KG, N = K.entities.length, D = K.dim, H = D / 2;
const E = Float32Array.from(K.emb), R = Float32Array.from(K.rel_emb);
const id = {}; K.entities.forEach((n, i) => id[n.toLowerCase()] = i);
const rid = {}; K.relations.forEach((n, i) => rid[n.toLowerCase()] = i);
const vec = i => E.subarray(i * D, (i + 1) * D);
function cos(a, b) { let d = 0, x = 0, y = 0; for (let i = 0; i < D; i++) { d += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; } return d / (Math.sqrt(x * y) + 1e-9); }
function rot(h, r) { const o = new Float32Array(D); for (let k = 0; k < H; k++) { const a = h[2 * k], b = h[2 * k + 1], c = r[2 * k], d = r[2 * k + 1]; o[2 * k] = a * c - b * d; o[2 * k + 1] = a * d + b * c; } return o; }
function dist(a, b) { let s = 0; for (let i = 0; i < D; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s); }
const score = (h, r, t) => -dist(rot(vec(h), R.subarray(r * D, (r + 1) * D)), vec(t));
function nn(v, skip, n) { const o = []; for (let i = 0; i < N; i++) { if (i === skip) continue; o.push([K.entities[i], cos(v, vec(i))]); } return o.sort((a, b) => b[1] - a[1]).slice(0, n); }
const fmt = a => a.map(([n, s], i) => (i + 1) + '. ' + n + ' ' + s.toFixed(4)).join(', ');
const L = ['# Phase 7 gate. The three query modes end to end', '',
  '**Below the Phase 6 entry gate.** The model was trained on 1,845 triples against',
  'a stated floor of 10,000, on instruction. This report shows the modes execute on',
  'real data and returns their output verbatim. The rankings are model hypotheses,',
  'labelled `predicted` and `uncited` on the page.', '',
  'Entities ' + N + ', relations ' + K.relations.length + ', stored dimension ' + D +
  ' float32 (' + H + ' complex). Training seed ' + K.train_seed + ', UMAP seed ' + K.umap_seed + '.',
  'Same data file and same scoring code as `out/index.html`.', ''];
// 1. nearest neighbours
L.push('## Mode 1. Nearest neighbours in vector space', '');
['Arjuna', 'Karna', 'Droupadi', 'Duryodhana'].forEach(n => {
  const i = id[n.toLowerCase()];
  L.push('- **' + n + '**: ' + (i === undefined ? 'not in the graph' : fmt(nn(vec(i), i, 8))));
});
// 2. link prediction
L.push('', '## Mode 2. Link prediction, incomplete triple', '');
[['Duryodhana', 'foughtWith'], ['Arjuna', 'hasBiologicalParent'], ['Yudhishthira', 'speaksTo']].forEach(([n, p]) => {
  const h = id[n.toLowerCase()], r = rid[p.toLowerCase()];
  const asserted = new Set(K.edges.filter(e => e[1] === r && e[0] === h).map(e => e[2]));
  const o = []; for (let i = 0; i < N; i++) if (i !== h) o.push([K.entities[i], score(h, r, i)]);
  o.sort((a, b) => b[1] - a[1]);
  L.push('- **(' + n + ', ' + p + ', ?)**: ' + fmt(o.slice(0, 8)) +
    '. Of the top 8, ' + o.slice(0, 8).filter(x => asserted.has(id[x[0].toLowerCase()])).length +
    ' are already asserted, the rest are predicted and uncited.');
});
// 3. analogy
L.push('', '## Mode 3. Analogy by vector arithmetic', '');
const ranks = [];
[['Arjuna', 'Kunti', 'Gandhari', 'Duryodhana'], ['Arjuna', 'Pandu', 'Dhritarashtra', 'Duryodhana'],
 ['Yudhishthira', 'Kunti', 'Gandhari', 'Duryodhana']].forEach(([a, b, c, want]) => {
  const va = vec(id[a.toLowerCase()]), vb = vec(id[b.toLowerCase()]), vc = vec(id[c.toLowerCase()]);
  const v = new Float32Array(D); for (let i = 0; i < D; i++) v[i] = va[i] - vb[i] + vc[i];
  const used = [a, b, c];
  const all = nn(v, -1, N).filter(x => !used.includes(x[0]));
  const rank = all.findIndex(x => x[0] === want) + 1;
  ranks.push([a + ' - ' + b + ' + ' + c, want, rank]);
  L.push('- **' + a + ' - ' + b + ' + ' + c + '**: ' + fmt(all.slice(0, 8)) +
    '. ' + want + ' ranks ' + rank + ' of ' + all.length + '.');
});
const best = ranks.reduce((x, y) => y[2] < x[2] ? y : x);
L.push('', 'All three modes return ranked output.',
  'The analogy demo is not working. Its best form here is `' + best[0] + '`, which puts ' +
  best[1] + ' at rank ' + best[2] + ';',
  'the other forms tried put it at ' + ranks.filter(r => r !== best).map(r => r[2]).join(' and ') + '.',
  'A demo that has to land at rank 1 does not land at rank ' + best[2] + '. That is the',
  'expected consequence of training below the gate and it is not presented as working.', '');
fs.writeFileSync(__dirname + '/../out/phase7_check.md', L.join('\n'));
console.log('wrote out/phase7_check.md');
