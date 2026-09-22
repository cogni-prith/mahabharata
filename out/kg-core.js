/* Shared by the offline explorer and its checks. No DOM or network calls. */
(function (root) {
  "use strict";
  function create(data, teaching = [], review = null) {
    if (review) {
      if (JSON.stringify(review.base_entities) !== JSON.stringify(data.entities) ||
          JSON.stringify(review.base_relations) !== JSON.stringify(data.relations))
        throw Error("The reviewed graph and embedding index differ. Re-export viewer evidence.");
      const keys = (edges, rels) => new Set(edges.map(e=>e[0]+':'+rels[e[1]]+':'+e[2]));
      const bundled = keys(data.edges, data.relations), displayed = keys(review.edges, review.relations);
      review = {...review, stats: {...review.stats, model_graph_matches_review:
        bundled.size === displayed.size && [...bundled].every(k=>displayed.has(k))}};
      data = {...data, vector_relations: data.relations, relations: review.relations,
        edges: review.edges, review};
    }
    const names = data.entities, N = names.length, D = data.dim;
    const vectorRelations = data.vector_relations || data.relations;
    if (!N || !D || D % 2 || data.emb.length !== N * D || data.rel_emb.length !== vectorRelations.length * D)
      throw Error("Embedding dimensions do not match their index.");
    const E = Float32Array.from(data.emb), R = Float32Array.from(data.rel_emb);
    if (![...E, ...R].every(Number.isFinite)) throw Error("Non-finite embedding coordinate.");
    const ids = new Map(names.map((n, i) => [n.toLowerCase(), i]));
    const rids = new Map(data.relations.map((n, i) => [n.toLowerCase(), i]));
    const vrids = new Map(vectorRelations.map((n, i) => [n.toLowerCase(), i]));
    const canPredict = relation => vrids.has(relation.toLowerCase());
    const symmetric = new Set(["foughtWith", "hasSpouse", "isAlliedWith", "isEnemyOf"]);
    const adj = names.map(() => []);
    data.edges.forEach((e, i) => {adj[e[0]].push(i); if (e[2] !== e[0]) adj[e[2]].push(i);});
    const norms = names.map((_, i) => Math.hypot(...E.subarray(i * D, (i + 1) * D)));
    const words = {
      hasBiologicalParent: ["parent", "child"], hasSpouse: ["spouse", "spouse"],
      hasDivineProgenitor: ["divine parent", "divine child"],
      hasFosterParent: ["foster parent", "foster child"], hasLegalParent: ["legal parent", "legal child"],
      bornTo: ["parentage", "child"],
      foughtWith: ["fought", "fought"], slainBy: ["slain by", "slew"],
      defeatedBy: ["defeated by", "defeated"], rules: ["rules", "ruled by"],
      instructedBy: ["taught by", "taught"], cursedBy: ["cursed by", "cursed"],
      speaksTo: ["speaks to", "addressed by"], isEnemyOf: ["enemy of", "enemy of"],
      isAlliedWith: ["allied with", "allied with"]
    };
    const spellings = {draupadi: "Droupadi"};
    const resolve = s => ids.get(String(s).trim().toLowerCase()) ?? null;
    const nameOf = s => resolve(s) ?? resolve(spellings[String(s).trim().toLowerCase()] || "");
    const vec = i => E.subarray(i * D, (i + 1) * D);
    function nearest(v, skip = []) {
      const omit = new Set(Array.isArray(skip) ? skip : [skip]), vn = Math.hypot(...v);
      return names.flatMap((_, i) => {
        if (omit.has(i)) return [];
        let dot = 0; for (let k = 0; k < D; k++) dot += v[k] * E[i * D + k];
        return [[i, dot / (vn * norms[i] || 1)]];
      }).sort((a, b) => b[1] - a[1] || a[0] - b[0]);
    }
    function score(h, r, t) {
      let sum = 0;
      for (let k = 0; k < D; k += 2) {
        const a = E[h * D + k], b = E[h * D + k + 1], c = R[r * D + k], d = R[r * D + k + 1];
        sum += (a * c - b * d - E[t * D + k]) ** 2 + (a * d + b * c - E[t * D + k + 1]) ** 2;
      }
      return -Math.sqrt(sum);
    }
    function predict(focus, relation, direction = "s") {
      const r = vrids.get(relation.toLowerCase());
      if (r === undefined) throw Error("This relation has no trained vector: " + relation);
      return names.map((_, i) => [i, direction === "s" ? score(focus, r, i) : score(i, r, focus)])
        .sort((a, b) => b[1] - a[1] || a[0] - b[0]);
    }
    function relations(focus, relation, direction = "s") {
      const r = rids.get(relation.toLowerCase());
      return adj[focus].filter(i => {const e = data.edges[i]; return e[1] === r &&
        (symmetric.has(relation) || (direction === "s" ? e[0] === focus : e[2] === focus));});
    }
    function parentageOf(focus, direction = "s") {
      return (review?.parentage || review?.withheld || []).filter(r=>r.parentage_unspecified &&
        r[direction==='s'?'subject':'object']===names[focus]);
    }
    function displayEdges(indexes, focus) {
      const account=review?.death_accounts?.[names[focus]];
      if(!account) return indexes;
      const hasNamedSlayer=indexes.some(i=>{const e=data.edges[i];return e[0]===focus &&
        data.relations[e[1]]==='slainBy' && names[e[2]]===account.slayer;});
      return hasNamedSlayer?indexes.filter(i=>{const e=data.edges[i];return !(e[0]===focus &&
        data.relations[e[1]]==='slainBy' && account.appellations.includes(names[e[2]]));}):indexes;
    }
    function pathBetween(start, target, kinOnly = false) {
      const visited = new Map([[start, null]]), queue = [start];
      for (let k = 0; k < queue.length && !visited.has(target); k++) {
        const from = queue[k];
        for (const edge of adj[from]) {
          const e = data.edges[edge], p = data.relations[e[1]];
          if (kinOnly && !["hasBiologicalParent", "hasSpouse"].includes(p)) continue;
          const to = e[0] === from ? e[2] : e[0];
          if (!visited.has(to)) {visited.set(to, {from, to, edge}); queue.push(to);}
        }
      }
      if (!visited.has(target)) return null;
      const path = [];
      for (let node = target; node !== start;) {const step = visited.get(node); path.unshift(step); node = step.from;}
      return path;
    }
    const patterns = [
      [/^who (?:killed|slew|slayed) (.+)$/, "slainBy", "s"],
      [/^who (?:was|is) (.+) (?:killed|slain) by$/, "slainBy", "s"],
      [/^who (?:did|does) (.+) (?:kill|slay)$/, "slainBy", "o"],
      [/^who (?:fought|battled)(?: against| with)? (.+)$/, "foughtWith", "s"],
      [/^who (?:did|does) (.+) (?:fight|battle)(?: with| against)?$/, "foughtWith", "s"],
      [/^who (?:are|were) (.+?)'s (?:enemies|opponents|foes)$/, "isEnemyOf", "s"],
      [/^who (?:are|were) (.+?)'s allies$/, "isAlliedWith", "s"],
      [/^who (?:is|was|are|were) allied (?:with|to) (.+)$/, "isAlliedWith", "s"],
      [/^who (?:is|was|are|were) (?:the )?enem(?:y|ies) of (.+)$/, "isEnemyOf", "s"],
      [/^who defeated (.+)$/, "defeatedBy", "s"],
      [/^who did (.+) defeat$/, "defeatedBy", "o"],
      [/^who (?:is|are|was|were) (?:the )?(?:father|mother|parents?) of (.+)$/, "hasBiologicalParent", "s"],
      [/^who (?:is|are|was|were) (.+?)'s (?:father|mother|parents?)$/, "hasBiologicalParent", "s"],
      [/^who (?:are|were|is) (?:the )?(?:children|sons?|daughters?) of (.+)$/, "hasBiologicalParent", "o"],
      [/^who (?:are|were|is) (.+?)'s (?:children|sons?|daughters?)$/, "hasBiologicalParent", "o"],
      [/^who (?:is|was) (.+) married to$/, "hasSpouse", "s"],
      [/^who (?:is|are|was|were) (.+?)'s (?:wife|husband|spouse|wives)$/, "hasSpouse", "s"],
      [/^who (?:is|are|was|were) (?:the )?(?:wife|husband|spouse|wives) of (.+)$/, "hasSpouse", "s"],
      [/^who (?:does|did) (.+?) (?:speak|talk) to$/, "speaksTo", "s"],
      [/^who (?:speaks|spoke|talks|talked) to (.+)$/, "speaksTo", "o"],
      [/^who (?:does|did) (.+?) address$/, "speaksTo", "s"],
      [/^who (?:rules|ruled) (.+)$/, "rules", "o"],
      [/^what (?:does|did) (.+?) rule$/, "rules", "s"],
      [/^who (?:taught|instructed|trained) (.+)$/, "instructedBy", "s"],
      [/^who (?:was|is) (.+) (?:taught|trained) by$/, "instructedBy", "s"],
      [/^who (?:is|was|are|were) (.+?)'s (?:teachers?|preceptors?|gurus?)$/, "instructedBy", "s"],
      [/^who (?:is|was|are|were) (?:the )?(?:teachers?|preceptors?|gurus?) of (.+)$/, "instructedBy", "s"],
      [/^who (?:did|does) (.+?) (?:study|train) under$/, "instructedBy", "s"],
      [/^who studied under (.+)$/, "instructedBy", "o"],
      [/^who did (.+) (?:teach|train)$/, "instructedBy", "o"],
      [/^who cursed (.+)$/, "cursedBy", "s"],
      [/^who did (.+) curse$/, "cursedBy", "o"],
      [/^(?:who is|who are|find (?:characters|entities)) (?:like|similar to|closest to|near) (.+)$/, "~", "s"],
      [/^(?:show(?: me)? )?(?:characters|entities) (?:like|similar to) (.+)$/, "~", "s"],
      [/^(?:who|what) (?:is|was) (.+)$/, "", "s"],
      [/^(?:tell me about|show me|explore) (.+)$/, "", "s"],
      [/^(?:show(?: me)? )?(.+?)'s (?:connections|relationships)$/, "connections", "s"]
    ];
    function parse(query) {
      const original = String(query).trim(), s = original.replace(/[\u2018\u2019]/g, "'").replace(/\s+/g, " ").replace(/[?.!]+$/, "").toLowerCase();
      const direct = nameOf(s);
      if (direct !== null) return {kind: "nearest", focus: direct, notice: resolve(s) === null ? "Corpus spelling: " + names[direct] + "." : ""};
      const connection = s.match(/^how (?:is|was|are) (.+?) (related|connected) to (.+)$/);
      if (connection) {
        const focus = nameOf(connection[1]), target = nameOf(connection[3]);
        if (focus !== null && target !== null) return {kind: "path", focus, target, kinFirst: connection[2] === "related"};
      }
      const expression = s.match(/^(.+?)\s*([+-])\s*(.+?)\s*([+-])\s*(.+)$/);
      if (expression) {
        const terms = [expression[1], expression[3], expression[5]].map(nameOf);
        if (terms.every(i => i !== null)) return {kind: "analogy", focus: terms[0], terms,
          signs: [1, expression[2] === "+" ? 1 : -1, expression[4] === "+" ? 1 : -1]};
      }
      const triple = original.replace(/[(),]/g, " ").trim().split(/\s+/);
      if (triple.length === 3 && triple.filter(t => t === "?").length === 1 && triple[1] !== "?") {
        const direction = triple[0] === "?" ? "o" : "s", focus = nameOf(triple[direction === "s" ? 0 : 2]);
        const r = rids.get(triple[1].toLowerCase());
        if (focus !== null && r !== undefined) return {kind: "relation", focus, relation: data.relations[r], direction};
      }
      for (const [re, relation, direction] of patterns) {
        const m = s.match(re); if (!m) continue;
        const focus = nameOf(m[1]); if (focus === null) continue;
        const notice = resolve(m[1]) === null ? "Corpus spelling: " + names[focus] + ". " : "";
        if (!relation || relation === "~") return {kind: "nearest", focus, notice};
        if (relation === "connections") return {kind: "connections", focus, notice};
        if (!rids.has(relation.toLowerCase()) && relation !== 'hasBiologicalParent') return {kind: "unavailable", focus, relation,
          notice: "The current graph has no " + relation + " relation. A recorded fight does not establish standing enmity or alliance."};
        return {kind: "relation", focus, relation, direction, notice: notice +
          (relation === "hasBiologicalParent" && /father|mother|son|daughter/.test(s) ? "Parentage is recorded without a gender classification." : "")};
      }
      return null;
    }
    function execute(query) {
      const result = parse(query);
      if (!result) return {kind: "error", message: "No matching entity or relation question. Try a name, who fought Bhishma, or who taught Arjuna."};
      if (result.kind === "nearest") result.ranking = nearest(vec(result.focus), [result.focus]);
      if (result.kind === "relation") {
        if (names[result.focus] === "Abhimanyu" && result.relation === "slainBy" && result.direction === "s")
          return {...result, kind: "contested", ranking: [], edges: [], notice: "The project leaves Abhimanyu's slayer unasserted because the event is contested. No single slayer is selected."};
        result.edges = relations(result.focus, result.relation, result.direction);
        if (result.relation === 'hasBiologicalParent')
          result.edges.push(...['hasDivineProgenitor','hasFosterParent','hasLegalParent'].flatMap(p=>relations(result.focus,p,result.direction)));
        if (result.relation === 'hasBiologicalParent')
          result.parentage = parentageOf(result.focus, result.direction);
        result.ranking = canPredict(result.relation) ? predict(result.focus, result.relation, result.direction) : [];
        if (result.relation === "instructedBy") {
          const name = names[result.focus], slot = result.direction === "s" ? "student" : "teacher";
          result.passages = teaching.filter(p => p.names.includes(name))
            .filter(p => !p.teacher || p[slot] === name)
            .sort((a,b) => Number(!!b.teacher)-Number(!!a.teacher) || Number(b.cited)-Number(a.cited));
        }
      }
      if (result.kind === "path") {
        const a=parentageOf(result.focus), b=parentageOf(result.target);
        result.sharedParentage=[...new Set(a.map(r=>r.object))].filter(n=>b.some(r=>r.object===n))
          .map(parent=>({parent,sources:[...a,...b].filter(r=>r.object===parent)}));
        const kinPath = result.kinFirst ? pathBetween(result.focus, result.target, true) : null;
        result.path = kinPath ?? pathBetween(result.focus, result.target);
        result.edges = (result.path || []).map(s => s.edge);
        result.notice = "A path through automatically extracted " + (kinPath ? "parentage or marriage" : "graph") +
          " links, not a verified claim of kinship. Arrows retain each relation's direction.";
      }
      if (result.kind === "connections") result.edges = adj[result.focus];
      if (result.kind === "analogy") {
        const v = new Float32Array(D);
        result.terms.forEach((i, j) => {for (let k = 0; k < D; k++) v[k] += result.signs[j] * E[i * D + k];});
        result.ranking = nearest(v, result.terms);
      }
      return result;
    }
    return {data, N, D, names, adj, symmetric, words, resolve, vec, score, nearest, predict, canPredict, relations, parentageOf, displayEdges, parse, execute};
  }
  if (typeof module === "object" && module.exports) module.exports = {create};
  else root.KGCore = {create};
})(typeof window === "object" ? window : globalThis);
