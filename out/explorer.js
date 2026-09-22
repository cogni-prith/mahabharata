(function () {
  "use strict";
  const $ = id => document.getElementById(id), rawK = window.KG;
  if (!rawK || !window.KGCore || !window.ViewerLib) {$("message").textContent = "A local viewer file is missing. Keep index.html beside projection.js, kg-core.js and the vendor folder."; return;}
  const core = KGCore.create(rawK, window.KGTeaching || [], window.KGReview), K = core.data, L = ViewerLib, T = L.THREE;
  const COL = ["#c16f55", "#5886b5", "#8f9e46", "#9468a8", "#cf9b50", "#4c9e8f", "#bd718e", "#74966c"];
  const selectedClusters = new Set(K.cluster), labelNodes = new Map();
  const FAMILIES = [
    {id:'kin', name:'Kinship', color:'#287259', predicates:['hasBiologicalParent','hasFosterParent','hasLegalParent','hasSpouse','bornTo']},
    {id:'conflict', name:'Conflict', color:'#aa514d', predicates:['foughtWith','slainBy','defeatedBy','isEnemyOf']},
    {id:'rule', name:'Rule & tutelage', color:'#3f7997', predicates:['rules','instructedBy','isAlliedWith']},
    {id:'divine', name:'Divine & fated', color:'#91648d', predicates:['cursedBy','hasDivineProgenitor']},
    {id:'telling', name:'The telling', color:'#92772e', predicates:['speaksTo']}
  ];
  const enabledFamilies = new Set(FAMILIES.map(f=>f.id));
  const family = e => FAMILIES.find(f=>f.predicates.includes(K.relations[e[1]])) || FAMILIES[4];
  const measure = document.createElement('canvas').getContext('2d');
  function labelSize(text, size, width) {measure.font=size+'px system-ui';return Math.min(size,size*width/Math.max(1,measure.measureText(text).width));}
  let focus = core.resolve("Karna") ?? 0, result, mode = "similar", detail = "relations", selectedEdge = null;
  let limit = 12, tableRows = [], dimension = 3, sceneApi, graphExpanded = false;
  const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
  const icon = name => '<i data-lucide="' + name + '"></i>';
  const icons = () => L.createIcons({icons: L, attrs: {"stroke-width": 1.6}});
  const entity = i => '<button class="entity-link" data-entity="' + i + '">' + esc(K.entities[i]) + '</button>';
  const badge = (name, cls = name) => '<span class="badge ' + cls + '">' + name + '</span>';
  const other = (e, f = focus) => e[0] === f ? e[2] : e[0];
  const multipleSlayers = f => new Set(core.displayEdges(core.adj[f].filter(i=>K.edges[i][0]===f && K.relations[K.edges[i][1]]==='slainBy'),f).map(i=>K.edges[i][2])).size > 1;
  const deathCaveat = 'Names used in separate source accounts, not verified distinct killers. Read the passages before interpreting the relationship.';
  const label = (e, f = focus) => K.relations[e[1]]==='slainBy' && e[0]===f && multipleSlayers(f) ? 'slaying accounts' : (core.words[K.relations[e[1]]] || [K.relations[e[1]], K.relations[e[1]]])[e[0] === f ? 0 : 1];
  const citations = e => e[5] ? esc(e[3].replace(/_/g, " ")) : badge("uncited");
  const textEdges = () => core.adj[focus];
  function say(s) {$("message").textContent = s || "";}
  function dedupEdges(indexes) {
    const seen = new Set();
    return core.displayEdges(indexes,focus).filter(i => {const e = K.edges[i], key = [e[1], other(e), core.symmetric.has(K.relations[e[1]]) ? 0 : e[0] === focus].join(":");
      if (seen.has(key)) return false; seen.add(key); return true;});
  }
  function sourcePassage(p) {
    return '<blockquote>'+esc(p.text)+'</blockquote><p class="source-citation">'+esc(p.parva.replace(/_/g,' '))+
      (p.cited?' / '+p.chapter+' '+badge('cited'):' '+badge('uncited'))+' &middot; sentence '+p.sentence_id+'</p>';
  }
  function sourceKinship() {
    const groups=new Map(), parents=['bornTo','hasBiologicalParent','hasFosterParent','hasLegalParent','hasDivineProgenitor'];
    for(const direction of ['s','o']) for(const r of core.parentageOf(focus,direction)) {
      const name=r[direction==='s'?'object':'subject'],peer=core.resolve(name),title=direction==='s'?'parent':'child';
      if(textEdges().some(i=>{const e=K.edges[i];return parents.includes(K.relations[e[1]])&&
        K.entities[e[0]]===r.subject&&K.entities[e[2]]===r.object;})) continue;
      const key=title+':'+name;
      if(!groups.has(key))groups.set(key,{name,peer,title,sources:[],family:FAMILIES[0]});
      const g=groups.get(key);
      if(!g.sources.some(p=>p.parva===r.parva&&p.sentence_id===r.sentence_id))g.sources.push(r);
    }
    return [...groups.values()];
  }
  function deathNames() {
    const a=K.review?.death_accounts?.[K.entities[focus]];
    return a?'<details class="death-names"><summary>'+esc(a.appellations.join(', '))+' is a source name of '+esc(a.slayer)+'</summary>'+sourcePassage(a)+'</details>':'';
  }
  function run(q) {
    const next = core.execute(q);
    if (next.kind === "error") {say(next.message); return;}
    result = next; focus = result.focus; limit = 12; selectedEdge = null; detail = "relations"; graphExpanded = false;
    mode = ["relation", "connections", "path", "contested", "unavailable"].includes(result.kind) ? "connections" : "similar";
    if (result.relation && core.canPredict(result.relation)) $("prediction-relation").value = result.relation;
    $("q").value = q; say(result.notice);
    selectedClusters.add(K.cluster[focus]); updateClusterButtons();
    drawAnswer(); drawGraph(); drawProfile(); drawResults(); sceneApi?.update();
    try {history.replaceState(null, "", "#q=" + encodeURIComponent(q));} catch (_) {}
    icons();
  }
  function drawAnswer() {
    const box=$("answer");box.hidden=!['relation','path','contested','unavailable'].includes(result.kind);
    if(box.hidden){box.replaceChildren();return;}
    if(result.kind==='path') {
      let html='<p class="eyebrow">EXTRACTED GRAPH PATH</p><h2>'+esc(K.entities[focus])+' &amp; '+esc(K.entities[result.target])+'</h2>';
      if(result.sharedParentage?.length)html='<p class="eyebrow">SHARED PARENTAGE</p><h2>'+esc(K.entities[focus])+' &amp; '+esc(K.entities[result.target])+'</h2>'+result.sharedParentage.map(p=>
        '<p class="shared-parentage">Both are described as children of <strong>'+esc(p.parent)+'</strong>.</p><details class="parentage-evidence"><summary>Read the parentage passages</summary>'+p.sources.map(sourcePassage).join('')+'</details>').join('')+'<h3 class="secondary-path">Other recorded connection</h3>';
      if(result.path===null) html+='<p>No path is recorded between these names.</p>';
      else if(!result.path.length) html+='<p>Both names identify the same indexed entity.</p>';
      else html+='<div class="path-chain">'+entity(focus)+result.path.map(s=>{const e=K.edges[s.edge],reverse=e[0]!==s.from;
        return '<span class="path-step"><button data-evidence="'+s.edge+'" title="Read this edge\'s evidence">'+(reverse?'&larr; ':'&rarr; ')+esc((core.words[K.relations[e[1]]]||[K.relations[e[1]]])[0])+' '+(reverse?'&larr;':'&rarr;')+'</button>'+entity(s.to)+'</span>';}).join('')+'</div>';
      box.innerHTML=html+'<p class="result-note">'+esc(result.notice)+'</p>';return;
    }
    if(result.kind!=='relation'){box.innerHTML='<p class="eyebrow">NOT ASSERTED</p><p>'+esc(result.notice)+'</p>';return;}
    const accounts=result.relation==='slainBy' && result.direction==='s' && multipleSlayers(focus);
    const relationLabel=accounts?'slaying accounts':(core.words[result.relation]||[result.relation,result.relation])[result.direction==='s'?0:1];
    const answerEdges=dedupEdges(result.edges),peers=[...new Set(answerEdges.map(i=>other(K.edges[i])))];
    let html='<p class="eyebrow">'+(accounts?'SOURCE ACCOUNTS':peers.length?'EXTRACTED ANSWER':'GRAPH COVERAGE GAP')+'</p><h2>'+esc(K.entities[focus])+' / '+esc(relationLabel)+'</h2>';
    if(accounts)html+='<p class="result-note">'+deathCaveat+'</p>';
    html+=peers.length?'<div class="answer-names">'+answerEdges.map(i=>{const e=K.edges[i];return '<span>'+entity(other(e))+' <small>'+esc(label(e))+'</small></span>';}).join('')+'</div><p class="result-note">Read the supporting passages in Evidence.</p>':'<p>No matching typed relation. Available source passages are listed below.</p>';
    if(result.relation==='slainBy'&&result.direction==='s')html+=deathNames();
    const passages=[...(result.passages||[]), ...(result.parentage||[])];
    if(passages.length){
      html+='<div class="source-matches"><h3>From the source text '+badge('source search','vector')+'</h3>';
      const show=p=>'<article class="source-passage">'+(p.teacher?'<p class="source-binding">'+esc(p.teacher)+' &rarr; taught &rarr; '+esc(p.student)+' '+badge(p.in_training_graph?'also in graph':'not in trained graph','vector')+'</p>':p.parentage_unspecified?'<p class="source-binding">'+esc(p.subject)+' &rarr; parentage &rarr; '+esc(p.object)+' '+badge('kind unspecified','vector')+'</p>':'')+
        '<blockquote>'+esc(p.text)+'</blockquote><p class="source-citation">'+esc(p.parva.replace(/_/g,' '))+(p.cited?' / '+p.chapter+' '+badge('cited'): ' '+badge('uncited'))+' &middot; sentence '+p.sentence_id+'</p></article>';
      html+=show(passages[0]);
      if(passages.length>1)html+='<details><summary>'+(passages.length-1)+' more source passages</summary>'+passages.slice(1).map(show).join('')+'</details>';
      html+='<p class="result-note">Names retain their source spelling. Appellations such as Bhargava are not silently merged with another person. A source match is not an embedding prediction.</p></div>';
    }
    box.innerHTML=html;
  }
  function drawGraph() {
    const all=dedupEdges(textEdges()),sources=sourceKinship(),groups=new Map(),container=$("graph-container");
    const links=all.map(i=>{const e=K.edges[i];return {edge:i,peer:other(e),name:K.entities[other(e)],
      family:family(e),title:label(e),cited:!!e[5]};}).concat(sources.map(s=>({...s,cited:s.sources.some(p=>p.cited)})));
    links.forEach(link=>{const f=link.family;if(!enabledFamilies.has(f.id))return;
      const key=f.id+':'+link.title;if(!groups.has(key))groups.set(key,{family:f,title:link.title,edges:[]});groups.get(key).edges.push(link);});
    const columns=[[],[]], heights=[0,0];
    for(const f of FAMILIES) for(const g of groups.values()) if(g.family===f){
      g.edges.sort((a,b)=>a.name.localeCompare(b.name));
      g.total=g.edges.length;if(!graphExpanded)g.edges=g.edges.slice(0,6);
      const side=heights[0]<=heights[1]?0:1;columns[side].push(g);heights[side]+=48+g.edges.reduce((n,e)=>n+(e.cited?26:38),0);
    }
    const W=Math.max(320,container.clientWidth),H=Math.max(container.clientHeight,...heights.map(h=>h+40));
    const cx=W/2,cy=H/2, rootWidth=W<500?104:148, chip=Math.min(150,(W-rootWidth-72)/2);
    let paths='',nodes='';
    columns.forEach((column,side)=>{let y=(H-heights[side])/2;
      const x=side?W-18-chip:18,edgeX=side?x:x+chip,centerX=cx+(side?rootWidth/2:-rootWidth/2);
      column.forEach(g=>{nodes+='<text class="fan-heading" x="'+(side?x+chip:x)+'" y="'+(y+13)+'" text-anchor="'+(side?'end':'start')+'" fill="'+g.family.color+'">'+esc(g.title.toUpperCase())+(g.total>g.edges.length?' '+g.edges.length+'/'+g.total:'')+'</text>';y+=30;
        g.edges.forEach(link=>{const color=g.family.color,e=link.edge===undefined?null:K.edges[link.edge];
          paths+='<path class="fan-edge'+(e?'':' source-parentage')+'" '+(e?'data-edge="'+link.edge+'"':'')+' d="M '+centerX+' '+cy+' C '+(centerX+(edgeX-centerX)*.4)+' '+cy+', '+(centerX+(edgeX-centerX)*.6)+' '+(y+10)+', '+edgeX+' '+(y+10)+'" stroke="'+color+'" fill="none"/>';
          const title=e?K.entities[e[0]]+' '+K.relations[e[1]]+' '+K.entities[e[2]]+'; '+(e[5]?e[3]:'uncited'):link.title+': '+link.name+'; source parentage, kind unspecified';
          nodes+='<g class="fan-node" '+(link.peer===null?'':'data-entity="'+link.peer+'" tabindex="0" role="button" aria-label="Explore '+esc(link.name)+'"')+'><title>'+esc(title)+'</title><rect x="'+x+'" y="'+y+'" width="'+chip+'" height="'+(link.cited?21:33)+'" rx="2" fill="#fff" stroke="#d8dfdc"/><text x="'+(x+8)+'" y="'+(y+14)+'" font-size="'+labelSize(link.name,11,chip-16)+'" fill="#34463d">'+esc(link.name)+'</text></g>';
          if(!link.cited)nodes+='<text x="'+(x+8)+'" y="'+(y+28)+'" font-size="9" fill="#a54e50">uncited</text>';
          y+=link.cited?26:38;});y+=18;});});
    const svg=$("graph");svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.style.width=W+'px';svg.style.height=H+'px';
    svg.innerHTML=paths+nodes+'<g class="fan-focus"><rect x="'+(cx-rootWidth/2)+'" y="'+(cy-25)+'" width="'+rootWidth+'" height="50" rx="3" fill="#233d32"/><text x="'+cx+'" y="'+(cy-3)+'" text-anchor="middle" font-size="'+labelSize(K.entities[focus],16,rootWidth-16)+'" fill="white">'+esc(K.entities[focus])+'</text><text x="'+cx+'" y="'+(cy+15)+'" text-anchor="middle" font-size="9" fill="#c9d7cc">SELECTED ENTITY</text></g>';
    const visible=[...groups.values()].reduce((n,g)=>n+g.edges.length,0);
    $("graph-count").textContent=visible+' / '+links.length+' links';$("graph-count").title=all.length+' extracted links; '+sources.length+' source-parentage links. Appellations identified in the death account are grouped.';
    $("all-connections").firstChild.textContent=graphExpanded?'Show fewer ':'Show all ';
    container.scrollTop=0;container.scrollLeft=(W-container.clientWidth)/2;
  }
  function drawProfile() {
    const name=K.entities[focus],meta=K.meta[name]||{},es=textEdges(),neighbours=new Set(es.map(i=>other(K.edges[i])));
    $("entity-name").textContent=name;
    $("entity-cluster").textContent="Cluster "+(K.cluster[focus]+1);
    $("entity-cluster").style.setProperty("--swatch",COL[K.cluster[focus]]);
    const counts=new Map(); es.forEach(i=>{const p=K.relations[K.edges[i][1]];counts.set(p,(counts.get(p)||0)+1);});
    const common=[...counts].sort((a,b)=>b[1]-a[1]).slice(0,2).map(([p])=>({speaksTo:"dialogue",hasBiologicalParent:"parentage",foughtWith:"combat",slainBy:"slaying",rules:"rulership",hasSpouse:"marriage",instructedBy:"instruction",cursedBy:"curses",defeatedBy:"defeat"}[p]||p));
    const usage=K.review?.notes?.[name];
    $("entity-description").textContent=usage?usage.text:"A recorded name in the corpus. "+(common.length?"Most extracted connections concern "+common.join(" and ")+".":"No retained relationships available after the integrity checks.");
    $("entity-metrics").innerHTML=[[meta.mentions||0,"mentions"],[neighbours.size,"neighbours"],[meta.tells||0,"tellings"]].map(([n,t])=>'<div class="metric"><b>'+n.toLocaleString()+'</b><span>'+t+'</span></div>').join('');
    document.querySelectorAll('[data-detail]').forEach(b=>b.setAttribute("aria-selected",String(b.dataset.detail===detail)));
    if(detail==="relations") {
      const groups=new Map();
      dedupEdges(es).forEach(i=>{const e=K.edges[i],w=label(e);if(!groups.has(w))groups.set(w,[]);groups.get(w).push(i);});
      const row=i=>{const e=K.edges[i];return '<div class="relation-item">'+entity(other(e))+'<button class="evidence-link" data-evidence="'+i+'" aria-label="Read evidence for '+esc(K.entities[other(e)])+'" title="Read source evidence">'+(e[5]?icon('book-open'):badge('uncited'))+'</button></div>';};
      const kin=sourceKinship();
      const kinHTML=kin.length?'<section class="relation-group source-kinship" style="--family:#287259"><h3>KINSHIP <span>source passages</span></h3>'+kin.map(g=>'<div class="parentage-item"><span class="parentage-role">'+esc(g.title)+'</span> '+(g.peer===null?esc(g.name):entity(g.peer))+'<details><summary>'+g.sources.length+' source passage'+(g.sources.length===1?'':'s')+'</summary>'+g.sources.map(sourcePassage).join('')+'</details></div>').join('')+'<p class="result-note">Parentage is stated; biological or adoptive status is unspecified in these passages.</p></section>':'';
      $("detail-panel").innerHTML=kinHTML+([...groups].sort((a,b)=>FAMILIES.indexOf(family(K.edges[a[1][0]]))-FAMILIES.indexOf(family(K.edges[b[1][0]]))).map(([w,list])=>'<section class="relation-group" style="--family:'+family(K.edges[list[0]]).color+'"><h3>'+esc(w.toUpperCase())+'<span>'+list.length+'</span></h3>'+(w==='slaying accounts'?'<p class="result-note">'+deathCaveat+'</p>':'')+list.slice(0,6).map(row).join('')+(list.length>6?'<details class="more-relations"><summary>'+(list.length-6)+' more connections</summary>'+list.slice(6).map(row).join('')+'</details>':'')+(w==='slain by'?deathNames():'')+'</section>').join('')||'<p class="empty">No extracted relations.</p>');
    } else if(detail==="aliases") {
      const aliases=K.alias[name]||[];
      $("detail-panel").innerHTML='<p class="result-note">Appellation candidates remain unresolved. These names have not been merged.</p>'+aliases.map(a=>'<div class="alias-item"><strong>'+esc(a[0])+'</strong> '+badge('unresolved','vector')+'<p>'+a[2]+' apposition sighting'+(a[2]===1?'':'s')+' / '+esc(a[3])+'</p></div>').join('')+(aliases.length?'':'<p class="empty">No appellation candidates recorded for this name.</p>');
    } else {
      const indexes=selectedEdge===null?dedupEdges(es):[selectedEdge];
      let html=selectedEdge===null?'<p class="result-note">Full source passages for the extracted relationships.</p>':'<button class="quiet-button" id="all-evidence">All passages for '+esc(name)+'</button>';
      indexes.forEach(i=>{const e=K.edges[i], key=e.slice(0,3).join(':'), rows=window.KGEvidence?.[key]||[{text:e[6],parva:e[3],chapter:null,cited:!!e[5],confidence:e[7],depth:e[8],preview:true}];
        rows.forEach(r=>{html+='<article class="evidence-block"><div class="evidence-title">'+esc(r.preview?e[3]:r.parva.replace(/_/g,' ')+(r.cited?' / '+r.chapter:''))+(r.cited?badge('cited'):badge('uncited'))+'</div><p class="evidence-triple">'+esc(K.entities[e[0]]+' '+K.relations[e[1]]+' '+K.entities[e[2]])+'</p><blockquote>'+esc(r.text||'No passage text available.')+'</blockquote><p class="evidence-meta">Extraction score '+Number(r.confidence).toFixed(2)+' &middot; telling depth '+r.depth+(r.preview?' &middot; stored preview':' &middot; sentence '+r.sentence_id)+'</p></article>';});});
      $("detail-panel").innerHTML=html;
    }
    if(usage) $("detail-panel").insertAdjacentHTML('afterbegin','<details class="source-usage"><summary>Name usage in the source</summary><blockquote>'+esc(usage.evidence)+'</blockquote><p>'+esc(usage.parva.replace(/_/g,' '))+(usage.cited?' / '+usage.chapter+' '+badge('cited'):' '+badge('uncited'))+'</p></details>');
    const held=K.review?.withheld.filter(r=>!r.parentage_unspecified&&(r.subject===name||r.object===name))||[];
    if(held.length) $("detail-panel").insertAdjacentHTML('beforeend','<details class="withheld-links"><summary>'+held.length+' withheld extraction records</summary>'+held.map(r=>'<article><p>'+badge('withheld','uncited')+' '+esc(r.subject+' / '+r.predicate+' / '+r.object)+'</p><p>'+esc(r.reason)+'</p><blockquote>'+esc(r.text)+'</blockquote><p>'+esc(r.parva.replace(/_/g,' '))+(r.cited&&r.chapter!==null?' / '+r.chapter:' '+badge('uncited'))+'</p></article>').join('')+'</details>');
    icons();
  }
  function edgeRow(i,n,from=focus) {
    const e=K.edges[i];return {entity:other(e,from),kind:"extracted",edge:i,html:'<tr><td>'+n+'</td><td>'+entity(other(e,from))+'</td><td>'+esc(label(e,from))+'</td><td>'+citations(e)+'</td><td><button class="evidence-link" data-evidence="'+i+'">'+icon('book-open')+' '+e[4]+' source'+(e[4]===1?'':'s')+'</button></td></tr>'};
  }
  function drawResults() {
    document.querySelectorAll('.result-tabs button').forEach(b=>b.setAttribute('aria-selected',String(b.id==='tab-'+mode)));
    $("prediction-options").hidden=mode!=="predictions";tableRows=[];
    let heads,title,note="",kind;
    if(mode==="connections") {
      const list=["contested","unavailable"].includes(result.kind)?[]:result.kind==="relation"?dedupEdges(result.edges):dedupEdges(textEdges());
      title=result.kind==="relation"?K.entities[focus]+" / "+(core.words[result.relation]||[result.relation,result.relation])[result.direction==='s'?0:1]:"Connections of "+K.entities[focus];
      kind="TEXTUAL RELATIONS";heads=['#','Entity','Relation','Citation','Evidence'];note=result.notice||"Automatically extracted from the prose; inspect the supporting passage before treating a relation as settled.";
      tableRows=list.map((i,n)=>edgeRow(i,n+1));
      if(result.kind==='path') {
        title=K.entities[focus]+' to '+K.entities[result.target];kind='EXTRACTED GRAPH PATH';
        tableRows=(result.path||[]).map((s,n)=>edgeRow(s.edge,n+1,s.from));
      }
    } else if(mode==="predictions") {
      const relation=$("prediction-relation").value,direction=result.kind==='relation'?result.direction:'s';
      const existing=new Set(core.relations(focus,relation,direction).map(i=>other(K.edges[i])));
      const blocked=K.entities[focus]==='Abhimanyu'&&relation==='slainBy'&&direction==='s';
      const ranking=blocked?[]:core.predict(focus,relation,direction).filter(([j])=>!$("novel-only").checked||!existing.has(j));
      title=direction==='s'?K.entities[focus]+' '+relation+' ?':'? '+relation+' '+K.entities[focus];kind="LINK PREDICTION";heads=['Rank','Candidate','RotatE score','Status'];
      note=blocked?'A single slayer is not selected for this contested event.':'All '+core.N+' candidates scored in learned vector space. Higher scores rank first. Predictions are uncited hypotheses.'+(K.review&&!K.review.stats.model_graph_matches_review?' The bundled model graph differs from this display; predictions may reproduce withdrawn links.':'');
      tableRows=ranking.map(([j,s],n)=>({entity:j,score:s,kind:'predicted',cited:false,html:'<tr><td>'+(n+1)+'</td><td>'+entity(j)+'</td><td class="score">'+s.toFixed(3)+'</td><td>'+badge('predicted')+badge('uncited')+(existing.has(j)?' '+badge('in graph','vector'):'')+'</td></tr>'}));
    } else {
      const analogy=result.kind==='analogy',ranking=analogy?result.ranking:core.nearest(core.vec(focus),[focus]);
      const ns=new Set(textEdges().map(i=>other(K.edges[i])));
      title=analogy?$("q").value:"Similar to "+K.entities[focus];kind=analogy?"VECTOR ARITHMETIC":"VECTOR SIMILARITY";heads=['Rank','Entity','Cosine similarity','Connection'];
      note=(analogy?'Analogy candidates are predicted and uncited; this model has not established the proposed analogy.':'Ranked by cosine similarity in '+core.D/2+' complex dimensions. UMAP coordinates are used only for the projection.')+(K.review&&!K.review.stats.model_graph_matches_review?' The bundled model graph differs from this display.':'');
      tableRows=ranking.map(([j,s],n)=>({entity:j,score:s,kind:analogy?'predicted':'similarity',cited:false,html:'<tr><td>'+(n+1)+'</td><td>'+entity(j)+'</td><td class="score">'+s.toFixed(4)+'<div class="match-bar"><i style="width:'+Math.max(0,Math.min(100,s*100))+'%"></i></div></td><td>'+(analogy?badge('predicted')+badge('uncited'):ns.has(j)?badge('also in graph'):badge('vector only','vector'))+'</td></tr>'}));
    }
    $("result-title").textContent=title;$("result-kind").textContent=kind;$("result-note").textContent=note;
    $("result-head").innerHTML='<tr>'+heads.map(h=>'<th scope="col">'+h+'</th>').join('')+'</tr>';
    $("results").innerHTML=tableRows.slice(0,limit).map(r=>r.html).join('')||'<tr><td colspan="5" class="empty">'+
      (mode==='connections'&&(result.passages?.length||result.parentage?.length)?'No typed graph links. Source passages are shown above.':'No supported results for this query.')+'</td></tr>';
    $("results-count").textContent=Math.min(limit,tableRows.length)+' of '+tableRows.length+' results';
    $("more-results").hidden=limit>=tableRows.length;icons();
  }
  function download(blob,name) {const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  function updateClusterButtons() {document.querySelectorAll('[data-cluster]').forEach(b=>b.setAttribute('aria-pressed',String(selectedClusters.has(Number(b.dataset.cluster)))));}

  function initScene() {
    const stage=$("stage");let renderer;
    try {renderer=new T.WebGLRenderer({antialias:true,alpha:false,preserveDrawingBuffer:true});}
    catch(e) {stage.insertAdjacentHTML('beforeend','<p class="scene-error">WebGL is unavailable. Displaying the stored 2D UMAP projection.</p>');return fallbackScene(stage);}
    renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0xf0f4ed);stage.prepend(renderer.domElement);
    const scene=new T.Scene(),camera=new T.PerspectiveCamera(40,1,.01,100);
    const controls=new L.OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.1;controls.autoRotateSpeed=.6;controls.minDistance=.12;controls.maxDistance=15;
    const textureCanvas=document.createElement('canvas');textureCanvas.width=64;textureCanvas.height=64;
    const ctx=textureCanvas.getContext('2d');ctx.fillStyle='white';ctx.beginPath();ctx.arc(32,32,26,0,Math.PI*2);ctx.fill();
    const texture=new T.CanvasTexture(textureCanvas),geometry=new T.BufferGeometry(),colorArray=new Float32Array(core.N*3);
    geometry.setAttribute('position',new T.BufferAttribute(new Float32Array(core.N*3),3));geometry.setAttribute('color',new T.BufferAttribute(colorArray,3));
    const dots=new T.Points(geometry,new T.PointsMaterial({size:6,sizeAttenuation:false,vertexColors:true,map:texture,transparent:true,alphaTest:.15}));scene.add(dots);
    const focusGeometry=new T.BufferGeometry();focusGeometry.setAttribute('position',new T.BufferAttribute(new Float32Array(3),3));
    const focusDot=new T.Points(focusGeometry,new T.PointsMaterial({size:13,sizeAttenuation:false,color:0x235c40,map:texture,transparent:true,alphaTest:.15,depthTest:false}));focusDot.renderOrder=2;scene.add(focusDot);
    const lines=new T.LineSegments(new T.BufferGeometry(),new T.LineBasicMaterial({color:0x8aa17a,transparent:true,opacity:.35}));scene.add(lines);
    const grid=new T.GridHelper(1.5,6,0xd5dece,0xe2e8dc);grid.position.y=-.56;scene.add(grid);
    const flat=K.xy.map(p=>[p[0],p[1],.5]);
    const points=()=>dimension===3?K.xyz:flat;
    const pos=i=>new T.Vector3(...points()[i].map(x=>x-.5));
    function fit() {
      if(!stage.clientWidth || !stage.clientHeight)return;
      controls.target.set(0,0,0);
      const direction=dimension===3?new T.Vector3(.3,.18,1).normalize():new T.Vector3(0,0,1);
      let low=.65,high=12;
      for(let iteration=0;iteration<16;iteration++) {
        const distance=(low+high)/2;camera.position.copy(direction).multiplyScalar(distance);camera.lookAt(controls.target);camera.updateMatrixWorld();
        const fits=points().every(p=>{const v=new T.Vector3(...p.map(x=>x-.5)).project(camera);return v.z<1&&v.z>-1&&Math.abs(v.x)<.82&&Math.abs(v.y)<.74;});
        if(fits)high=distance;else low=distance;
      }
      camera.position.copy(direction).multiplyScalar(high);controls.update();
    }
    function resize() {const w=stage.clientWidth,h=stage.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}
    function update() {
      const values=geometry.attributes.position.array;
      points().forEach((p,i)=>{const shown=selectedClusters.has(K.cluster[i]);for(let k=0;k<3;k++)values[i*3+k]=shown?p[k]-.5:1000;
        new T.Color(COL[K.cluster[i]]).toArray(colorArray,i*3);});
      geometry.attributes.position.needsUpdate=true;geometry.attributes.color.needsUpdate=true;geometry.computeBoundingSphere();
      pos(focus).toArray(focusGeometry.attributes.position.array);focusGeometry.attributes.position.needsUpdate=true;
      const coords=[];textEdges().forEach(i=>{const e=K.edges[i];if(selectedClusters.has(K.cluster[e[0]])&&selectedClusters.has(K.cluster[e[2]]))coords.push(...pos(e[0]).toArray(),...pos(e[2]).toArray());});
      lines.geometry.dispose();lines.geometry=new T.BufferGeometry();lines.geometry.setAttribute('position',new T.Float32BufferAttribute(coords,3));lines.visible=$("show-edges").checked;
      grid.visible=dimension===3;controls.enableRotate=dimension===3;focusDot.visible=selectedClusters.has(K.cluster[focus]);
      $("point-count").textContent=K.cluster.filter(c=>selectedClusters.has(c)).length+' entities';
      const chosen=[focus,...core.nearest(core.vec(focus),[focus]).slice(0,6).map(r=>r[0])];
      $("point-labels").replaceChildren();labelNodes.clear();chosen.forEach(i=>{const n=document.createElement('span');n.className='point-label'+(i===focus?' focus-label':'');n.textContent=K.entities[i];$("point-labels").append(n);labelNodes.set(i,n);});
    }
    function labelPositions() {
      const occupied=[],w=stage.clientWidth,h=stage.clientHeight;
      labelNodes.forEach((node,i)=>{const v=pos(i).project(camera),x=(v.x+1)/2*w,y=(1-v.y)/2*h;
        const box={x:x+8,y:y-10,w:node.offsetWidth||90,h:22};
        const collision=occupied.some(b=>box.x<b.x+b.w&&box.x+box.w>b.x&&box.y<b.y+b.h&&box.y+box.h>b.y);
        const show=$("show-labels").checked&&selectedClusters.has(K.cluster[i])&&v.z<1&&v.z>-1&&x>4&&x+box.w<w-40&&y>39&&y<h-40&&!collision;
        node.style.display=show?'block':'none';if(show){node.style.left=x+'px';node.style.top=y+'px';occupied.push(box);}});
    }
    // Pixel hit testing uses projected Three.js coordinates, so its radius stays
    // usable when orbiting or zooming a dense cloud.
    function hit(event) {const rect=renderer.domElement.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;
      let found=null,best=100;K.entities.forEach((_,i)=>{if(!selectedClusters.has(K.cluster[i]))return;const v=pos(i).project(camera);if(v.z>1||v.z<-1)return;
        const d=((v.x+1)/2*rect.width-x)**2+((1-v.y)/2*rect.height-y)**2;if(d<best){best=d;found=i;}});return found;}
    let down=null;
    renderer.domElement.addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY];});
    renderer.domElement.addEventListener('pointerup',e=>{if(down&&Math.hypot(e.clientX-down[0],e.clientY-down[1])<5){const i=hit(e);if(i!==null)run(K.entities[i]);}down=null;});
    renderer.domElement.addEventListener('pointermove',e=>{if(e.buttons){$("point-tooltip").hidden=true;return;}const i=hit(e),tip=$("point-tooltip");tip.hidden=i===null;
      if(i!==null){tip.textContent=K.entities[i]+' / cluster '+(K.cluster[i]+1);const rect=stage.getBoundingClientRect();tip.style.left=Math.min(e.clientX-rect.left+12,rect.width-210)+'px';tip.style.top=Math.max(0,e.clientY-rect.top-30)+'px';}});
    renderer.domElement.addEventListener('pointerleave',()=>{$("point-tooltip").hidden=true;});
    renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();say('The 3D graphics context was lost. Reload to restore the scene.');});
    renderer.domElement.addEventListener('webglcontextrestored',()=>{renderer.setClearColor(0xf0f4ed);update();say('');});
    new ResizeObserver(()=>{if(stage.clientWidth && stage.clientHeight){resize();fit();}}).observe(stage);resize();fit();update();
    renderer.setAnimationLoop(()=>{if(document.hidden)return;controls.update();labelPositions();renderer.render(scene,camera);});
    return {update,fit,dimension:()=>{controls.autoRotate=false;$("rotate").setAttribute('aria-pressed','false');update();fit();},
      centre:()=>{const next=pos(focus),delta=next.clone().sub(controls.target);camera.position.add(delta);controls.target.copy(next);controls.update();},
      rotate:()=>{controls.autoRotate=!controls.autoRotate;$("rotate").setAttribute('aria-pressed',String(controls.autoRotate));},
      zoom:factor=>{camera.position.sub(controls.target).multiplyScalar(factor).add(controls.target);controls.update();},
      snapshot:()=>{renderer.render(scene,camera);renderer.domElement.toBlob(b=>{if(b)download(b,'mahabharata-umap-'+dimension+'d.png');});},
      pointScreen:i=>{const p=pos(i).project(camera);return {x:(p.x+1)/2*stage.clientWidth,y:(1-p.y)/2*stage.clientHeight};},renderer};
  }
  function fallbackScene(stage) {
    const canvas=document.createElement('canvas');stage.prepend(canvas);const ctx=canvas.getContext('2d');dimension=2;$("projection-label").textContent='UMAP / 2D fallback';
    ["dim3","rotate","zoom-in","zoom-out","focus-camera"].forEach(id=>$(id).disabled=true);
    function update(){canvas.width=stage.clientWidth*devicePixelRatio;canvas.height=stage.clientHeight*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);ctx.fillStyle='#f0f4ed';ctx.fillRect(0,0,stage.clientWidth,stage.clientHeight);
      K.xy.forEach((p,i)=>{if(!selectedClusters.has(K.cluster[i]))return;ctx.fillStyle=i===focus?'#235c40':COL[K.cluster[i]];ctx.beginPath();ctx.arc(40+p[0]*(stage.clientWidth-90),45+(1-p[1])*(stage.clientHeight-90),i===focus?6:3,0,Math.PI*2);ctx.fill();});}
    new ResizeObserver(update).observe(stage);update();return {update,fit:update,dimension:update,centre:update,rotate(){},zoom(){},snapshot:()=>canvas.toBlob(b=>download(b,'mahabharata-umap-2d.png'))};
  }

  $("stats").innerHTML=[[K.entities.length,'entities'],[K.edges.length,'text relations'],[K.relations.length,'predicates']].map(([n,s])=>'<span><b>'+n.toLocaleString()+'</b>'+s+'</span>').join('');
  $("run-info").textContent='RotatE / '+core.D/2+' complex dimensions / UMAP seed '+K.umap_seed;
  $("prediction-relation").innerHTML=K.relations.filter(core.canPredict).map(p=>'<option>'+esc(p)+'</option>').join('');$("prediction-relation").value='foughtWith';
  $("clusters").innerHTML=COL.map((c,i)=>{
    const members=K.entities.map((name,j)=>({name,j})).filter(e=>K.cluster[e.j]===i)
      .sort((a,b)=>(K.meta[b.name]?.mentions||0)-(K.meta[a.name]?.mentions||0));
    const title='Cluster '+(i+1)+' / '+members.length+' entities / '+members.slice(0,4).map(e=>e.name).join(', ')+'. K-means grouping of learned vectors; colours do not denote factions or confidence.';
    return '<button data-cluster="'+i+'" style="--swatch:'+c+'" aria-pressed="true" aria-label="'+esc(title)+'" title="'+esc(title)+'"></button>';
  }).join('');
  $("families").innerHTML=FAMILIES.map(f=>'<button data-family="'+f.id+'" aria-pressed="true" style="--family:'+f.color+'"><span></span>'+f.name+'</button>').join('');
  $("model-details").innerHTML='<dl>'+[['Model',K.model],['Gate','Below threshold'],['Asserted triples',K.n_asserted.toLocaleString()],['After symmetric dedup',K.n_trained.toLocaleString()],['Training seed',K.train_seed],['UMAP seed',K.umap_seed],['Dimensions',core.D/2+' complex'],['Recension','BORI critical edition']].map(([a,b])=>'<dt>'+esc(a)+'</dt><dd>'+esc(b)+'</dd>').join('')+'</dl>';
  if(K.review){
    const state=K.review.stats.model_graph_matches_review?'Matches bundled model graph.':'Differs from bundled model graph.';
    document.querySelector('.view-toolbar').insertAdjacentHTML('beforebegin','<div class="review-status">Rechecked graph: '+K.edges.length.toLocaleString()+' links &middot; '+K.review.stats.withheld_instances+' extraction records withheld. <strong>'+state+'</strong></div>');
    $("model-details").insertAdjacentHTML('beforeend','<p>'+state+' This is a rule-based recheck, not independent factual validation. Withheld records remain available with their evidence. Predictions are still unverified.</p>');
  }
  $("query-form").onsubmit=e=>{e.preventDefault();run($("q").value);};
  document.addEventListener('click',e=>{
    const filter=e.target.closest('[data-family]');if(filter){const id=filter.dataset.family;enabledFamilies.has(id)?enabledFamilies.delete(id):enabledFamilies.add(id);filter.setAttribute('aria-pressed',String(enabledFamilies.has(id)));drawGraph();return;}
    const layout=e.target.closest('[data-layout]');if(layout?.tagName==='BUTTON'){
      document.querySelector('.views').dataset.layout=layout.dataset.layout;
      document.querySelectorAll('button[data-layout]').forEach(b=>b.setAttribute('aria-pressed',String(b===layout)));
      requestAnimationFrame(()=>{drawGraph();sceneApi.fit();});return;}
    const query=e.target.closest('[data-query]');if(query){run(query.dataset.query);return;}
    const ent=e.target.closest('[data-entity]');if(ent){run(K.entities[Number(ent.dataset.entity)]);return;}
    const ev=e.target.closest('[data-evidence]');if(ev){selectedEdge=Number(ev.dataset.evidence);detail='evidence';drawProfile();$("detail-panel").scrollIntoView({block:'nearest',behavior:'smooth'});return;}
    const tab=e.target.closest('[data-detail]');if(tab){detail=tab.dataset.detail;selectedEdge=null;drawProfile();return;}
    const cluster=e.target.closest('[data-cluster]');if(cluster){const c=Number(cluster.dataset.cluster);selectedClusters.has(c)?selectedClusters.delete(c):selectedClusters.add(c);updateClusterButtons();sceneApi.update();}
    if(e.target.closest('#all-evidence')){selectedEdge=null;drawProfile();}
  });
  document.addEventListener('keydown',e=>{const n=e.target.closest('[data-entity]');if(n&&(e.key==='Enter'||e.key===' ')){e.preventDefault();run(K.entities[Number(n.dataset.entity)]);}});
  ['similar','connections','predictions'].forEach(m=>{$('tab-'+m).onclick=()=>{mode=m;limit=12;drawResults();};});
  $("all-connections").onclick=()=>{graphExpanded=!graphExpanded;drawGraph();};
  $("more-results").onclick=()=>{limit+=25;drawResults();};
  $("prediction-relation").onchange=$("novel-only").onchange=()=>{limit=12;drawResults();};
  $("all-clusters").onclick=()=>{K.cluster.forEach(c=>selectedClusters.add(c));updateClusterButtons();sceneApi.update();};
  $("show-labels").onchange=$("show-edges").onchange=()=>sceneApi.update();
  [2,3].forEach(d=>{$('dim'+d).onclick=()=>{dimension=d;$("projection-label").textContent='UMAP / '+d+'D';$('dim2').setAttribute('aria-pressed',String(d===2));$('dim3').setAttribute('aria-pressed',String(d===3));$('rotate').disabled=d===2;sceneApi.dimension();};});
  $("reset-camera").onclick=()=>sceneApi.fit();$("focus-camera").onclick=()=>sceneApi.centre();$("rotate").onclick=()=>sceneApi.rotate();
  $("zoom-in").onclick=()=>sceneApi.zoom(.8);$("zoom-out").onclick=()=>sceneApi.zoom(1.25);$("scene-download").onclick=()=>sceneApi.snapshot();
  $("model-info").onclick=()=>$("model-dialog").showModal();$("close-model").onclick=()=>$("model-dialog").close();
  $("export-results").onclick=()=>download(new Blob([JSON.stringify({query:$("q").value,mode,below_gate:K.below_gate,train_seed:K.train_seed,
    source_matches:[...(result.passages||[]),...(result.parentage||[])],
    results:tableRows.map(({html,...r})=>({...r,entity:K.entities[r.entity],...(r.edge!==undefined?{triple:K.edges[r.edge].slice(0,3).map((n,i)=>i===1?K.relations[n]:K.entities[n]),cited:!!K.edges[r.edge][5],citation:K.edges[r.edge][3]}:{})}))},null,2)],{type:'application/json'}),'mahabharata-query.json');
  let initial='Karna';try{const p=new URLSearchParams(location.hash.slice(1));if(p.get('q'))initial=p.get('q');}catch(_){}
  run('Karna');sceneApi=initScene();if(initial!=='Karna')run(initial);icons();
  new ResizeObserver(()=>{if($("graph-container").clientWidth)drawGraph();}).observe($("graph-container"));
  window.MBHExplorer={run,core,get state(){return {focus,result,mode,dimension};},get scene(){return sceneApi;}};
})();
