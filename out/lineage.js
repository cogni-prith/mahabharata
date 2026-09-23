/* Lineage mode. A family tree, a generational graph and a question box over
   the curated lineage table in annotations/lineage/.

   That table is hand-authored and uncited. It is shown beside the text, never
   merged into it: an edge the asserted graph also states carries its citation,
   every other edge is drawn dashed and labelled curated. Answers are computed
   here from the table, so nothing is generated and nothing leaves the page. */
(function () {
  'use strict';
  const D=window.LINEAGE, $=id=>document.getElementById(id), L=window.ViewerLib;
  if (!D || !$('lineage-workspace')) return;
  const esc=x=>String(x==null?'':x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const icons=()=>L&&L.createIcons({icons:L,attrs:{'stroke-width':1.6}});
  const badge=(s,c='')=>'<span class="chat-badge '+c+'">'+esc(s)+'</span>';
  const push=(m,k,v)=>{(m.get(k)||m.set(k,[]).get(k)).push(v);};
  const examples=['Who are the parents of Karna?','Who are the brothers of Arjuna?','How is Karna related to Duryodhana?','Descendants of Abhimanyu','Who is Chitrasena?'];

  const byId=new Map(D.nodes.map(n=>[n.id,n])), parents=new Map(), children=new Map(), edge=new Map();
  D.edges.forEach(e=>{push(parents,e.child,e.parent); push(children,e.parent,e.child); edge.set(e.parent+'\0'+e.child,e);});
  const textParents=new Map(), textChildren=new Map();
  D.graph_only.forEach(f=>{push(textParents,f.child,f); push(textChildren,f.parent,f);});
  const up=n=>parents.get(n)||[], down=n=>children.get(n)||[];
  const color=n=>D.colors[byId.get(n)?.clan]||D.colors.other;

  let active=false, focus=D.initial.includes('Karna')?'Karna':D.initial[0], view='tree', otherQuery='', latest=null;
  let box=null, drag=null;

  /* ---------- provenance ---------- */
  function cites(e) {
    if (!e.text.length) return badge('curated, uncited','uncited');
    return e.text.flatMap(f=>f.citations.map(c=>c==='uncited'?badge('in the text, no chapter','uncited'):badge(c.replace(/_/g,' '),'cited')))
      .concat(e.text.map(f=>badge(f.predicate+(f.support>1?', stated '+f.support+'x':''),'support'))).join('');
  }
  const textOnly=f=>badge('in the text, not in the table','unresolved')+badge(f.predicate,'support')+
    f.citations.map(c=>c==='uncited'?badge('in the text, no chapter','uncited'):badge(c.replace(/_/g,' '),'cited')).join('');

  /* ---------- names ---------- */
  const base=s=>s.toLowerCase().replace(/\s*\([^)]*\)/g,'').trim();
  const inner=s=>((/\(([^)]*)\)/.exec(s)||[])[1]||'').toLowerCase();
  function lev(a,b) {
    let row=[...Array(b.length+1).keys()];
    for (let i=1;i<=a.length;i++) {
      let prev=row[0]; row[0]=i;
      for (let j=1;j<=b.length;j++) { const t=row[j]; row[j]=Math.min(row[j]+1,row[j-1]+1,prev+(a[i-1]===b[j-1]?0:1)); prev=t; }
    }
    return row[b.length];
  }
  /* Never a silent choice. One exact label wins; several qualified labels
     ("Chitrasena (Kaurava)", "Chitrasena (son of Karna)") come back as a set. */
  function resolve(term) {
    const t=term.toLowerCase();
    const exact=D.nodes.find(n=>n.id.toLowerCase()===t);
    const related=D.nodes.filter(n=>n.id.toLowerCase()!==t&&(base(n.id)===t||inner(n.id)===t)).map(n=>n.id);
    if (exact) return {id:exact.id,via:'exact',others:related};
    // A lone qualified match is still offered, not taken: "Bhimasena" reaching
    // "Bhimasena (ancestor)" would silently answer about the wrong man.
    if (related.length) return {candidates:related};
    if (t.length<5) return null;
    const scored=D.nodes.map(n=>[n.id,lev(t,base(n.id))]).filter(([,d])=>d<=Math.floor(t.length/4)).sort((a,b)=>a[1]-b[1]);
    if (!scored.length) return null;
    const best=scored.filter(s=>s[1]===scored[0][1]).map(s=>s[0]);
    return best.length===1?{id:best[0],via:'spelling',surface:term,others:[]}:{candidates:best};
  }
  const STOP=new Set('who whom whose is are was were the a an of to and in how what which tell me about related relation relationship between with his her their its did does do show list all give father fathers mother mothers parent parents son sons daughter daughters child children kids brother brothers sister sisters sibling siblings grandfather grandmother grandparents grandson granddaughter grandchildren ancestor ancestors descendant descendants lineage line family tree wife wives husband spouse married born from same'.split(' '));
  function names(q) {
    const found=[];
    for (let w of q.match(/[A-Za-z]+(?:'s)?/g)||[]) {
      w=w.replace(/'s$/,'');
      if (STOP.has(w.toLowerCase())) continue;
      const r=resolve(w)||(w.endsWith('s')?resolve(w.slice(0,-1)):null);
      if (r&&!found.some(f=>f.id&&f.id===r.id)) found.push({...r,surface:w});
    }
    return found;
  }

  /* ---------- kinship ---------- */
  function walk(start,next) {
    const dist=new Map([[start,0]]), queue=[start];
    while (queue.length) { const n=queue.shift(); for (const m of next(n)) if (!dist.has(m)) { dist.set(m,dist.get(n)+1); queue.push(m); } }
    dist.delete(start); return dist;
  }
  function siblings(x) {
    const out=new Map();
    up(x).forEach(p=>down(p).forEach(s=>{ if (s!==x) push(out,s,p); }));
    return out;
  }
  // Keyed on (generations from x up, generations from y up); reads as "y is ... of x".
  const KIN={'1,0':'a parent of','0,1':'a child of','2,0':'a grandparent of','0,2':'a grandchild of',
    '1,1':'a sibling or half-sibling of','2,1':'an uncle or aunt of','1,2':'a nephew or niece of','2,2':'a first cousin of'};
  function relate(x,y) {
    const ax=walk(x,up), ay=walk(y,up); ax.set(x,0); ay.set(y,0);
    let best=null;
    for (const [a,u] of ax) if (ay.has(a)) { const d=ay.get(a); if (!best||u+d<best.u+best.d) best={a,u,d}; }
    if (best) {
      const {u,d}=best;
      const phrase=KIN[u+','+d]||(u===0?'a descendant, '+d+' generations down, of':d===0?'an ancestor, '+u+' generations up, of':null);
      return {best, phrase, chain:[...climb(x,best.a).reverse(),...climb(y,best.a)]};
    }
    // No shared ancestor: connected, if at all, through a co-parent.
    const prev=new Map([[x,null]]), queue=[x];
    while (queue.length&&!prev.has(y)) { const n=queue.shift(); for (const m of [...up(n),...down(n)]) if (!prev.has(m)) { prev.set(m,n); queue.push(m); } }
    if (!prev.has(y)) return null;
    const path=[]; for (let n=y;prev.get(n)!=null;n=prev.get(n)) path.unshift(edge.has(prev.get(n)+'\0'+n)?[prev.get(n),n]:[n,prev.get(n)]);
    return {best:null,chain:path};
  }
  function climb(x,a) { // [parent, child] edges from ancestor a down to x
    const prev=new Map([[x,null]]), queue=[x];
    while (queue.length&&!prev.has(a)) { const n=queue.shift(); for (const p of up(n)) if (!prev.has(p)) { prev.set(p,n); queue.push(p); } }
    const out=[]; for (let n=a;prev.get(n)!=null;n=prev.get(n)) out.push([n,prev.get(n)]);
    return out;
  }

  /* ---------- questions ---------- */
  const INTENTS=[
    ['grandparents',/grand(father|mother|parent)/],['grandchildren',/grand(son|daughter|child)/],
    ['parents',/whose (son|daughter|child)|\bborn (to|of)\b/],
    ['ancestors',/ancestor|forefather|descended from/],['descendants',/descendant|progeny|dynasty|lineage|family line/],
    ['siblings',/brother|sister|sibling/],['spouse',/wife|wives|husband|spouse|married/],
    ['children',/\b(sons?|daughters?|child|children|kids?)\b/],['parents',/father|mother|parent/]];
  const person=n=>!byId.has(n)?'<b class="lineage-name">'+esc(n)+'</b>':'<button class="lineage-name" data-lineage-node="'+esc(n)+'"><i style="background:'+esc(color(n))+'"></i>'+esc(n)+'</button>';
  const row=(n,meta,note='')=>'<li>'+person(n)+'<span class="chat-meta">'+meta+'</span>'+(note?'<small>'+esc(note)+'</small>':'')+'</li>';
  const list=(items,empty)=>items.length?'<ul class="lineage-list">'+items.join('')+'</ul>':'<p class="chat-empty">'+esc(empty)+'</p>';
  const SEX='The table does not record which parent is the father or the mother, nor whether a parent is biological, legal or divine. Every listed parent is shown.';

  function answer(q,id) {
    const found=id?[{id,via:'exact',others:[],surface:id}]:names(q), people=found.filter(f=>f.id), sets=found.filter(f=>f.candidates);
    const intent=(INTENTS.find(([,re])=>re.test(q.toLowerCase()))||['summary'])[0];
    let title='', body='', notes=[];
    sets.forEach(s=>notes.push('"'+s.surface+'" matches several people in the table and none is chosen for you: '+s.candidates.join(', ')+'. Pick one.'));
    people.forEach(p=>{
      if (p.via==='spelling') notes.push('Read "'+p.surface+'" as '+p.id+' by spelling.');
      if (p.others.length) notes.push(p.id+' is shown. The table also has '+p.others.join(', ')+' as separate people, and nothing here says they are the same person.');
    });
    if (!people.length) {
      body=sets.length?list(sets.flatMap(s=>s.candidates.map(c=>row(c,badge('candidate','unresolved')))),'')
        :'<p class="chat-empty">No name in this question is in the lineage table. Try one of the examples above.</p>';
      return {title:sets.length?'Which one?':'No match',body,notes};
    }
    const x=people[0].id; focus=x;
    if (people.length>1) {
      const y=people[1].id, r=relate(x,y);
      title='How '+x+' and '+y+' are related';
      if (!r) body='<p class="chat-empty">The table connects '+esc(x)+' and '+esc(y)+' by no chain of parents and children.</p>';
      else {
        const b=r.best, through=b&&b.u&&b.d?', through '+esc(b.a):'';
        body=(b?'<p class="lineage-verdict">'+(r.phrase?'<b>'+esc(y)+'</b> is '+esc(r.phrase)+' <b>'+esc(x)+'</b>'+through+'.':
            '<b>'+esc(x)+'</b> and <b>'+esc(y)+'</b> share an ancestor, '+esc(b.a)+', '+b.u+' generations above '+esc(x)+' and '+b.d+' above '+esc(y)+'.')+'</p>':
          '<p class="lineage-verdict">No common ancestor in the table. They connect only through a shared child, as co-parents.</p>')+
          '<ol class="lineage-chain">'+r.chain.map(([p,c])=>'<li>'+person(p)+'<small>parent of</small>'+person(c)+
          '<span class="chat-meta">'+cites(edge.get(p+'\0'+c))+'</span></li>').join('')+'</ol>';
        notes.push('Every step below is one row of the curated table. Only steps with a chapter badge are also stated in the text.');
        if (b&&b.u===1&&b.d===1) notes.push('A sibling here shares at least one listed parent. Whether that parent is biological, legal or divine is not recorded.');
      }
      return {title,body,notes,focus:x};
    }
    const parentRows=n=>up(n).map(p=>row(p,cites(edge.get(p+'\0'+n)))).concat((textParents.get(n)||[]).map(f=>row(f.parent,textOnly(f))));
    const childRows=n=>down(n).map(c=>row(c,cites(edge.get(n+'\0'+c)))).concat((textChildren.get(n)||[]).map(f=>row(f.child,textOnly(f))));
    const generations=(dist,word)=>{
      const g=new Map(); dist.forEach((d,n)=>push(g,d,n));
      return [...g].sort((a,b)=>a[0]-b[0]).map(([d,ns])=>'<h3 class="chat-section">'+d+' generation'+(d>1?'s':'')+' '+word+'</h3>'+list(ns.map(n=>row(n,'')),'')).join('')||'<p class="chat-empty">None in the table.</p>';
    };
    if (intent==='parents') { title='Parents of '+x; body=list(parentRows(x),'The table lists no parent of '+x+'.'); notes.push(SEX); }
    else if (intent==='children') { title='Children of '+x; body=list(childRows(x),'The table lists no child of '+x+'.'); }
    else if (intent==='siblings') {
      title='Siblings of '+x;
      body=list([...siblings(x)].map(([s,ps])=>row(s,badge('shares '+ps.join(', '),'support'))),'The table lists no one who shares a parent with '+x+'.');
      notes.push('A sibling here shares at least one listed parent. Brother and sister are not told apart, because the table records no sex.');
    } else if (intent==='grandparents') { title='Grandparents of '+x; body=list([...walk(x,up)].filter(([,d])=>d===2).map(([n])=>row(n,'')),'None in the table.'); }
    else if (intent==='grandchildren') { title='Grandchildren of '+x; body=list([...walk(x,down)].filter(([,d])=>d===2).map(([n])=>row(n,'')),'None in the table.'); }
    else if (intent==='ancestors') { title='Ancestors of '+x; body=generations(walk(x,up),'up'); }
    else if (intent==='descendants') { title='Descendants of '+x; body=generations(walk(x,down),'down'); }
    else if (intent==='spouse') {
      title='Co-parents of '+x;
      const co=new Map(); down(x).forEach(c=>up(c).forEach(p=>{ if (p!==x) push(co,p,c); }));
      body=list([...co].map(([p,cs])=>row(p,badge('shared child: '+cs.join(', '),'support'))),'The table lists no other parent for any child of '+x+'.');
      notes.push('The table records no marriages. These are people listed as the other parent of a shared child, which includes gods.');
    } else {
      title=x;
      body='<h3 class="chat-section">Parents</h3>'+list(parentRows(x),'None listed.')+
        '<h3 class="chat-section">Children</h3>'+list(childRows(x),'None listed.')+
        '<h3 class="chat-section">Shares a parent with</h3>'+list([...siblings(x)].map(([s,ps])=>row(s,badge('via '+ps.join(', '),'support'))),'No one.');
      notes.push(SEX);
    }
    return {title,body,notes,focus:x};
  }

  function ask(q,id) {
    q=String(q||'').trim(); if (!q) return;
    latest={query:q,...answer(q,id)};
    $('lineage-answer').innerHTML=(id?'':'<p class="eyebrow">'+esc(q)+'</p>')+'<h2>'+esc(latest.title)+'</h2>'+
      latest.notes.map(n=>'<p class="chat-note lineage-caveat">'+esc(n)+'</p>').join('')+latest.body;
    draw(); icons();
  }

  /* ---------- drawing ---------- */
  const W=132, H=34, GAP=16, ROW=86;
  function nodeSvg(n,x,y,cls='') {
    const label=n.length>17?n.slice(0,16)+'…':n;
    return '<g class="lnode '+cls+'" data-lineage-node="'+esc(n)+'" transform="translate('+x+','+y+')"><title>'+esc(n)+'</title>'+
      '<rect x="'+(-W/2)+'" y="'+(-H/2)+'" width="'+W+'" height="'+H+'" rx="4" fill="'+esc(color(n))+'"/>'+
      '<text text-anchor="middle" dy="4">'+esc(label)+'</text></g>';
  }
  function edgeSvg(e,a,b,cls='') {
    const my=(a.y+b.y)/2;
    return '<path class="ledge'+(e.text.length?' is-cited':'')+' '+cls+'" d="M'+a.x+','+(a.y+H/2)+' C'+a.x+','+my+' '+b.x+','+my+' '+b.x+','+(b.y-H/2)+'"><title>'+
      esc(e.parent+' parent of '+e.child+(e.text.length?' / in the text':' / curated, uncited'))+'</title></path>';
  }
  /* Rows by generation, each row ordered by the mean position of its links in
     the rows already placed, then packed left to right without overlap. */
  function place(rows,link,sweeps=4) {
    const pos=new Map();
    rows.forEach(r=>r.forEach((n,i)=>pos.set(n,{x:(i-(r.length-1)/2)*(W+GAP)})));
    for (let s=0;s<sweeps;s++) {
      const order=s%2?[...rows].reverse():rows;
      order.forEach(r=>{
        const want=r.map(n=>{const ns=link(n).filter(m=>pos.has(m)); return [n,ns.length?ns.reduce((t,m)=>t+pos.get(m).x,0)/ns.length:pos.get(n).x];}).sort((a,b)=>a[1]-b[1]);
        let last=-Infinity; const xs=want.map(([,w])=>(last=Math.max(w,last+W+GAP)));
        const shift=(want.reduce((t,w)=>t+w[1],0)-xs.reduce((t,v)=>t+v,0))/xs.length;
        want.forEach(([n],i)=>pos.get(n).x=xs[i]+shift);
      });
    }
    rows.forEach((r,i)=>r.forEach(n=>pos.get(n).y=i*ROW));
    return pos;
  }
  function frame(pos) {
    const xs=[...pos.values()].map(p=>p.x), ys=[...pos.values()].map(p=>p.y);
    return {x:Math.min(...xs)-W, y:Math.min(...ys)-H*2, w:Math.max(...xs)-Math.min(...xs)+W*2, h:Math.max(...ys)-Math.min(...ys)+H*4};
  }

  function tree() {
    const levels=[[focus]], seen=new Set([focus]);
    for (let i=0;i<3;i++) { const next=[...new Set(levels[0].flatMap(up))].filter(n=>!seen.has(n)); if (!next.length) break; next.forEach(n=>seen.add(n)); levels.unshift(next); }
    const mid=levels.length-1;
    const sibs=[...siblings(focus).keys()].filter(n=>!seen.has(n)); sibs.forEach(n=>seen.add(n)); levels[mid]=[...sibs.slice(0,Math.ceil(sibs.length/2)),focus,...sibs.slice(Math.ceil(sibs.length/2))];
    for (let i=0;i<2;i++) { const next=[...new Set(levels[levels.length-1].filter(n=>n===focus||i>0).flatMap(down))].filter(n=>!seen.has(n)); if (!next.length) break; next.forEach(n=>seen.add(n)); levels.push(next); }
    const pos=place(levels,n=>[...up(n),...down(n)],2);
    let svg='';
    D.edges.forEach(e=>{ if (pos.has(e.parent)&&pos.has(e.child)&&pos.get(e.parent).y<pos.get(e.child).y) svg+=edgeSvg(e,pos.get(e.parent),pos.get(e.child),(e.parent===focus||e.child===focus)?'is-focus':''); });
    pos.forEach((p,n)=>svg+=nodeSvg(n,p.x,p.y,n===focus?'is-focus':sibs.includes(n)?'is-sibling':''));
    return {svg,box:frame(pos),caption:'Three generations up, two down, and everyone who shares a parent with '+focus+'.'};
  }
  let whole=null;
  function graph() {
    if (!whole) {
      const rows=[]; D.nodes.forEach(n=>(rows[n.generation] ||= []).push(n.id));
      whole=place(rows.filter(Boolean),n=>[...up(n),...down(n)]);
    }
    const near=new Set([focus,...up(focus),...down(focus)]);
    let svg='';
    D.edges.forEach(e=>svg+=edgeSvg(e,whole.get(e.parent),whole.get(e.child),(e.parent===focus||e.child===focus)?'is-focus':''));
    whole.forEach((p,n)=>svg+=nodeSvg(n,p.x,p.y,n===focus?'is-focus':near.has(n)?'':'is-far'));
    const f=whole.get(focus), full=frame(whole);
    return {svg,box:{x:f.x-700,y:f.y-380,w:1400,h:760},full,caption:D.nodes.length+' people in '+(Math.max(...D.nodes.map(n=>n.generation))+1)+' generations. Drag to pan, scroll to zoom.'};
  }
  function draw() {
    const r=view==='tree'?tree():graph();
    box=r.box; $('lineage-svg').innerHTML=r.svg; setBox();
    $('lineage-caption').textContent=r.caption;
    $('lineage-focus').textContent=focus;
    document.querySelectorAll('[data-lineage-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.lineageView===view)));
  }
  const setBox=()=>$('lineage-svg').setAttribute('viewBox',[box.x,box.y,box.w,box.h].map(Math.round).join(' '));
  function zoom(k,cx=.5,cy=.5) {
    const w=Math.min(Math.max(box.w*k,300),60000), h=box.h*w/box.w;
    box={x:box.x+(box.w-w)*cx,y:box.y+(box.h-h)*cy,w,h}; setBox();
  }
  function fit() { box=view==='tree'?tree().box:graph().full; setBox(); }

  /* ---------- mode ---------- */
  function mode(on,toGraph=false) {
    if (active===on) return;
    if (on) { otherQuery=$('q').value; window.ChatPanel?.mode(false); window.SemanticExplorer?.mode(false); }
    active=on;
    $('lineage-workspace').hidden=!on;
    $('mode-lineage').setAttribute('aria-pressed',String(on));
    if (on) {
      document.querySelector('.workspace').hidden=true;
      ['mode-semantic','mode-graph','mode-chat'].forEach(b=>$(b).setAttribute('aria-pressed','false'));
      $('model-info').hidden=true; $('message').textContent=''; $('q').value='';
      $('q').placeholder='Ask about a family: parents of Karna, brothers of Arjuna, how is Bhishma related to Pandu...';
      $('run-info').textContent='Curated lineage table / answers computed on this page, nothing generated';
      document.querySelector('.examples').innerHTML='<span>Lineage</span>'+examples.map(q=>'<button data-lineage-query="'+esc(q)+'">'+esc(q)+'</button>').join('');
      if (!latest) ask(focus,focus); else draw();
    } else if (toGraph) {
      // Only the graph model is restored here. The semantic and Ask panels
      // restore their own state when they take over.
      document.querySelector('.workspace').hidden=false; $('model-info').hidden=false;
      $('q').value=otherQuery; $('q').placeholder='Who fought Bhishma? Ask a question or explore a character...';
      $('run-info').textContent=originalRun; document.querySelector('.examples').innerHTML=originalExamples;
    }
    icons();
  }
  const originalExamples=document.querySelector('.examples').innerHTML, originalRun=$('run-info').textContent;

  $('mode-lineage').addEventListener('click',()=>mode(true));
  $('mode-graph').addEventListener('click',()=>mode(false,true));
  $('mode-semantic').addEventListener('click',()=>mode(false));
  $('mode-chat').addEventListener('click',()=>mode(false));
  $('query-form').addEventListener('submit',e=>{ if (!active) return; e.preventDefault(); e.stopImmediatePropagation(); ask($('q').value); $('q').value=''; },true);
  document.addEventListener('click',e=>{
    if (!active) return;
    const t=e.target.closest('[data-lineage-query],[data-lineage-node],[data-lineage-view],[data-lineage-zoom]'); if (!t) return;
    if (t.dataset.lineageQuery) { e.stopImmediatePropagation(); ask(t.dataset.lineageQuery); }
    else if (t.dataset.lineageNode) { if (drag?.moved) return; ask(t.dataset.lineageNode,t.dataset.lineageNode); }
    else if (t.dataset.lineageView) { view=t.dataset.lineageView; draw(); }
    else ({in:()=>zoom(.8),out:()=>zoom(1.25),fit}[t.dataset.lineageZoom])();
  },true);
  const svgEl=$('lineage-svg');
  svgEl.addEventListener('wheel',e=>{ e.preventDefault(); const r=svgEl.getBoundingClientRect(); zoom(e.deltaY>0?1.12:.89,(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height); },{passive:false});
  svgEl.addEventListener('pointerdown',e=>{ drag={x:e.clientX,y:e.clientY,box:{...box},moved:false}; });
  window.addEventListener('pointermove',e=>{
    if (!drag) return;
    const r=svgEl.getBoundingClientRect(), dx=e.clientX-drag.x, dy=e.clientY-drag.y;
    if (Math.abs(dx)+Math.abs(dy)>4) drag.moved=true;
    if (drag.moved) { box={...drag.box,x:drag.box.x-dx*drag.box.w/r.width,y:drag.box.y-dy*drag.box.h/r.height}; setBox(); }
  });
  window.addEventListener('pointerup',()=>setTimeout(()=>{drag=null;},0));
  $('lineage-source').textContent=D.edges.length+' parent edges, '+D.edges.filter(e=>e.text.length).length+' of them also stated in the text.';

  window.LineageExplorer={ask,mode,resolve,relate,get state(){return {active,focus,view,latest};}};
})();
