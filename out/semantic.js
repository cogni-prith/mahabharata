(function () {
  'use strict';
  const $ = id => document.getElementById(id), L = window.ViewerLib;
  if (!L || !$('semantic-workspace')) return;
  const colours = {Person:'#39775c',Place:'#3c80b0',Group:'#87963f',Concept:'#be7a42',Event:'#9b62a8',Artifact:'#b95b72',TextWork:'#429798',CelestialBody:'#b09932'};
  const esc = x => String(x == null ? '' : x).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const icons = () => L.createIcons({icons:L,attrs:{'stroke-width':1.6}});
  const dot = kind => '<span class="sem-dot" style="background:'+colours[kind]+'"></span>';
  const badge = (s,c='') => '<span class="sem-badge '+c+'">'+esc(s)+'</span>';
  const nodeButton = n => '<button data-sn="'+esc(n.id)+'">'+esc(n.label)+'</button>';
  const cite = p => '<div class="sem-citation">'+esc(p.parva.replaceAll('_',' '))+(p.cited?' / '+p.chapter:'')+' '+badge(p.cited?'chapter cited':'uncited',p.cited?'':'uncited')+'<span>Sentences '+esc(p.sentence_ids.join(', '))+'</span></div>';
  let active=false, ready=false, data=null, selected=null, tab='nodes', kind=null, scene=null, map=null, userMode=false;
  let searchController=null, request=0, selectionRequest=0, graphQuery=$('q').value, semanticQuery='war';
  const originalExamples=document.querySelector('.examples').innerHTML;
  const originalStats=$('stats').innerHTML, originalRun=$('run-info').textContent;
  const examples=['war','Battles at Kurukshetra','dharma','Weapons of Arjuna','The dice game'];
  async function api(path, options={}) {
    const response=await fetch(path,{...options,credentials:'omit'});
    const result=await response.json();
    if (!response.ok) throw new Error(result.error || 'Local search failed');
    return result;
  }
  function status(text,error=false) {
    $('semantic-status').textContent=text;
    $('semantic-status').classList.toggle('error',error);
  }
  function mode(on) {
    if (active===on) {if(on&&ready&&!scene){scene=makeScene(map);search(semanticQuery);}return;}
    if (on) graphQuery=$('q').value; else semanticQuery=$('q').value;
    active=on; $('semantic-workspace').hidden=!on; document.querySelector('.workspace').hidden=on;
    $('mode-semantic').setAttribute('aria-pressed',String(on)); $('mode-graph').setAttribute('aria-pressed',String(!on));
    $('message').textContent=''; $('q').value=on?semanticQuery:graphQuery;
    $('model-info').hidden=on;
    $('stats').innerHTML=on&&map?'<span><b>'+map.nodes.length.toLocaleString()+'</b>candidate nodes</span><span><b>18</b>parvas</span><span><b>384</b>dimensions</span>':originalStats;
    $('run-info').textContent=on?'Local text embeddings / source-cited retrieval':originalRun;
    $('q').placeholder=on?'Search a concept, event, place, or question...':'Who fought Bhishma? Ask a question or explore a character...';
    document.querySelector('.examples').innerHTML=on?'<span>Explore</span>'+examples.map(q=>'<button data-sem-query="'+esc(q)+'">'+esc(q)+'</button>').join(''):originalExamples;
    if (on && ready) { scene ||= makeScene(map); scene.resize(); if (!data) search(semanticQuery); }
    else if (on) status('Semantic service offline at this address.',true);
  }
  function renderResults() {
    document.querySelectorAll('[data-sem-tab]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.semTab===tab)));
    if (!data) return;
    $('semantic-result-heading').textContent=data.query;
    const card=n=>'<article class="sem-result '+(selected===n.id?'is-selected':'')+'">'+nodeButton(n)+'<div class="sem-meta">'+dot(n.kind)+esc(n.kind)+' '+(n.exact_match?badge('exact name'):'')+'</div><div class="sem-meta">'+(n.basis==='event_proposals'?n.source_count+' location accounts '+badge('proposed','proposed'):'Cosine '+Number(n.score).toFixed(3))+'</div></article>';
    if (tab==='nodes') {
      $('semantic-results').innerHTML=data.nodes.slice(0,4).map(card).join('')+(!kind?'<h3 class="sem-section-label">'+(data.related_places[0]?.basis==='event_proposals'?'Places in event accounts':'Vector-nearest places')+'</h3>'+data.related_places.slice(0,5).map(card).join(''):'')+'<h3 class="sem-section-label">More matches</h3>'+data.nodes.slice(4).map(card).join('');
    } else {
      $('semantic-results').innerHTML=data.passages.map(p=>'<article class="sem-result"><button data-sp="'+p.id+'">'+esc(p.parva.replaceAll('_',' '))+(p.cited?' / '+p.chapter:'')+'</button><div class="sem-meta">'+badge(p.cited?'chapter cited':'uncited',p.cited?'':'uncited')+'Cosine '+p.score.toFixed(3)+'</div><blockquote>'+esc(p.text.slice(0,340))+(p.text.length>340?'...':'')+'</blockquote></article>').join('');
    }
  }
  async function search(query) {
    if (!ready) {mode(true); return;}
    const id=++request; ++selectionRequest; searchController?.abort(); searchController=new AbortController();
    semanticQuery=query; $('q').value=query; status('Searching the local corpus...');
    data=null;selected=null;scene?.update();
    $('semantic-results').innerHTML='<p class="semantic-empty">Searching...</p>';
    $('semantic-inspector').innerHTML='<p class="semantic-empty">Retrieving source passages...</p>';
    document.querySelector('.semantic-status').setAttribute('aria-busy','true');
    $('semantic-results').setAttribute('aria-busy','true');
    try {
      const result=await api('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query,kind}),signal:searchController.signal});
      if (id!==request) return;
      data=result; selected=null; renderResults(); scene?.update();
      status(result.status==='weak_match'?'Weak source matches. No supported answer established.':'Retrieved source passages. Similarity is not factual confidence.');
      if (data.nodes.length && result.status!=='weak_match') await select(data.nodes[0].id);
      else $('semantic-inspector').innerHTML='<div class="sem-profile"><h1>No supported answer</h1><p>Related passages are available for inspection; no fact has been inferred.</p></div>';
    } catch(e) { if (e.name!=='AbortError' && id===request) {status(e.message,true); $('semantic-results').innerHTML='<p class="sem-error">'+esc(e.message)+'</p>';$('semantic-inspector').innerHTML='<p class="semantic-empty">No current results.</p>';} }
    finally { if(id===request) {$('semantic-results').setAttribute('aria-busy','false');document.querySelector('.semantic-status').setAttribute('aria-busy','false');} }
  }
  async function select(id) {
    const seq=++selectionRequest; selected=id; renderResults(); scene?.update();
    $('semantic-inspector').innerHTML='<p class="semantic-empty">Loading cited passages...</p>';
    try {
      const n=await api('/api/node?id='+encodeURIComponent(id));
      if(seq!==selectionRequest) return;
      const names={occurredAt:'location candidate',hasParticipant:'participant candidate',hasTopic:'topic candidate'};
      $('semantic-inspector').innerHTML='<div class="sem-profile"><div class="sem-meta">'+dot(n.kind)+esc(n.kind)+' '+badge('candidate classification')+'</div><h1>'+esc(n.label)+'</h1><p class="eyebrow">SOURCE EXCERPT</p><blockquote>'+esc(n.description)+'</blockquote>'+(n.description_source?cite(n.description_source):badge('uncited','uncited'))+'<p class="sem-profile-note">'+n.source_count+' source passages. A shared name is not a resolved identity.</p></div>'+
        '<details class="sem-evidence"><summary>Source passages ('+n.source_count+')</summary>'+n.passages.map(p=>'<blockquote>'+esc(p.text)+'</blockquote>'+cite(p)).join('')+(n.source_count>n.passages.length?'<p class="sem-profile-note">First '+n.passages.length+' passages shown.</p>':'')+'</details>'+
        '<details class="sem-evidence"><summary>Event-link proposals ('+n.relations.length+')</summary><p class="sem-profile-note">Heuristic candidate scores, not probabilities or verified facts.</p>'+n.relations.map(r=>'<div class="sem-link">'+nodeButton({id:r.subject,label:r.subject_label})+'<small>'+esc(names[r.predicate]||r.predicate)+'</small>'+nodeButton({id:r.object,label:r.object_label})+' '+badge('proposed','proposed')+'<small>Score '+r.score.toFixed(2)+' &middot; <button data-sp="'+r.passage_id+'">Source passage</button></small></div>').join('')+'</details>'+
        '<details class="sem-evidence"><summary>Semantic neighbours</summary>'+n.neighbours.map(r=>'<div class="sem-link">'+nodeButton(r)+'<small>'+esc(r.kind)+' &middot; cosine '+r.score.toFixed(3)+'</small></div>').join('')+'</details>';
      $('semantic-inspector').scrollTop=0;
    } catch(e) {if(seq===selectionRequest)$('semantic-inspector').innerHTML='<p class="sem-error">'+esc(e.message)+'</p>';}
  }
  async function passage(id) {
    const seq=++selectionRequest;
    try {const p=await api('/api/passage?id='+encodeURIComponent(id)); if(seq!==selectionRequest)return;
      $('semantic-inspector').innerHTML='<div class="sem-profile"><p class="eyebrow">SOURCE PASSAGE</p><h1>'+esc(p.parva.replaceAll('_',' '))+(p.cited?' / '+p.chapter:'')+'</h1>'+cite(p)+'<blockquote>'+esc(p.text)+'</blockquote></div>';
      $('semantic-inspector').scrollTop=0;
    } catch(e) {status(e.message,true);}
  }
  async function ontology() {
    const seq=++selectionRequest;
    try {const schema=await api('/api/schema');if(seq!==selectionRequest)return;
      $('semantic-inspector').innerHTML='<div class="sem-profile"><p class="eyebrow">ONTOLOGY</p><h1>Classes &amp; evidence</h1><p class="sem-profile-note">Corpus labels supply candidate types. OWL/SHACL check structure, not historical or narrative truth. Colour is class, not faction.</p>'+schema.classes.map(c=>'<section class="sem-schema"><h3>'+esc(c.label)+'</h3><p>'+esc(c.description)+'</p><div class="sem-meta">'+esc(c.parents.join(' / '))+'</div></section>').join('')+'<a href="/api/schema">Ontology schema JSON</a></div>';
      $('semantic-inspector').scrollTop=0;
    } catch(e) {status(e.message,true);}
  }
  function makeScene(m) {
    const T=L.THREE, stage=$('semantic-stage'), world=new T.Scene(); world.background=new T.Color('#f2f5f3');
    const camera=new T.PerspectiveCamera(42,1,.1,1000), renderer=new T.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
    renderer.setPixelRatio(Math.min(devicePixelRatio,2)); stage.prepend(renderer.domElement);
    const controls=new L.OrbitControls(camera,renderer.domElement); controls.enableDamping=true; controls.autoRotateSpeed=.6;
    const grid=new T.GridHelper(110,11,'#cedbd1','#e1e8e1');grid.position.y=-53;world.add(grid);
    const sprite=document.createElement('canvas');sprite.width=sprite.height=32;
    const ctx=sprite.getContext('2d');ctx.fillStyle='white';ctx.beginPath();ctx.arc(16,16,14,0,Math.PI*2);ctx.fill();
    const material=new T.PointsMaterial({size:1.3,vertexColors:true,map:new T.CanvasTexture(sprite),transparent:true,alphaTest:.2,sizeAttenuation:true});
    const geometry=new T.BufferGeometry(), points=new T.Points(geometry,material);world.add(points);
    const marker=new T.Mesh(new T.SphereGeometry(1.35,16,12),new T.MeshBasicMaterial({color:'#143e2b'}));world.add(marker);marker.visible=false;
    let dim=3, visible=[], labels=[], positions=[], down=null;
    const nodeIndex=new Map(m.nodes.map((n,i)=>[n.id,i]));
    function pos(i) {const p=(dim===3?m.xyz:m.xy)[i];return new T.Vector3((p[0]-.5)*100,(p[1]-.5)*100,dim===3?(p[2]-.5)*100:0);}
    function update() {
      visible=m.nodes.map((_,i)=>i).filter(i=>!kind||m.nodes[i].kind===kind);positions=[];const coloursArray=[];
      visible.forEach(i=>{positions.push(...pos(i).toArray());const c=new T.Color(colours[m.nodes[i].kind]);coloursArray.push(c.r,c.g,c.b);});
      geometry.setAttribute('position',new T.Float32BufferAttribute(positions,3));geometry.setAttribute('color',new T.Float32BufferAttribute(coloursArray,3));geometry.computeBoundingSphere();
      const highlighted=new Set([selected,...(data?.nodes||[]).slice(0,5).map(n=>n.id),...(data?.related_places||[]).slice(0,2).map(n=>n.id)]);
      labels=visible.filter(i=>highlighted.has(m.nodes[i].id)).sort((a,b)=>Number(m.nodes[b].id===selected)-Number(m.nodes[a].id===selected)); $('semantic-labels').replaceChildren();
      labels.forEach(i=>{const el=document.createElement('span');el.textContent=m.nodes[i].label;el.className=m.nodes[i].id===selected?'selected':'';$('semantic-labels').append(el);});
      marker.visible=!!selected && nodeIndex.has(selected) && (!kind||m.nodes[nodeIndex.get(selected)].kind===kind);
      if(marker.visible) marker.position.copy(pos(nodeIndex.get(selected)));
      $('semantic-visible').textContent=visible.length.toLocaleString()+' nodes';$('semantic-point-count').textContent=m.nodes.length.toLocaleString()+' typed profiles';
    }
    function fit(){controls.target.set(0,0,0);camera.position.set(dim===3?88:0,dim===3?48:0,dim===3?165:165);camera.lookAt(0,0,0);controls.update();}
    function resize(){const w=stage.clientWidth,h=stage.clientHeight;if(!w||!h)return;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
    function pointScreen(id) {if(!nodeIndex.has(id))return null;const p=pos(nodeIndex.get(id)).project(camera);return {x:(p.x+1)*stage.clientWidth/2,y:(1-p.y)*stage.clientHeight/2,z:p.z};}
    const ray=new T.Raycaster();ray.params.Points.threshold=1.1;
    function pick(e){const b=renderer.domElement.getBoundingClientRect();ray.setFromCamera(new T.Vector2((e.clientX-b.left)/b.width*2-1,-(e.clientY-b.top)/b.height*2+1),camera);const hit=ray.intersectObject(points)[0];return hit?m.nodes[visible[hit.index]]:null;}
    renderer.domElement.addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY];});
    renderer.domElement.addEventListener('pointerup',e=>{if(down&&Math.hypot(e.clientX-down[0],e.clientY-down[1])<5){const n=pick(e);if(n)select(n.id);}down=null;});
    renderer.domElement.addEventListener('pointermove',e=>{if(e.buttons)return;const n=pick(e),tip=$('semantic-tooltip');tip.hidden=!n;if(n){tip.textContent=n.label+' / '+n.kind;const b=stage.getBoundingClientRect();tip.style.left=Math.min(e.clientX-b.left+10,b.width-220)+'px';tip.style.top=Math.min(e.clientY-b.top+12,b.height-60)+'px';}});
    renderer.domElement.addEventListener('pointerleave',()=>{$('semantic-tooltip').hidden=true;});
    function dimension(d){dim=d;grid.visible=d===3;controls.enableRotate=d===3;if(d===2)controls.autoRotate=false;$('semantic-rotate').disabled=d===2;$('semantic-rotate').setAttribute('aria-pressed',String(controls.autoRotate));$('semantic-3d').setAttribute('aria-pressed',String(d===3));$('semantic-2d').setAttribute('aria-pressed',String(d===2));$('semantic-projection-label').textContent='TEXT UMAP / '+d+'D';update();fit();}
    $('semantic-3d').onclick=()=>dimension(3);$('semantic-2d').onclick=()=>dimension(2);$('semantic-fit').onclick=fit;
    $('semantic-centre').onclick=()=>{if(selected&&nodeIndex.has(selected)){const target=pos(nodeIndex.get(selected));camera.position.add(target.clone().sub(controls.target));controls.target.copy(target);controls.update();}};
    $('semantic-rotate').onclick=()=>{controls.autoRotate=!controls.autoRotate;$('semantic-rotate').setAttribute('aria-pressed',String(controls.autoRotate));};
    const zoom=f=>{camera.position.copy(controls.target.clone().add(camera.position.clone().sub(controls.target).multiplyScalar(f)));controls.update();};
    $('semantic-plus').onclick=()=>zoom(.82);$('semantic-minus').onclick=()=>zoom(1.22);
    stage.addEventListener('keydown',e=>{if(e.key==='+'||e.key==='='){e.preventDefault();zoom(.82);}if(e.key==='-'){e.preventDefault();zoom(1.22);}if(e.key==='Home'){e.preventDefault();fit();}});
    new ResizeObserver(resize).observe(stage);fit();resize();update();
    renderer.setAnimationLoop(()=>{
      if(!active||document.hidden)return;controls.update();renderer.render(world,camera);
      const b=stage.getBoundingClientRect(),occupied=[];
      [...$('semantic-labels').children].forEach((el,k)=>{
        const p=pointScreen(m.nodes[labels[k]].id);el.hidden=false;
        const w=el.offsetWidth,h=el.offsetHeight;
        const candidates=[[p.x+8,p.y-h/2],[p.x-w-8,p.y-h/2],[p.x+8,p.y+h]];
        const slot=p.z<1&&candidates.find(([x,y])=>x>=0&&y>=0&&x+w<b.width&&y+h<b.height&&!occupied.some(r=>x<r[2]+3&&x+w>r[0]-3&&y<r[3]+3&&y+h>r[1]-3));
        el.hidden=!slot;if(slot){el.style.left=slot[0]+'px';el.style.top=slot[1]+'px';occupied.push([slot[0],slot[1],slot[0]+w,slot[1]+h]);}
      });
    });
    return {renderer,update,resize,pointScreen,fit,get dimension(){return dim;}};
  }
  $('mode-semantic').onclick=()=>{userMode=true;mode(true);};$('mode-graph').onclick=()=>{userMode=true;mode(false);};
  $('query-form').addEventListener('submit',e=>{if(active){e.preventDefault();e.stopImmediatePropagation();search($('q').value);}},true);
  document.addEventListener('click',e=>{
    if(!active)return;const b=e.target.closest('button');if(!b)return;
    if(b.dataset.semQuery){e.stopImmediatePropagation();search(b.dataset.semQuery);}
    if(b.dataset.sn)select(b.dataset.sn);
    if(b.dataset.sp)passage(b.dataset.sp);
    if(b.dataset.semTab){tab=b.dataset.semTab;renderResults();}
    if(b.dataset.semClass!==undefined){kind=b.dataset.semClass||null;document.querySelectorAll('[data-sem-class]').forEach(el=>el.setAttribute('aria-pressed',String((el.dataset.semClass||null)===kind)));scene?.update();search(semanticQuery);}
  },true);
  $('semantic-ontology').onclick=ontology;
  async function init() {
    icons();
    if(location.protocol==='file:') return;
    try {
      const metadata=await api('/api/status');map=await api('/api/map');ready=true;
      $('semantic-classes').innerHTML='<button data-sem-class="" aria-pressed="true">All</button>'+Object.keys(colours).map(k=>'<button data-sem-class="'+k+'" aria-pressed="false" title="Candidate '+k+' classifications">'+dot(k)+k+' <span class="muted">'+(map.classes[k]||0).toLocaleString()+'</span></button>').join('');
      $('semantic-provenance').innerHTML='<p>'+metadata.passages.toLocaleString()+' passages &middot; '+metadata.cited_passages.toLocaleString()+' chapter-cited &middot; '+metadata.parvas+' parvas.</p><p>Pretrained text encoder: '+esc(metadata.model)+'. '+metadata.dimension+' dimensions. UMAP seed '+metadata.projection_seed+'.</p><p>Not the RotatE model. Map proximity is a projection, not a fact. Event links and entity types remain candidates.</p>';
      if(!userMode||active)mode(true);icons();
    } catch(e) {status('Local semantic index unavailable: '+e.message,true);}
  }
  window.SemanticExplorer={search,select,mode,get state(){return {active,ready,selected,kind,data};},get scene(){return scene;},get map(){return map;}};
  init();
})();
