/* Ask mode. The one place in this interface where a question leaves the
   machine, and only when the service was started with a key.

   Two things the panel exists to keep honest. An answer is prose and prose can
   round off a caveat, so every answer is drawn beside the evidence it was
   built from, with the citation on each fact rather than inside the sentence.
   And an appellation candidate is never rendered as an identity: unresolved
   names get their own tab and their own marker, whatever the model wrote.

   The other two modes are not modified. This file registers its own listeners
   and asks SemanticExplorer to stand down when Ask is opened. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id), L = window.ViewerLib;
  if (!$('chat-workspace')) return;
  const esc = x => String(x == null ? '' : x).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const icons = () => L && L.createIcons({icons:L,attrs:{'stroke-width':1.6}});
  const badge = (s,c='') => '<span class="chat-badge '+c+'">'+esc(s)+'</span>';
  const parva = p => esc(String(p).replaceAll('_',' '));
  const examples = ['Who killed Karna?','Who is Radheya?','Who taught Arjuna?','Tell me about Yudhistira'];

  let active=false, ready=false, config=null, tab='facts', latest=null, pending=false;
  let controller=null, request=0, history=[], otherQuery='';
  const originalExamples=document.querySelector('.examples').innerHTML;
  const originalRun=$('run-info').textContent;

  async function api(path,options={}) {
    const response=await fetch(path,{...options,credentials:'omit'});
    const result=await response.json();
    if (!response.ok) throw new Error(result.error||'Local request failed');
    return result;
  }
  function status(text,error=false) {
    $('chat-status').textContent=text;
    $('chat-status').classList.toggle('error',error);
  }

  function mode(on) {
    if (active===on) return;
    if (on) { otherQuery=$('q').value; window.SemanticExplorer?.mode(false); }
    active=on;
    $('chat-workspace').hidden=!on;
    document.querySelector('.workspace').hidden=on;
    $('mode-chat').setAttribute('aria-pressed',String(on));
    if (on) { $('mode-semantic').setAttribute('aria-pressed','false'); $('mode-graph').setAttribute('aria-pressed','false'); }
    $('model-info').hidden=on;
    $('message').textContent='';
    $('q').value=on?'':otherQuery;
    $('q').placeholder=on?'Ask a question. Spelling and grammar do not have to be right...':'Who fought Bhishma? Ask a question or explore a character...';
    $('run-info').textContent=on?(config?.generated?'Generated from cited retrieval / '+config.model:'Retrieval only / no model configured'):originalRun;
    document.querySelector('.examples').innerHTML=on?'<span>Ask</span>'+examples.map(q=>'<button data-chat-query="'+esc(q)+'">'+esc(q)+'</button>').join(''):originalExamples;
    icons();
  }

  function renderEvidence() {
    // Scoped to the tablist: the jump button in an answer shares the data
    // attribute but is not a tab, and aria-selected on it is a lie.
    document.querySelectorAll('.chat-tabs [data-chat-tab]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.chatTab===tab)));
    if (!latest) return;
    const box=$('chat-evidence');
    if (tab==='facts') {
      box.innerHTML=latest.facts.length?latest.facts.map(f=>
        '<article class="chat-fact'+(f.unresolved_via.length?' is-unresolved':'')+'">'+
        '<div class="chat-triple"><b>'+esc(f.subject)+'</b><small>'+esc(f.predicate)+'</small><b>'+esc(f.object)+'</b></div>'+
        '<div class="chat-meta">'+(f.citations||[]).map(c=>badge(c.replace(/_/g,' '),c==='uncited'?'uncited':'cited')).join('')+
        (f.support>1?badge('stated in '+f.support+' places','support'):'')+
        badge('confidence '+Number(f.confidence).toFixed(2))+badge(f.pattern)+
        (f.unresolved_via.length?badge('via unresolved '+f.unresolved_via.join(', '),'unresolved'):'')+
        (f.may_be||[]).map(n=>badge(n.appellation+' may be '+n.candidate+' ('+Number(n.confidence).toFixed(2)+')','unresolved')).join('')+'</div>'+
        (f.evidence&&f.quoted?'<blockquote>'+esc(f.evidence)+'</blockquote>':'')+'</article>').join(''):
        '<p class="chat-empty">The graph holds no relation for the names in this question.</p>';
    } else if (tab==='passages') {
      box.innerHTML=latest.passages.length?latest.passages.map(p=>
        '<article class="chat-fact"><div class="chat-meta">'+
        (p.cited?badge(parva(p.parva)+' '+p.chapter,'cited'):badge('uncited','uncited'))+
        badge('cosine '+Number(p.score).toFixed(3))+'</div><blockquote>'+esc(p.text)+'</blockquote></article>').join(''):
        '<p class="chat-empty">No passage matched this question.</p>';
    } else if (tab==='names') {
      const unresolved=latest.resolutions.filter(r=>!r.resolved), resolved=latest.resolutions.filter(r=>r.resolved);
      box.innerHTML=(unresolved.length?'<h3 class="chat-section">Unresolved</h3><p class="chat-note">Candidate referents. The corpus does not settle these and the graph merges none of them.</p>'+
          unresolved.map(r=>'<article class="chat-fact is-unresolved"><div class="chat-triple"><b>'+esc(r.appellation)+'</b><small>candidate</small><b>'+esc(r.name)+'</b></div>'+
          '<div class="chat-meta">'+(r.confidence==null
            ?badge('no apposition evidence, corpus prior only','uncited')
            :badge('confidence '+Number(r.confidence).toFixed(2),'unresolved')+badge(r.evidence_count+' sightings'))+
          badge(r.status)+badge('1 of '+r.candidates)+'</div></article>').join(''):'')+
        (resolved.length?'<h3 class="chat-section">Resolved</h3>'+resolved.map(r=>
          '<article class="chat-fact"><div class="chat-triple"><b>'+esc(r.name)+'</b></div><div class="chat-meta">'+
          badge(r.via==='exact'?'exact name':'spelling match on "'+r.surface+'"')+
          (r.via==='fuzzy'?badge('score '+Math.round(r.score)):'')+'</div></article>').join(''):'')||
        '<p class="chat-empty">No name in this question matched the graph.</p>';
      if ((latest.coreference||[]).length) box.insertAdjacentHTML('beforeend',
        '<h3 class="chat-section">May be the same person</h3>'+
        '<p class="chat-note">Apposition evidence from the corpus, not a merge. A list of rival '+
        'answers above may be one person under several epithets. Nothing here is settled: '+
        'Kounteya is any son of Kunti, and Partha is used of Yudhishthira almost as often as of Arjuna.</p>'+
        latest.coreference.map(n=>'<article class="chat-fact is-unresolved">'+
        '<div class="chat-triple"><b>'+esc(n.appellation)+'</b><small>may be</small><b>'+esc(n.candidate)+'</b></div>'+
        '<div class="chat-meta">'+badge('confidence '+Number(n.confidence).toFixed(2),'unresolved')+
        badge(n.evidence_count+' sightings')+badge('1 of '+n.candidates)+'</div></article>').join(''));
      const variants={};
      resolved.forEach(r=>{(variants[r.surface] ||= []).push(r);});
      const many=Object.entries(variants).filter(([,g])=>g.length>1);
      if (many.length) box.insertAdjacentHTML('beforeend','<p class="chat-note">'+many.map(([s,g])=>
        '"'+esc(s)+'" matched '+g.length+' spellings. They are separate entities and nothing here says they are the same person.').join(' ')+'</p>');
    } else {
      box.innerHTML='<p class="chat-note">'+esc(config?.generated?'Exactly this text was sent, with the question and the last few turns.':'This is what would be sent. Nothing was.')+'</p><pre class="chat-context">'+esc(latest.context)+'</pre>';
    }
    icons();
  }

  /* A small Markdown subset, applied AFTER escaping. Because every < and &
     is already an entity by this point, no pattern below can reintroduce a
     tag from model output; the only tags are the ones this function writes. */
  function markdown(text) {
    const lines=esc(text).split('\n');
    let html='', list=false;
    const inline=s=>s.replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>').replace(/(?<!\*)\*([^*]+)\*(?!\*)/g,'<i>$1</i>');
    for (const raw of lines) {
      const line=raw.trim();
      const bullet=/^[-*]\s+(.*)$/.exec(line);
      if (bullet) { if(!list){html+='<ul>';list=true;} html+='<li>'+inline(bullet[1])+'</li>'; continue; }
      if (list) { html+='</ul>'; list=false; }
      if (!line || /^([-=_])\1{2,}$/.test(line)) continue;
      html+='<p>'+inline(line.replace(/^#{1,6}\s*/,''))+'</p>';
    }
    return html+(list?'</ul>':'');
  }
  function bubble(role,text,extra='') {
    const el=document.createElement('article');
    el.className='chat-turn chat-'+role;
    const body=role==='user'?'<p>'+esc(text)+'</p>':markdown(text);
    el.innerHTML='<div class="chat-role">'+(role==='user'?'You':'Answer')+'</div><div class="chat-body">'+body+'</div>'+extra;
    $('chat-log').append(el);
    el.scrollIntoView({block:'nearest'});
    return el;
  }

  async function ask(query) {
    if (!ready) { status('Ask service unavailable at this address.',true); return; }
    if (pending) return;
    query=String(query||'').trim();
    if (!query) return;
    const id=++request; pending=true; controller?.abort(); controller=new AbortController();
    $('q').value='';
    bubble('user',query);
    const waiting=bubble('assistant',config.generated?'Retrieving and answering...':'Retrieving...');
    waiting.classList.add('is-waiting');
    status('Retrieving from the graph and the corpus...');
    try {
      const result=await api('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({query,history}),signal:controller.signal});
      if (id!==request) return;
      latest=result; waiting.remove();
      const marks=[];
      if (result.retrieval_status==='weak_match') marks.push(badge('weak passage match','uncited'));
      if (result.resolutions.some(r=>!r.resolved)) marks.push(badge('rests on an unresolved name','unresolved'));
      if (!result.facts.length) marks.push(badge('no asserted facts','uncited'));
      marks.push(badge(result.generated?'generated':'retrieval only',result.generated?'generated':''));
      const foot='<div class="chat-marks">'+marks.join('')+'<button data-chat-tab="facts" class="chat-jump">'+result.facts.length+' facts, '+result.passages.length+' passages</button></div>';
      if (result.generated) {
        bubble('assistant',result.answer,foot);
        history=[...history,{role:'user',content:query},{role:'assistant',content:result.answer}].slice(-6);
      } else {
        const el=bubble('assistant','No model is configured, so nothing was generated and nothing was sent. The evidence an answer would be built from is on the right.',foot);
        el.classList.add('is-retrieval');
      }
      status(result.notice);
      tab='facts'; renderEvidence();
    } catch(e) {
      if (e.name==='AbortError') return;
      if (id===request) { waiting.remove(); bubble('assistant',e.message).classList.add('is-error'); status(e.message,true); }
    } finally { if (id===request) pending=false; icons(); }
  }

  $('mode-chat').addEventListener('click',()=>mode(true));
  $('mode-semantic').addEventListener('click',()=>mode(false));
  $('mode-graph').addEventListener('click',()=>mode(false));
  $('query-form').addEventListener('submit',e=>{
    if (!active) return;
    e.preventDefault(); e.stopImmediatePropagation(); ask($('q').value);
  },true);
  document.addEventListener('click',e=>{
    if (!active) return;
    const b=e.target.closest('button'); if (!b) return;
    if (b.dataset.chatQuery) { e.stopImmediatePropagation(); ask(b.dataset.chatQuery); }
    if (b.dataset.chatTab) { tab=b.dataset.chatTab; renderEvidence(); }
  },true);

  async function init() {
    if (location.protocol==='file:') return;
    try {
      const metadata=await api('/api/status');
      config=metadata.chat;
      if (!config?.enabled) return;
      ready=true; $('mode-chat').hidden=false;
      $('chat-remote').hidden=false;
      $('chat-remote').textContent=config.remote
        ? 'This mode sends your question and the retrieved context to '+config.model+'. The other modes send nothing.'
        : 'No key configured. Nothing leaves this machine.';
      $('chat-remote').classList.toggle('is-remote',!!config.remote);
      status(config.notice);
      icons();
    } catch(e) { /* the semantic panel already reports a dead service */ }
  }

  window.ChatPanel={ask,mode,render:markdown,get state(){return {active,ready,pending,tab,latest,config};}};
  init();
})();
