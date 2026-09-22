"""Independent cited retrieval index. Never reads annotations or rewrites KGE inputs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3

import pandas as pd
from rdflib import Dataset, Literal, Namespace, RDF, RDFS
from vocab import NOT_A_NAME
from spacy.lang.en.stop_words import STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
MBH = Namespace('http://example.org/mbh/')
KINDS = {'PERSON':'Person','GPE':'Place','LOC':'Place','DYNASTY':'Group',
         'NORP':'Group','CONCEPT':'Concept','WEAPON':'Artifact','SACRED_TEXT':'TextWork','CELESTIAL':'CelestialBody'}
EVENT_WORD = re.compile(r'\b(battle|war|duel|sacrifice|marriage|exile)\b', re.I)
STOP_NAMES = {n.casefold() for n in NOT_A_NAME} | {'king','lord','queen','prince'}


def content_id(*parts):
    return hashlib.sha256(json.dumps(parts,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()[:24]


def location_in(text, place):
    return bool(re.search(r'\b(?:at|in|on|near)\s+(?:(?:the|holy|fields?|plains?|of|sacred)\s+){0,8}'
                          +re.escape(place)+r'\b',text,re.I))


def build_rows(master, mentions, relations=None):
    type_counts=Counter((str(r.surface).strip(),KINDS.get(r.label)) for r in mentions.itertuples())
    dominant={}
    for (label,kind),count in type_counts.items():
        if kind and count>dominant.get(label,(None,0))[1]: dominant[label]=(kind,count)
    rejected=Counter()
    anchors, by_sentence = {}, defaultdict(list)
    for r in mentions.itertuples():
        key=(r.parva,int(r.sent_id))
        chapter=int(r.adhyaya_ordinal) if pd.notna(r.adhyaya_ordinal) and r.chapter_cited else None
        anchors[key]=(r.sentence,chapter)
        by_sentence[key].append((str(r.surface).strip(),str(r.label)))
    records=[]
    for r in master.dropna(subset=['sentence']).sort_values(['parva','sent_id']).itertuples():
        if not re.match(r'^(0[1-9]|1[0-8])_',r.parva):
            continue
        key=(r.parva,int(r.sent_id)); anchor=anchors.get(key)
        chapter=anchor[1] if anchor and anchor[0]==r.sentence else None
        records.append({'key':key,'text':str(r.sentence),'chapter':chapter})
    passages, lookup, batch = [], {}, []

    def flush():
        if not batch:
            return
        text='\n'.join(r['text'] for r in batch)
        parva=batch[0]['key'][0]; chapter=batch[0]['chapter']
        ids=[r['key'][1] for r in batch]
        p={'id':content_id(parva,chapter,ids,text),'parva':parva,'chapter':chapter,
           'cited':chapter is not None,'text':text,'sentence_ids':ids}
        passages.append(p)
        for r in batch:
            lookup[r['key']]=p
        batch.clear()

    for r in records:
        if batch and ((batch[0]['key'][0],batch[0]['chapter']) != (r['key'][0],r['chapter'])
                      or sum(len(x['text'].split()) for x in batch)+len(r['text'].split())>120):
            flush()
        batch.append(r)
    flush()
    nodes, links = {}, {}

    def node(label,kind,p,text,identity=None):
        label=re.sub(r'\s+',' ',label).strip()
        nid=content_id(kind,identity if identity else label.casefold())
        if nid not in nodes:
            nodes[nid]={'id':nid,'label':label,'kind':kind,'description':text,
                        'mention_count':0,'status':'candidate','passage_ids':[]}
        n=nodes[nid]; n['mention_count']+=1
        if p['id'] not in n['passage_ids']:
            n['passage_ids'].append(p['id'])
        return nid

    def link(s,pred,o,p,score):
        key=content_id(s,pred,o,p['id'])
        links[key]={'id':key,'subject':s,'predicate':pred,'object':o,'passage_id':p['id'],
                    'score':score,'status':'proposed'}

    direct=defaultdict(list)
    if relations is not None:
        for r in relations[relations.predicate.isin(['foughtWith','slainBy','defeatedBy'])].itertuples():
            if all(pd.isna(getattr(r,c,None)) for c in ['coref_subject','coref_object','nplink_subject','nplink_object']):
                direct[(r.parva,int(r.sent_id))].append((r.subject,r.object))
    for r in records:
        key=r['key']; p=lookup[key]; text=r['text']; places=[]; actors={}
        marks=by_sentence.get(key,[])
        for label,tag in marks:
            kind=KINDS.get(tag)
            if not kind or len(label)<2 or not any(c.isalpha() for c in label):
                continue
            if label.casefold() in STOP_NAMES or label.casefold() in STOP_WORDS:
                rejected['stopword_or_title']+=1
                continue
            if kind!='Concept' and not label[0].isupper():
                rejected['nonproper_named_type']+=1
                continue
            if kind in ('Place','Group','Artifact','TextWork') and dominant[label][0]!=kind:
                rejected['conflicting_dominant_type']+=1
                continue
            if kind=='TextWork' and EVENT_WORD.search(label):
                rejected['event_phrase_as_text_work']+=1
                continue
            nid=node(label,kind,p,text)
            if kind=='Place': places.append((label,nid))
            if kind in ('Person','Deity','Group'): actors[label]=nid
        match=EVENT_WORD.search(text)
        spatial=[(label,nid) for label,nid in places if location_in(text,label)]
        is_event=any(tag=='EVENT' for _,tag in marks) or (match and spatial)
        if is_event:
            term=match.group(1).lower() if match else next(label for label,tag in marks if tag=='EVENT')
            title=term.capitalize()+' account / '+key[0].replace('_',' ')+' / '+str(r['chapter'] or '?')+' / '+str(key[1])
            event=node(title,'Event',p,text,identity=key)
            topic='war' if term in ('war','battle','duel') else term.lower()
            concept=node(topic,'Concept',p,text)
            link(event,'hasTopic',concept,p,.75)
            for _,place in spatial:
                if not re.search(r'\b(will|would|should|might|could|not|never)\b',text,re.I):
                    link(event,'occurredAt',place,p,.6)
            for pair in direct[key]:
                for name in pair:
                    if name in actors:
                        link(event,'hasParticipant',actors[name],p,.7)
        # Generic war search also has a topic node when the sentence has no NER span.
        elif re.search(r'\bwar\b',text,re.I):
            node('war','Concept',p,text)
    build_rows.rejected=dict(rejected)
    return passages,list(nodes.values()),list(links.values())


def write_database(path,passages,nodes,links,metadata):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.sqlite')
    if tmp.exists(): tmp.unlink()  # Only our incomplete build, never a live database.
    with sqlite3.connect(tmp) as db:
        db.executescript('''
        CREATE TABLE passages(id TEXT PRIMARY KEY,parva TEXT,chapter INTEGER,cited INTEGER,text TEXT,sentence_ids TEXT);
        CREATE TABLE nodes(id TEXT PRIMARY KEY,label TEXT,kind TEXT,description TEXT,mention_count INTEGER,status TEXT);
        CREATE TABLE node_passages(node_id TEXT,passage_id TEXT,PRIMARY KEY(node_id,passage_id));
        CREATE TABLE links(id TEXT PRIMARY KEY,subject TEXT,predicate TEXT,object TEXT,passage_id TEXT,score REAL,status TEXT);
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
        CREATE VIRTUAL TABLE passage_fts USING fts5(text,content=passages,content_rowid=rowid);
        CREATE INDEX node_kind ON nodes(kind);
        CREATE INDEX link_subject ON links(subject);
        CREATE INDEX link_object ON links(object);
        ''')
        db.executemany('INSERT INTO passages VALUES(?,?,?,?,?,?)',[(p['id'],p['parva'],p['chapter'],int(p['cited']),p['text'],json.dumps(p['sentence_ids'])) for p in passages])
        db.executemany('INSERT INTO nodes VALUES(?,?,?,?,?,?)',[(n['id'],n['label'],n['kind'],n['description'],n['mention_count'],n['status']) for n in nodes])
        db.executemany('INSERT INTO node_passages VALUES(?,?)',[(n['id'],p) for n in nodes for p in n['passage_ids']])
        db.executemany('INSERT INTO links VALUES(?,?,?,?,?,?,?)',[tuple(l[k] for k in ['id','subject','predicate','object','passage_id','score','status']) for l in links])
        db.executemany('INSERT INTO metadata VALUES(?,?)',[(k,json.dumps(v)) for k,v in metadata.items()])
        db.execute("INSERT INTO passage_fts(passage_fts) VALUES('rebuild')")
    tmp.replace(path)


def export_rdf(path,passages,nodes,links):
    ds=Dataset(); ds.bind('mbh',MBH)
    index=ds.graph(MBH['graph/semantic-index']); proposals=ds.graph(MBH['graph/semantic-candidates'])
    uri=lambda kind,key:MBH['semantic/'+kind+'/'+key]
    for p in passages:
        u=uri('passage',p['id'])
        for pred,value in [(RDF.type,MBH.Passage),(MBH.sourceText,Literal(p['text'])),
                           (MBH.parvan,Literal(p['parva'])),(MBH.sourceSentenceIds,Literal(json.dumps(p['sentence_ids']))),
                           (MBH.citedAtChapterLevel,Literal(p['cited']))]: index.add((u,pred,value))
        if p['chapter'] is not None: index.add((u,MBH.chapterOrdinal,Literal(p['chapter'])))
    for n in nodes:
        u=uri('entity',n['id'])
        index.add((u,RDF.type,MBH[n['kind']])); index.add((u,RDFS.label,Literal(n['label'])))
        index.add((u,MBH.status,Literal('candidate')))
        for p in n['passage_ids']: index.add((u,MBH.sourcePassage,uri('passage',p)))
    for l in links:
        u=uri('proposal',l['id'])
        for pred,value in [(RDF.type,MBH.Proposal),(MBH.status,Literal('proposed')),
                           (MBH.subject,uri('entity',l['subject'])),(MBH.predicate,MBH[l['predicate']]),
                           (MBH.object,uri('entity',l['object'])),(MBH.score,Literal(l['score'])),
                           (MBH.sourcePassage,uri('passage',l['passage_id']))]: proposals.add((u,pred,value))
    ds.serialize(path,format='trig')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=ROOT)
    root=parser.parse_args().root
    master=root/'mahabharata_ner_by_sentence_master (2).csv'; mentions=root/'build/mentions.parquet'
    relations=root/'build/reviewed_relations.parquet'
    passages,nodes,links=build_rows(pd.read_csv(master),pd.read_parquet(mentions),
                                   pd.read_parquet(relations) if relations.exists() else None)
    meta={'source_sha256':hashlib.file_digest(master.open('rb'),'sha256').hexdigest(),
          'mentions_sha256':hashlib.file_digest(mentions.open('rb'),'sha256').hexdigest(),
          'passages':len(passages),'nodes':len(nodes),'proposals':len(links),
          'classes':dict(Counter(n['kind'] for n in nodes)),
          'cited_passages':sum(p['cited'] for p in passages),
          'parvas':len({p['parva'] for p in passages}),
          'rejected_mentions':build_rows.rejected,
          'limitations':'Types and event links are candidates. Descriptions are exact excerpts. No independent factual evaluation.'}
    write_database(root/'build/semantic/index.sqlite',passages,nodes,links,meta)
    export_rdf(root/'out/semantic.trig',passages,nodes,links)
    (root/'build/semantic/index_report.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))


if __name__=='__main__':
    main()
