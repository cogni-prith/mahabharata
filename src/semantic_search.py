"""Local hybrid retrieval. Similarity is relevance, never evidence of a true relation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading

import numpy as np
from semantic_index import ROOT

MODEL = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
SEED = 20260916
KINDS = {'Person','Deity','Place','Group','Event','Concept','Artifact','TextWork','CelestialBody'}


def validate_query(query):
    if not isinstance(query,str) or not query.strip() or len(query)>1000:
        raise ValueError('Enter a question between 1 and 1000 characters.')
    return query.strip()


def fts_query(query):
    return ' OR '.join('"'+t+'"' for t in re.findall(r'[^\W_]+',query,re.UNICODE)[:16])


def connect(path):
    db=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    return db


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def build_embeddings(root=ROOT):
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer
    import torch
    import umap
    folder=root/'build/semantic'; path=folder/'index.sqlite'
    with connect(path) as db:
        passages=[dict(r) for r in db.execute('SELECT * FROM passages ORDER BY rowid')]
        nodes=[dict(r) for r in db.execute('SELECT * FROM nodes ORDER BY rowid')]
        metadata={r['key']:json.loads(r['value']) for r in db.execute('SELECT * FROM metadata')}
        contexts=[]
        for n in nodes:
            refs=db.execute('SELECT p.id FROM node_passages np JOIN passages p ON p.id=np.passage_id WHERE np.node_id=? ORDER BY p.cited DESC,p.rowid',(n['id'],)).fetchall()
            contexts.append([refs[i]['id'] for i in np.linspace(0,len(refs)-1,min(12,len(refs)),dtype=int)])
    revision=HfApi().model_info(MODEL).sha
    model=SentenceTransformer(MODEL,revision=revision,cache_folder=str(folder/'models'),
                              device='cuda' if torch.cuda.is_available() else 'cpu',trust_remote_code=False)
    model.max_seq_length=256
    torch.set_num_threads(4)
    print('Encoding',len(passages),'passages and',len(nodes),'typed profiles on',model.device,flush=True)
    vectors=model.encode([p['text'] for p in passages]+[n['label']+'. '+n['kind'] for n in nodes],batch_size=128,
                         normalize_embeddings=True,show_progress_bar=True,convert_to_numpy=True)
    vectors=vectors.astype('float32')
    np.save(folder/'passage_vectors.npy',vectors[:len(passages)])
    pi={p['id']:i for i,p in enumerate(passages)}
    nv=vectors[len(passages):].copy()
    for i,refs in enumerate(contexts):
        nv[i]=.5*nv[i]+.5*vectors[[pi[p] for p in refs]].mean(axis=0)
    nv/=np.maximum(np.linalg.norm(nv,axis=1,keepdims=True),1e-8)
    np.save(folder/'node_vectors.npy',nv)
    coords=[]
    for dim in (2,3):
        print('Projecting',dim,'dimensions',flush=True)
        xyz=umap.UMAP(n_components=dim,n_neighbors=min(15,len(nodes)-1),min_dist=.1,
                      metric='cosine',random_state=SEED).fit_transform(nv)
        coords.append(((xyz-xyz.min(0))/np.maximum(np.ptp(xyz,axis=0),1e-6)).round(5).tolist())
    manifest={**metadata,'model':MODEL,'revision':revision,'dimension':int(nv.shape[1]),
              'max_tokens':256,'projection_seed':SEED,'metric':'cosine','training':'pretrained; no corpus fine-tuning',
              'node_encoding':'Normalized 50% typed label + 50% mean of up to 12 source passages, evenly sampled in cited-first source order',
              'index_sha256':digest(path),'node_ids':[n['id'] for n in nodes],
              'passage_ids':[p['id'] for p in passages],'ranking':'Dense/FTS reciprocal-rank fusion, k=60',
              'node_vectors_sha256':digest(folder/'node_vectors.npy'),
              'passage_vectors_sha256':digest(folder/'passage_vectors.npy')}
    (folder/'map.json').write_text(json.dumps({'nodes':[{k:n[k] for k in ['id','label','kind','mention_count']} for n in nodes],
        'xy':coords[0],'xyz':coords[1],'seed':SEED,'model':MODEL,'classes':metadata['classes']},separators=(',',':')))
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Semantic model ready:',len(nodes),'nodes;',len(passages),'passages; seed',SEED,flush=True)


class SearchIndex:
    def __init__(self,root=ROOT):
        self.root=Path(root); self.folder=self.root/'build/semantic'; self.path=self.folder/'index.sqlite'
        self.manifest=json.loads((self.folder/'manifest.json').read_text())
        if digest(self.path)!=self.manifest['index_sha256']:
            raise ValueError('The corpus index changed. Rebuild semantic embeddings before serving.')
        self.pv=np.load(self.folder/'passage_vectors.npy',allow_pickle=False)
        self.nv=np.load(self.folder/'node_vectors.npy',allow_pickle=False)
        self.pids=self.manifest['passage_ids']; self.nids=self.manifest['node_ids']
        if self.pv.shape!=(len(self.pids),self.manifest['dimension']) or self.nv.shape!=(len(self.nids),self.manifest['dimension']):
            raise ValueError('Semantic vectors do not match their manifest.')
        if not np.isfinite(self.pv).all() or not np.isfinite(self.nv).all():
            raise ValueError('Non-finite semantic vector.')
        for name in ['node_vectors','passage_vectors']:
            if digest(self.folder/(name+'.npy'))!=self.manifest[name+'_sha256']:
                raise ValueError('Vector checksum mismatch; rebuild semantic embeddings.')
        self.pindex={p:i for i,p in enumerate(self.pids)}; self.nindex={n:i for i,n in enumerate(self.nids)}
        with connect(self.path) as db:
            self.nodes={r['id']:dict(r) for r in db.execute('SELECT * FROM nodes')}
        if set(self.nodes)!=set(self.nids): raise ValueError('Node index mismatch')
        self.map=json.loads((self.folder/'map.json').read_text())
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(4)
        self.model=SentenceTransformer(self.manifest['model'],revision=self.manifest['revision'],
            cache_folder=str(self.folder/'models'),local_files_only=True,device='cpu',trust_remote_code=False)
        self.model.max_seq_length=self.manifest['max_tokens']
        # ponytail: one CPU encoder lock; batch concurrent requests if load warrants it.
        self.lock=threading.Lock()

    @staticmethod
    def passage_record(row):
        p=dict(row); p['sentence_ids']=json.loads(p['sentence_ids']); p['cited']=bool(p['cited'])
        return p

    def passage(self,pid):
        with connect(self.path) as db:
            p=db.execute('SELECT * FROM passages WHERE id=?',(pid,)).fetchone()
        if not p: raise KeyError('Passage not found')
        return self.passage_record(p)

    def node(self,nid):
        if nid not in self.nodes: raise KeyError('Node not found')
        n=dict(self.nodes[nid])
        with connect(self.path) as db:
            n['passages']=[self.passage_record(p) for p in db.execute(
                'SELECT p.* FROM node_passages np JOIN passages p ON p.id=np.passage_id WHERE np.node_id=? ORDER BY p.cited DESC,p.rowid LIMIT 12',(nid,))]
            desc=db.execute('SELECT p.* FROM node_passages np JOIN passages p ON p.id=np.passage_id WHERE np.node_id=? AND instr(p.text,?)>0 LIMIT 1',(nid,n['description'])).fetchone()
            n['description_source']=self.passage_record(desc) if desc else None
            n['relations']=[dict(r) for r in db.execute('SELECT * FROM links WHERE subject=? OR object=? LIMIT 120',(nid,nid))]
            for link in n['relations']:
                link['subject_label']=self.nodes[link['subject']]['label']
                link['object_label']=self.nodes[link['object']]['label']
            n['source_count']=db.execute('SELECT count(*) FROM node_passages WHERE node_id=?',(nid,)).fetchone()[0]
        scores=self.nv@self.nv[self.nindex[nid]]
        order=np.argsort(-scores)
        n['neighbours']=[{**{k:self.nodes[self.nids[i]][k] for k in ['id','label','kind']},'score':float(scores[i])}
                         for i in order if self.nids[i]!=nid][:12]
        return n

    def search(self,query,kind=None,limit=12):
        query=validate_query(query)
        if kind is not None and kind not in KINDS: raise ValueError('Unknown ontology class')
        if not isinstance(limit,int) or isinstance(limit,bool) or not 1<=limit<=48: raise ValueError('Limit must be between 1 and 48')
        with self.lock:
            q=self.model.encode(query,normalize_embeddings=True,convert_to_numpy=True)
        ps=self.pv@q; ns=self.nv@q
        order=np.argsort(-ps)[:150]; ranks={int(i):1/(60+rank) for rank,i in enumerate(order,1)}
        with connect(self.path) as db:
            term=fts_query(query)
            hits=db.execute('SELECT p.id FROM passage_fts f JOIN passages p ON p.rowid=f.rowid WHERE passage_fts MATCH ? ORDER BY bm25(passage_fts) LIMIT 150',(term,)).fetchall() if term else []
            for rank,row in enumerate(hits,1):
                i=self.pindex[row['id']]; ranks[i]=ranks.get(i,0)+1/(60+rank)
            best=sorted(ranks,key=lambda i:(-ranks[i],-float(ps[i]),i))[:limit]
            passages=[]
            for i in best:
                p=self.passage_record(db.execute('SELECT * FROM passages WHERE id=?',(self.pids[i],)).fetchone())
                p.update(score=float(ps[i]),rank_score=ranks[i]); passages.append(p)
        exact=query.casefold(); candidates=[i for i in np.argsort(-ns) if kind is None or self.nodes[self.nids[i]]['kind']==kind]
        candidates.sort(key=lambda i:(self.nodes[self.nids[i]]['label'].casefold()!=exact,-float(ns[i])))
        results=[{**{k:self.nodes[self.nids[i]][k] for k in ['id','label','kind','mention_count','status']},
                  'score':float(ns[i]),'exact_match':self.nodes[self.nids[i]]['label'].casefold()==exact} for i in candidates[:limit]]
        places=[{**{k:self.nodes[self.nids[i]][k] for k in ['id','label','kind']},'score':float(ns[i]),'basis':'cosine'}
                for i in np.argsort(-ns) if self.nodes[self.nids[i]]['kind']=='Place'][:8]
        topics=[n['id'] for n in self.nodes.values() if n['kind']=='Concept' and n['label'].casefold()==exact]
        if topics:
            with connect(self.path) as db:
                locations=db.execute('''SELECT l.object,count(DISTINCT l.passage_id) AS sources
                    FROM links l JOIN links t ON t.subject=l.subject
                    WHERE l.predicate='occurredAt' AND t.predicate='hasTopic' AND t.object=?
                    GROUP BY l.object ORDER BY sources DESC,l.object LIMIT 8''',(topics[0],)).fetchall()
            if locations:
                places=[{**{k:self.nodes[r['object']][k] for k in ['id','label','kind']},
                         'score':float(ns[self.nindex[r['object']]]),'basis':'event_proposals',
                         'source_count':r['sources'],'status':'proposed'} for r in locations]
        return {'query':query,'nodes':results,'passages':passages,'related_places':places,
                'status':'retrieved' if passages and max(p['score'] for p in passages)>.3 else 'weak_match',
                'notice':'Retrieved excerpts, not a generated answer. Relevance is not factual confidence; proposed links are unverified.',
                'model':self.manifest['model'],'ranking':self.manifest['ranking']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--build',action='store_true'); parser.add_argument('--query')
    args=parser.parse_args()
    if args.build: build_embeddings()
    elif args.query: print(json.dumps(SearchIndex().search(args.query),indent=2))
    else: parser.error('Use --build or --query')
