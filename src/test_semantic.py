"""Semantic explorer regression checks. Synthetic fixtures are not evaluation data."""
import importlib.util
import json
from pathlib import Path
import tempfile
import pandas as pd


def main():
    assert importlib.util.find_spec('semantic_index'), 'The cited semantic index is not implemented'
    from semantic_index import build_rows, write_database, export_rdf
    master = pd.DataFrame([
        {'parva':'06_Bhishma','sent_id':1,'sentence':'A battle took place on the fields of Kurukshetra.'},
        {'parva':'06_Bhishma','sent_id':2,'sentence':'Dharma spoke about dharma.'},
        {'parva':'06_Bhishma','sent_id':3,'sentence':'A battle took place at Kurukshetra.'},
        {'parva':'00_Harivamsha','sent_id':1,'sentence':'Excluded appendix.'},
        {'parva':'06_Bhishma','sent_id':4,'sentence':'An unanchored statement about war.'}])
    mentions = pd.DataFrame([
        {'parva':'06_Bhishma','sent_id':s,'sentence':master.iloc[s-1].sentence,
         'surface':name,'label':label,'adhyaya_ordinal':chapter,'chapter_cited':True}
        for s,chapter,name,label in [(1,1,'battle','EVENT'),(1,1,'Kurukshetra','GPE'),
            (2,2,'Dharma','PERSON'),(2,2,'dharma','CONCEPT'),(3,3,'battle','EVENT'),(3,3,'Kurukshetra','GPE')]])
    passages,nodes,links = build_rows(master, mentions)
    assert all(p['parva'] != '00_Harivamsha' for p in passages)
    assert any(p['chapter'] is None and not p['cited'] for p in passages if 'unanchored' in p['text'])
    assert all(len(set(p['sentence_ids'])) == len(p['sentence_ids']) for p in passages)
    assert {n['kind'] for n in nodes if n['label'].lower()=='dharma'} == {'Person','Concept'}
    assert len([n for n in nodes if n['kind']=='Event']) == 2, 'Two battles cannot silently merge'
    assert any(n['kind']=='Concept' and n['label']=='war' for n in nodes)
    assert any(l['predicate']=='occurredAt' for l in links)
    assert all(l['status']=='proposed' and l['passage_id'] for l in links)
    assert build_rows(master, mentions)[0] == passages, 'Content IDs must reproduce'
    noise=mentions.iloc[[0,1,2,3,3]].copy()
    noise['surface']=['oppressed','who','Rahu','Duryodhana','Mahabharata war']
    noise['label']=['GPE','GPE','CELESTIAL','GPE','SACRED_TEXT']
    people=noise.iloc[[3,3]].copy(); people['label']='PERSON'
    _,typed,_=build_rows(master,pd.concat([mentions,noise,people],ignore_index=True))
    assert not any(n['kind']=='Place' and n['label'] in {'oppressed','who','Duryodhana'} for n in typed)
    assert any(n['kind']=='CelestialBody' and n['label']=='Rahu' for n in typed)
    assert not any(n['kind']=='TextWork' and n['label']=='Mahabharata war' for n in typed)
    with tempfile.TemporaryDirectory() as tmp:
        db=Path(tmp)/'test.sqlite'
        write_database(db,passages,nodes,links,{})
        import sqlite3
        from contextlib import closing
        # sqlite3's own context manager commits but does not close, and Windows
        # will not delete an open file when the temporary folder is cleaned up.
        with closing(sqlite3.connect(db)) as c:
            assert c.execute('select count(*) from nodes').fetchone()[0] == len(nodes)
        export_rdf(Path(tmp)/'semantic.trig',passages,nodes,links)
        from pyshacl import validate
        from rdflib import Dataset, Graph
        root=Path(__file__).resolve().parents[1]
        ds=Dataset(default_union=True).parse(Path(tmp)/'semantic.trig',format='trig')
        valid,_,report=validate(ds, shacl_graph=Graph().parse(root/'ontology/shapes.ttl'),
                                ont_graph=Graph().parse(root/'ontology/mahabharata.ttl'),inference='rdfs')
        assert valid, report
    print('Semantic index checks passed: citation boundaries, ambiguity, event identity, RDF shapes')
    assert importlib.util.find_spec('semantic_search'), 'The semantic retrieval service is not implemented'
    from semantic_search import validate_query, fts_query
    assert validate_query(' war ') == 'war'
    for bad in ('', 'x'*1001, None, ['war']):
        try: validate_query(bad)
        except ValueError: pass
        else: raise AssertionError('Invalid query accepted')
    assert fts_query('war OR " DROP TABLE nodes; --') == '"war" OR "OR" OR "DROP" OR "TABLE" OR "nodes"'
    print('Query validation checks passed')


if __name__=='__main__':
    main()
