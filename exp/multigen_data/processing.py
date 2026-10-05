"""Train/Dev cleaning and labels; Test entry points fail closed before all I/O."""
from __future__ import annotations
import json
import math
import os
import sqlite3
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from . import common as C

def database(directory):
    db = sqlite3.connect(Path(directory) / 'pairs.sqlite3', timeout=60)
    db.execute('PRAGMA journal_mode=DELETE')
    db.executescript("""
    CREATE TABLE IF NOT EXISTS pairs(
      id TEXT PRIMARY KEY, split TEXT, source_id TEXT, cluster_id TEXT, input TEXT,
      q_current REAL, q_candidate REAL, delta REAL, label TEXT);
    CREATE TABLE IF NOT EXISTS occurrences(
      request_hash TEXT PRIMARY KEY, pair_id TEXT, split TEXT, generator TEXT,
      valid INTEGER, noop INTEGER, reason TEXT, provenance TEXT);
    CREATE TABLE IF NOT EXISTS processed(path TEXT PRIMARY KEY, sha256 TEXT);
    CREATE TABLE IF NOT EXISTS score_cache(key TEXT PRIMARY KEY, score REAL);
    """)
    return db

def clean(directory, split, sizer=None):
    directory = Path(directory)
    C.require_unsealed(directory, split)
    task = C.read(directory / 'protocol.json')['task']
    sizer = sizer or C.InputSizer()
    db = database(directory)
    with C.timing(directory, 'cleaning', split):
        for path in sorted((directory / 'raw' / split).glob('*/*/*.json')):
            sha = C.file_hash(path)
            old = db.execute('SELECT sha256 FROM processed WHERE path=?',(str(path),)).fetchone()
            if old:
                if old[0] != sha:
                    raise ValueError('Immutable raw data changed')
                continue
            r = C.read(path)
            if r['record_hash'] != C.digest({k:v for k,v in r.items() if k != 'record_hash'}):
                raise ValueError('Corrupt raw record')
            current = r['initial']['text']
            for candidate in r['candidates']:
                inputs = {'source':r['source'], 'current':current, 'candidate':candidate['text']}
                pid = C.digest([task, inputs['source'], current, inputs['candidate']])
                valid, reason = C.valid_pair(task, current, candidate, sizer, r['source'])
                noop = current == candidate['text']
                provenance = {k:candidate[k] for k in ('generator','temperature','slot','seed','request_hash',
                               'request_id','raw_record_location','model_version','protocol_version')}
                provenance.update(source_id=r['source_id'], trajectory_location=str(path.relative_to(directory)),
                                  request_record_hash=candidate['record_hash'], trajectory_sha256=sha)
                if valid and not noop:
                    db.execute('INSERT OR IGNORE INTO pairs(id,split,source_id,cluster_id,input) VALUES(?,?,?,?,?)',
                               (pid,split,r['source_id'],r['cluster_id'],C.canonical(inputs)))
                db.execute('INSERT INTO occurrences VALUES(?,?,?,?,?,?,?,?)',
                   (candidate['request_hash'],pid,split,r['generator'],int(valid),int(noop),reason,C.canonical(provenance)))
            db.execute('INSERT INTO processed VALUES(?,?)',(str(path),sha))
            db.commit()
    report = summarize(directory, split, db)
    db.close()
    return report

def summarize(directory, split, db=None):
    C.require_unsealed(directory, split)
    own = db is None
    db = db or database(directory)
    raw = db.execute('SELECT COUNT(*),COALESCE(SUM(valid),0),COALESCE(SUM(noop),0),COALESCE(SUM(valid AND NOT noop),0) FROM occurrences WHERE split=?',(split,)).fetchone()
    unique = db.execute('SELECT COUNT(*) FROM pairs WHERE split=?',(split,)).fetchone()[0]
    labels = dict(db.execute('SELECT label,COUNT(*) FROM pairs WHERE split=? AND label IS NOT NULL GROUP BY label',(split,)))
    sources = db.execute('SELECT COUNT(DISTINCT cluster_id) FROM pairs WHERE split=?',(split,)).fetchone()[0]
    report = {'split':split, 'raw_pairs':raw[0], 'valid_pairs':raw[1], 'no_op_pairs':raw[2],
              'valid_changed_occurrences':raw[3], 'unique_valid_changed_pairs':unique,
              'duplicate_valid_changed_occurrences':raw[3]-unique, 'source_clusters':sources,
              'labels':labels, 'unscored_pairs':unique-sum(labels.values()),
              'invalid_reasons':dict(db.execute('SELECT reason,COUNT(*) FROM occurrences WHERE split=? AND NOT valid GROUP BY reason',(split,))),
              'by_generator':{}, 'at':C.utc()}
    for (gen,) in db.execute('SELECT DISTINCT generator FROM occurrences WHERE split=?',(split,)).fetchall():
        counts = db.execute('SELECT COUNT(*),SUM(valid),SUM(noop),SUM(valid AND NOT noop) FROM occurrences WHERE split=? AND generator=?',(split,gen)).fetchone()
        report['by_generator'][gen] = {'raw_pairs':counts[0], 'valid_pairs':counts[1], 'no_op_pairs':counts[2],
           'valid_changed_occurrences':counts[3],
           'unique_changed_pairs':db.execute('SELECT COUNT(DISTINCT pair_id) FROM occurrences WHERE split=? AND generator=? AND valid AND NOT noop',(split,gen)).fetchone()[0],
           'labels':dict(db.execute('SELECT p.label,COUNT(DISTINCT p.id) FROM pairs p JOIN occurrences o ON p.id=o.pair_id WHERE p.split=? AND o.generator=? AND p.label IS NOT NULL AND o.valid AND NOT o.noop GROUP BY p.label',(split,gen)))}
    report['label_ratios'] = {k:v/max(1,sum(labels.values())) for k,v in labels.items()}
    report['by_temperature'] = {}
    for provenance, label in db.execute('SELECT o.provenance,p.label FROM occurrences o LEFT JOIN pairs p ON p.id=o.pair_id WHERE o.split=?',(split,)):
        temperature = str(json.loads(provenance)['temperature'])
        bucket = report['by_temperature'].setdefault(temperature,{'raw_occurrences':0,'scored_labels':{}})
        bucket['raw_occurrences'] += 1
        if label is not None:
            bucket['scored_labels'][label] = bucket['scored_labels'].get(label,0)+1
    for gen, bucket in report['by_generator'].items():
        bucket['label_ratios'] = {k:v/max(1,sum(bucket['labels'].values())) for k,v in bucket['labels'].items()}
        # Generator membership is a set of pair IDs, independent of occurrence count.
        delta_rows = [r[1] for r in db.execute('SELECT DISTINCT p.id,p.delta FROM pairs p JOIN occurrences o ON p.id=o.pair_id WHERE p.split=? AND o.generator=? AND p.delta IS NOT NULL',(split,gen))]
        if delta_rows:
            import numpy as np
            bucket['delta_quantiles'] = list(map(float,np.quantile(delta_rows,[0,.1,.5,.9,1])))
    values = [x[0] for x in db.execute('SELECT delta FROM pairs WHERE split=? AND delta IS NOT NULL',(split,))]
    if values:
        import numpy as np
        report['delta_quantiles'] = dict(zip(['min','p01','p10','p50','p90','p99','max'],
                            map(float,np.quantile(values,[0,.01,.1,.5,.9,.99,1]))))
    C.dump(Path(directory) / 'statistics' / (split+'.json'), report)
    if own:
        db.close()
    return report

def metric_scores(task, examples, cfg, logpath):
    if not task.startswith('wmt19'):
        from core.scoring import Scorer
        scorer = Scorer(task)
        return [scorer.primary(x['ref'], x['mt']) for x in examples]
    with tempfile.TemporaryDirectory(prefix='comet-multigen-', dir='/mnt/huawei/ymb/.tmp') as tmp:
        inp, out = Path(tmp)/'input.json', Path(tmp)/'output.json'
        C.dump(inp, examples)
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   CUDA_VISIBLE_DEVICES=cfg.get('visible_gpu','') if cfg['device']=='cuda' else '',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
        command = [cfg['python'],str(C.ROOT/'core/comet_feedback_worker.py'),'--input',str(inp),
                   '--output',str(out),'--checkpoint',cfg['checkpoint'],'--device',cfg['device'],
                   '--batch-size',str(cfg['batch_size'])]
        with Path(logpath).open('a') as log:
            done = subprocess.run(command, env=env, stdout=log, stderr=log)
        if done.returncode:
            raise RuntimeError('COMET scoring failed; no metric fallback. See '+str(logpath))
        scores = C.read(out)
        if len(scores) != len(examples) or any(not math.isfinite(s) for s in scores):
            raise ValueError('Invalid metric output')
        return scores

def score(directory, split, chunk_size=2048):
    directory = Path(directory)
    C.require_unsealed(directory, split)
    p = C.read(directory/'protocol.json')
    references = {r['source_id']:r['reference'] for r in C.jsonl(directory/'references'/(split+'.jsonl'))}
    db = database(directory)
    while True:
        rows = db.execute('SELECT id,source_id,input FROM pairs WHERE split=? AND label IS NULL ORDER BY id LIMIT ?',
                          (split,chunk_size)).fetchall()
        if not rows:
            break
        examples, key_pairs = {}, []
        for pid, sid, value in rows:
            x = json.loads(value)
            keys = []
            for field in ('current','candidate'):
                item = {'src':x['source'],'mt':x[field],'ref':references[sid]}
                key = C.digest([p['feedback'],item])
                keys.append(key)
                if not db.execute('SELECT 1 FROM score_cache WHERE key=?',(key,)).fetchone():
                    examples[key] = item
            key_pairs.append(keys)
        if examples:
            with C.timing(directory,'scoring',split,shard=rows[0][0]) as event:
                scores = metric_scores(p['task'],list(examples.values()),p['feedback'],directory/'scoring.log')
                db.executemany('INSERT OR IGNORE INTO score_cache VALUES(?,?)',zip(examples,scores))
                db.commit()
                event['texts_scored'] = len(scores)
        for (pid, sid, x), (a,b) in zip(rows,key_pairs):
            before = db.execute('SELECT score FROM score_cache WHERE key=?',(a,)).fetchone()[0]
            after = db.execute('SELECT score FROM score_cache WHERE key=?',(b,)).fetchone()[0]
            label, delta = C.label_scores(before,after)
            db.execute('UPDATE pairs SET q_current=?,q_candidate=?,delta=?,label=? WHERE id=?',
                       (before,after,delta,label,pid))
        db.commit()
        C.dump(directory/'scoring_status.json',{'split':split,'remaining':db.execute(
                   'SELECT COUNT(*) FROM pairs WHERE split=? AND label IS NULL',(split,)).fetchone()[0],'at':C.utc()})
    report = summarize(directory,split,db)
    db.close()
    return report

def export(directory, split):
    directory = Path(directory)
    C.require_unsealed(directory, split)
    db = database(directory)
    if db.execute('SELECT COUNT(*) FROM pairs WHERE split=? AND label IS NULL',(split,)).fetchone()[0]:
        raise ValueError('Unscored pairs cannot be exported as Reject')
    output = directory/'exports'/(split+'.jsonl')
    if output.exists():
        return {'path':str(output),'sha256':C.file_hash(output)}
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix('.jsonl.part')
    p = C.read(directory/'protocol.json')
    with C.timing(directory,'export',split):
        with temporary.open('w') as f:
            for pid, sid, cid, value, before, after, delta, label in db.execute(
               'SELECT id,source_id,cluster_id,input,q_current,q_candidate,delta,label FROM pairs WHERE split=? ORDER BY id',(split,)):
                provenance = [json.loads(r[0]) for r in db.execute(
                    'SELECT provenance FROM occurrences WHERE pair_id=? AND valid AND NOT noop ORDER BY request_hash',(pid,))]
                row = {'id':pid,'input':json.loads(value),'label':label,'metadata':{
                    'task':p['task'],'split':split,'source_id':sid,'cluster_id':cid,
                    'input_template_version':C.INPUT_VERSION,'protocol_sha256':C.file_hash(directory/'protocol.json'),
                    'q_current':before,'q_candidate':after,'delta':delta,'feedback':p['feedback'],
                    'provenance':provenance}}
                f.write(C.canonical(row)+'\n')
        os.link(temporary,output)
        temporary.unlink()
    db.close()
    result = {'path':str(output),'sha256':C.file_hash(output)}
    C.dump(directory/'exports'/(split+'.manifest.json'),result,immutable=True)
    return result

def evaluation_rows(directory, split):
    C.require_unsealed(directory, split)
    return list(C.jsonl(Path(directory)/'exports'/(split+'.jsonl')))

def clustered_bootstrap(rows, predictions, draws=2000, seed=C.MASTER_SEED):
    """All models share the same source resampling, suitable for paired differences."""
    import numpy as np
    ids = sorted({r['metadata']['cluster_id'] for r in rows})
    if not ids:
        raise ValueError('No source clusters')
    lookup = {s:i for i,s in enumerate(ids)}
    index = np.array([lookup[r['metadata']['cluster_id']] for r in rows])
    truth = np.array([r['label'] for r in rows])
    delta = np.array([r['metadata']['delta'] for r in rows])
    sample = np.random.default_rng(seed).integers(0,len(ids),(draws,len(ids)))
    denominator = np.bincount(index,minlength=len(ids))[sample].sum(1)
    result = {}
    for name, decisions in predictions.items():
        if len(decisions) != len(rows) or any(d not in ('Accept','Reject') for d in decisions):
            raise ValueError('Predictions must align with the exact pair list')
        accepted = np.array(decisions)=='Accept'
        correct = np.array(decisions)==truth
        counts = {}
        for label, mask in {'correct':correct,'gain':accepted*delta,
            'tp':accepted & (truth=='Accept'),'fp':accepted & (truth=='Reject'),
            'fn':(~accepted) & (truth=='Accept'),'tn':(~accepted) & (truth=='Reject')}.items():
            counts[label] = np.bincount(index,weights=mask,minlength=len(ids))[sample].sum(1)
        safe = lambda a,b: np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b!=0)
        tp,fp,fn,tn = (counts[k] for k in ('tp','fp','fn','tn'))
        result[name] = {'accuracy':counts['correct']/denominator,
           'net_delta_per_pair':counts['gain']/denominator,
           'accept_precision':safe(tp,tp+fp),'accept_recall':safe(tp,tp+fn),
           'macro_f1':(safe(2*tp,2*tp+fp+fn)+safe(2*tn,2*tn+fp+fn))/2}
    return result
