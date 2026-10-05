"""Incremental ingestion with the frozen cleaning semantics and full provenance.

Only completion-manifest entries are considered. Already ingested immutable files
are not re-read per shard; full hashes are audited once before final export.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from multigen_data import common as C, processing as P


def clean_entries(directory, split, entries, sizer, batch_size=128):
    directory = Path(directory)
    C.require_unsealed(directory, split)  # Must precede even reading entries.
    task = C.read(directory / 'protocol.json')['task']
    db = P.database(directory)
    fresh = skipped = 0
    try:
        with C.timing(directory, 'cleaning', split, shard='incremental') as event:
            for entry in entries:
                path = Path(entry['path'])
                relative = path.relative_to(directory / 'raw' / split)
                if len(relative.parts) != 3 or path.suffix != '.json':
                    raise ValueError('Unexpected raw path')
                old = db.execute('SELECT sha256 FROM processed WHERE path=?', (str(path),)).fetchone()
                if old:
                    if old[0] != entry['sha256']:
                        raise ValueError('Completion manifest conflicts with processed checksum')
                    skipped += 1
                    continue
                payload = path.read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                if digest != entry['sha256']:
                    raise ValueError('Raw file differs from immutable completion manifest')
                r = json.loads(payload)
                if r['record_hash'] != C.digest({k:v for k,v in r.items() if k != 'record_hash'}):
                    raise ValueError('Corrupt raw record')
                if r['dataset'] != task or r['split'] != split:
                    raise ValueError('Raw dataset/split mismatch')
                current = r['initial']['text']
                for candidate in r['candidates']:
                    inputs = {'source':r['source'], 'current':current, 'candidate':candidate['text']}
                    pid = C.digest([task, inputs['source'], current, inputs['candidate']])
                    valid, reason = C.valid_pair(task, current, candidate, sizer, r['source'])
                    noop = current == candidate['text']
                    provenance = {k:candidate[k] for k in ('generator','temperature','slot','seed','request_hash',
                                   'request_id','raw_record_location','model_version','protocol_version')}
                    provenance.update(source_id=r['source_id'], trajectory_location=str(path.relative_to(directory)),
                                      request_record_hash=candidate['record_hash'], trajectory_sha256=digest)
                    if valid and not noop:
                        db.execute('INSERT OR IGNORE INTO pairs(id,split,source_id,cluster_id,input) VALUES(?,?,?,?,?)',
                                   (pid,split,r['source_id'],r['cluster_id'],C.canonical(inputs)))
                    db.execute('INSERT INTO occurrences VALUES(?,?,?,?,?,?,?,?)',
                        (candidate['request_hash'],pid,split,r['generator'],int(valid),int(noop),reason,C.canonical(provenance)))
                db.execute('INSERT INTO processed VALUES(?,?)', (str(path),digest))
                fresh += 1
                if fresh % batch_size == 0:
                    db.commit()
            db.commit()
            event.update(new_raw_files=fresh, reused_raw_files=skipped)
            unique = db.execute('SELECT COUNT(*) FROM pairs WHERE split=?', (split,)).fetchone()[0]
            raw = db.execute('SELECT COUNT(*) FROM occurrences WHERE split=?', (split,)).fetchone()[0]
        result = dict(at=C.utc(), split=split, unique_valid_changed_pairs=unique, raw_pairs=raw,
                      new_raw_files=fresh, reused_raw_files=skipped)
        C.dump(directory/'incremental_counts'/(split+'.json'), result)
        return result
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def published_entries(directory, split):
    C.require_unsealed(directory, split)
    for path in sorted((Path(directory)/'completed'/split).glob('*/*.json')):
        yield from C.read(path)['raw_records']


def audit_processed(directory, split):
    """One full verification before finalizing Train/Dev; never a Test cleaner."""
    directory = Path(directory)
    C.require_unsealed(directory, split)
    db = P.database(directory)
    count = 0
    try:
        with C.timing(directory, 'raw_integrity_audit', split):
            for entry in published_entries(directory, split):
                path = Path(entry['path'])
                path.relative_to(directory/'raw'/split)
                saved = db.execute('SELECT sha256 FROM processed WHERE path=?',(str(path),)).fetchone()
                if not saved or saved[0] != entry['sha256'] or C.file_hash(path) != saved[0]:
                    raise ValueError('Final raw integrity audit failed: '+str(path))
                count += 1
        report = dict(at=C.utc(), split=split, verified_raw_files=count, status='passed')
        C.dump(directory/'audits'/(split+'-incremental-integrity.json'),report)
        return report
    finally:
        db.close()
