"""Cross-dataset GPU ready queue, incremental CPU cleaning, separate finalization.

Generation remains reference-blind and uses the frozen v1 collectors/cache.
A queue item always belongs to exactly one task, split, generator, and source shard.
"""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import fcntl
import threading
import time
from pathlib import Path
import core
from multigen_data import common as C, processing as P
from multigen_data.collect import verify_protocol, health, sealed_audit
from multigen_data.cli import status
from run_multigen_async import StagedCollector, shards, common_prefix, required_horizon
from multigen_streaming import clean_entries, published_entries, audit_processed

VERSION = 'cross-task-incremental-v2'


class TaskState:
    def __init__(self, directory, protocol, collectors):
        self.directory = Path(directory)
        self.protocol = protocol
        self.task = protocol['task']
        self.collectors = {c.model:c for c in collectors}
        self.total = protocol['pool_counts']['train']
        self.ranges = shards(self.total, protocol['initial_train_sources'], protocol['shard_sources'])
        saved = C.read(self.directory/'async_dispatch.json') if (self.directory/'async_dispatch.json').exists() else {}
        self.frontier = common_prefix(self.directory,self.collectors,self.ranges)
        self.limit = max(min(protocol['initial_train_sources'],self.total), self.frontier,
                         saved.get('allocated_source_prefix',0))
        self.train_done = (self.directory/'train_generation_complete.json').exists()
        self.finished = (self.directory/'collection_complete.json').exists()
        self.claimed = set()
        self.next_index = {m:0 for m in self.collectors}

    def next_job(self, model):
        if self.finished:
            return None
        c = self.collectors[model]
        i = self.next_index[model]
        while i < len(self.ranges) and c.ready('train',*self.ranges[i]):
            i += 1
        self.next_index[model] = i
        if not self.train_done and i < len(self.ranges):
            start,stop = self.ranges[i]
            key = (model,'train',start,stop)
            if stop <= self.limit and key not in self.claimed:
                self.claimed.add(key)
                return ('train',start,stop)
        # Holdouts are independently frozen. Generation only: no Test cleaning.
        for split in ('dev','test'):
            key = (model,split,0,500)
            if key not in self.claimed and not c.ready(split,0,500):
                self.claimed.add(key)
                return (split,0,500)
        return None


class Hub:
    def __init__(self, states):
        self.states = states
        self.cv = threading.Condition()
        self.error = None
        self.stopped = False

    def claim(self, model):
        with self.cv:
            while not self.stopped:
                for state in self.states:
                    job = state.next_job(model)
                    if job:
                        return state,job
                self.cv.wait(timeout=1)
            return None

    def notify(self):
        with self.cv:
            self.cv.notify_all()

    def fail(self, exc):
        with self.cv:
            self.error = exc
            self.stopped = True
            self.cv.notify_all()

    def check(self):
        if self.error:
            raise RuntimeError('Pipeline worker failed') from self.error

    def stop(self):
        with self.cv:
            self.stopped = True
            self.cv.notify_all()


def generation_worker(hub, model, output):
    try:
        while (claim := hub.claim(model)) is not None:
            state,(split,start,stop) = claim
            C.dump(Path(output)/'pipeline_workers'/(model+'.json'),dict(at=C.utc(),stage='generating',
                task=state.task,split=split,start=start,stop=stop,scheduler=VERSION))
            state.collectors[model].run(split,start,stop)
            C.dump(Path(output)/'pipeline_workers'/(model+'.json'),dict(at=C.utc(),stage='job_complete',
                task=state.task,split=split,start=start,stop=stop,scheduler=VERSION))
            hub.notify()
    except BaseException as e:
        hub.fail(e)
        raise


def wait_and_publish(state, hub, split, start, stop):
    with hub.cv:
        while not all(c.ready(split,start,stop) for c in state.collectors.values()):
            hub.check()
            if hub.stopped:
                raise RuntimeError('Pipeline stopped')
            hub.cv.wait(timeout=1)
        hub.check()
    entries = []
    for c in state.collectors.values():
        entries.extend(c.publish(split,start,stop)['raw_records'])
    return entries


def advance(state, hub, retained):
    p = state.protocol
    limit = required_horizon(state.frontier,retained,state.total,target=p['target_unique_valid_changed'],
        initial=p['initial_train_sources'],max_pairs_per_source=len(state.collectors)*p['candidates'],
        size=p['shard_sources'])
    if limit < state.limit:
        raise ValueError('Previously dispatched sources exceed guaranteed required prefix')
    record = dict(at=C.utc(),stage='train_generation_async',scheduler=VERSION,
        completed_source_prefix=state.frontier,allocated_source_prefix=limit,
        unique_valid_changed_pairs=retained,stop_check_uses='five-model complete source prefix only')
    # Fast counts wake generation BEFORE expensive summaries and timing reports.
    C.dump(state.directory/'async_dispatch.json',record)
    C.dump(state.directory/'progress.json',record)
    with hub.cv:
        state.limit = limit
        hub.cv.notify_all()


def train_incremental(state, hub, sizer):
    marker = state.directory/'train_generation_complete.json'
    if marker.exists():
        with hub.cv:
            state.train_done = True
            hub.cv.notify_all()
        return C.read(marker)
    # Recover all canonical completed batches, including interrupted ingestion.
    summary = clean_entries(state.directory,'train',published_entries(state.directory,'train'),sizer)
    retained = summary['unique_valid_changed_pairs']
    advance(state,hub,retained)
    p = state.protocol
    for start,stop in state.ranges:
        if stop <= state.frontier:
            continue
        if state.frontier >= min(p['initial_train_sources'],state.total) and retained >= p['target_unique_valid_changed']:
            break
        entries = wait_and_publish(state,hub,'train',start,stop)
        summary = clean_entries(state.directory,'train',entries,sizer)
        state.frontier = stop
        retained = summary['unique_valid_changed_pairs']
        advance(state,hub,retained)
    hub.check()
    if state.frontier != state.limit:
        raise ValueError('Generators did not end on the same required source prefix')
    report = dict(at=C.utc(),train_sources=state.frontier,unique_valid_changed_pairs=retained,
        target_met=retained>=p['target_unique_valid_changed'],pool_exhausted=state.frontier==state.total,scheduler=VERSION)
    C.dump(marker,report,immutable=True)
    with hub.cv:
        state.train_done = True
        hub.cv.notify_all()
    return report


def process_task(state, hub, sizer, comet_lock):
    try:
        if state.finished:
            return
        d = state.directory
        report = train_incremental(state,hub,sizer)
        C.dump(d/'progress.json',dict(stage='train_scoring_with_cross_task_generation',**report))
        if not (d/'exports/train.manifest.json').exists():
            audit_processed(d,'train')
            if state.task.startswith('wmt19'):
                with comet_lock:
                    P.score(d,'train')
            else:
                P.score(d,'train')
            P.export(d,'train')
        entries = wait_and_publish(state,hub,'dev',0,500)
        C.dump(d/'progress.json',dict(at=C.utc(),stage='dev_processing',scheduler=VERSION))
        if not (d/'exports/dev.manifest.json').exists():
            clean_entries(d,'dev',entries,sizer)
            audit_processed(d,'dev')
            if state.task.startswith('wmt19'):
                with comet_lock:
                    P.score(d,'dev')
            else:
                P.score(d,'dev')
            P.export(d,'dev')
        wait_and_publish(state,hub,'test',0,500)
        sealed_audit(d)
        report.update(task=state.task,test='sealed',finished_at_utc=C.utc())
        C.dump(d/'collection_complete.json',report,immutable=True)
        C.dump(d/'progress.json',dict(stage='complete' if report['target_met'] else 'source_pool_exhausted',**report))
        C.summarize_timing(d)
        with hub.cv:
            state.finished = True
            hub.cv.notify_all()
        print(C.canonical(report),flush=True)
    except BaseException as e:
        hub.fail(e)
        raise


def run(output):
    import torch
    torch.set_num_threads(4)
    from core.scoring import Scorer
    output = Path(output)
    amendment = C.read(output/'scheduling_amendments'/(VERSION+'.json'))
    for path,expected in amendment['implementation_hashes'].items():
        if C.file_hash(path) != expected:
            raise ValueError('Pipeline implementation changed after amendment')
    states, sizers = [], []
    for task in C.TASKS:
        d = output/task
        if (d/'collection_complete.json').exists():
            continue
        if not (d/'pilot_complete.json').exists():
            raise ValueError('Pilot must be completed before bulk collection')
        p = verify_protocol(d,full=True)
        if C.file_hash(d/'protocol.json') != amendment['protocol_hashes'][task]:
            raise ValueError('Frozen data protocol changed')
        C.dump(d/'service_health.json',dict(at=C.utc(),checks=health(p)))
        # Preload lazy Python/HF imports sequentially before starting worker threads.
        collectors = [StagedCollector(d,m) for m in p['models']]
        states.append(TaskState(d,p,collectors))
        sizers.append(C.InputSizer())
        if not task.startswith('wmt19'):
            Scorer(task)
    if not states:
        C.dump(output/'campaign_complete.json',status(output),immutable=True)
        return
    hub = Hub(states)
    models = list(states[0].collectors)
    if any(set(s.collectors) != set(models) for s in states):
        raise ValueError('All tasks must share the five frozen generators')
    comet_lock = threading.Lock()
    gpu = futures.ThreadPoolExecutor(len(models),thread_name_prefix='gpu')
    cpu = futures.ThreadPoolExecutor(len(states),thread_name_prefix='cpu')
    generation = [gpu.submit(generation_worker,hub,m,output) for m in models]
    processing = [cpu.submit(process_task,s,hub,z,comet_lock) for s,z in zip(states,sizers)]
    try:
        for job in futures.as_completed(processing):
            job.result()
        hub.check()
        hub.stop()
        for job in generation:
            job.result()
    finally:
        hub.stop()
        cpu.shutdown(wait=True)
        gpu.shutdown(wait=True)
    C.dump(output/'campaign_complete.json',status(output),immutable=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=C.OUT)
    args = parser.parse_args()
    with (args.output/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(args.output)
        except BaseException as e:
            C.dump(args.output/'async_failure.json',dict(at=C.utc(),type=type(e).__name__,message=str(e),scheduler=VERSION))
            raise


if __name__ == '__main__':
    main()
