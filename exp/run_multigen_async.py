"""Asynchronous collection scheduler around the unchanged frozen data protocol.

Only source prefixes provably required by the target are dispatched. Future raw
records stay in staging until all five models finish the same source shard.
"""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import fcntl
import math
import os
import threading
from pathlib import Path
import core
from multigen_data import common as C
from multigen_data.collect import Collector, health, verify_protocol, sealed_audit
from multigen_data import processing as P
from multigen_data.cli import status

VERSION = "async-required-prefix-v1"
PILOT = 128

def shards(total, initial=5000, size=500, pilot=PILOT):
    cursor = min(pilot,total)
    result = []
    while cursor < total:
        stop = min(total,cursor+size,initial if cursor < initial else total)
        result.append((cursor,stop))
        cursor = stop
    return result

def required_horizon(frontier, retained, total, *, target=100000, initial=5000,
                     max_pairs_per_source=20, size=500, pilot=PILOT):
    """No labels/yields are extrapolated: each future source adds at most 20 pairs.

    Every dispatched shard would still be necessary even if ALL its predecessors
    produced the theoretical maximum unique valid changed pairs.
    """
    if max_pairs_per_source <= 0:
        raise ValueError("A positive theoretical pair bound is required")
    if frontier >= min(initial,total) and retained >= target:
        return frontier
    needed = max(min(initial,total),
                 min(total,frontier+max(0,math.ceil((target-retained)/max_pairs_per_source))))
    for start,stop in shards(total,initial,size,pilot):
        if stop >= needed:
            return max(frontier,stop)
    return total

def link_immutable(source,destination,expected=None):
    source,destination = Path(source),Path(destination)
    expected = expected or C.file_hash(source)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if C.file_hash(destination) != expected:
            raise ValueError("Immutable publication conflict: "+str(destination))
        return
    if C.file_hash(source) != expected:
        raise ValueError("Staged artifact checksum mismatch: "+str(source))
    try:
        os.link(source,destination)
    except FileExistsError:
        if C.file_hash(destination) != expected:
            raise ValueError("Concurrent immutable publication conflict")

class StagedCollector(Collector):
    def __init__(self,directory,model):
        super().__init__(directory,model)
        self.published_directory = Path(directory)
        # HTTP requests always use the canonical cache and protocol hash.
        self.directory = self.published_directory/"staging"/VERSION/self.model
        self.directory.mkdir(parents=True,exist_ok=True)
        manifest_link = self.directory/"manifests"
        if not manifest_link.exists():
            manifest_link.symlink_to(self.published_directory/"manifests",target_is_directory=True)
        self.synced_events = set()
        self.sync_timing()

    def completion_path(self,base,split,start,stop):
        return Path(base)/"completed"/split/self.model/f"{start:08d}-{stop:08d}.json"

    def ready(self,split,start,stop):
        return any(self.completion_path(d,split,start,stop).exists()
                   for d in (self.published_directory,self.directory))

    def one(self,row):
        key = row["source_id"].split("/")[-1]
        published = self.published_directory/"raw"/row["split"]/self.model/key[:2]/(key+".json")
        if published.exists():
            r = C.read(published)
            if r["record_hash"] != C.digest({k:v for k,v in r.items() if k!="record_hash"}):
                raise ValueError("Published raw record checksum mismatch")
            return str(published)
        return super().one(row)

    def sync_timing(self):
        for path in (self.directory/"timing_events").glob("*.json"):
            if path.name not in self.synced_events:
                target = self.published_directory/"timing_events"/(VERSION+"-"+self.model+"-"+path.name)
                link_immutable(path,target)
                self.synced_events.add(path.name)

    def run(self,split,start,stop):
        published = self.completion_path(self.published_directory,split,start,stop)
        if published.exists():
            return C.read(published)
        C.dump(self.published_directory/"async_progress"/(self.model+".json"),
               {"stage":"generating","split":split,"source_start":start,"source_stop":stop,"at":C.utc()})
        try:
            result = super().run(split,start,stop)
        finally:
            self.sync_timing()
        C.dump(self.published_directory/"async_progress"/(self.model+".json"),
               {"stage":"shard_generated","split":split,"source_start":start,"source_stop":stop,"at":C.utc()})
        return result

    def publish(self,split,start,stop):
        published = self.completion_path(self.published_directory,split,start,stop)
        if published.exists():
            return C.read(published)
        staged = self.completion_path(self.directory,split,start,stop)
        record = C.read(staged)
        if (record["start"],record["stop"],record["split"],record["generator"]) != (start,stop,split,self.model):
            raise ValueError("Staged completion identity mismatch")
        if record["sources"] != stop-start or len(record["raw_records"]) != stop-start:
            raise ValueError("Incomplete source coverage")
        output = dict(record,raw_records=[])
        for entry in record["raw_records"]:
            source = Path(entry["path"])
            allowed = (self.directory/"raw"/split/self.model,
                       self.published_directory/"raw"/split/self.model)
            relative = None
            for base in allowed:
                try:
                    relative = source.relative_to(base)
                    break
                except ValueError:
                    pass
            if relative is None or len(relative.parts) != 2:
                raise ValueError("Unexpected staged raw path")
            target = self.published_directory/"raw"/split/self.model/relative
            link_immutable(source,target,entry["sha256"])
            output["raw_records"].append({"path":str(target),"sha256":entry["sha256"]})
        C.dump(published,output,immutable=True)
        return output

class Dispatch:
    def __init__(self,limit):
        self.limit = limit
        self.cv = threading.Condition()
        self.stopped = False
        self.errors = []

    def allow(self,stop):
        with self.cv:
            self.cv.wait_for(lambda:self.stopped or stop<=self.limit)
            return not self.stopped

    def advance(self,limit):
        with self.cv:
            if limit < self.limit:
                raise ValueError("Guaranteed required prefix unexpectedly decreased")
            self.limit = limit
            self.cv.notify_all()

    def notify(self):
        with self.cv:
            self.cv.notify_all()

    def fail(self,error):
        with self.cv:
            self.errors.append(error)
            self.stopped = True
            self.cv.notify_all()

    def check(self):
        if self.errors:
            raise RuntimeError("A generation worker failed") from self.errors[0]

    def stop(self):
        with self.cv:
            self.stopped = True
            self.cv.notify_all()

def common_prefix(directory,models,ranges):
    result = PILOT
    for start,stop in ranges:
        if not all((Path(directory)/"completed/train"/m/f"{start:08d}-{stop:08d}.json").exists()
                   for m in models):
            break
        result = stop
    return result

def train(directory,collectors,sizer,clean_fn=P.clean,timing_fn=C.summarize_timing):
    directory = Path(directory)
    complete = directory/"train_generation_complete.json"
    if complete.exists():
        return C.read(complete)
    p = C.read(directory/"protocol.json")
    total = p["pool_counts"]["train"]
    initial,target,size = p["initial_train_sources"],p["target_unique_valid_changed"],p["shard_sources"]
    ranges = shards(total,initial,size)
    frontier = common_prefix(directory,[c.model for c in collectors],ranges)
    # Complete any shard already dispatched by the preceding synchronous process
    # before cleaning; it may have partially published canonical raw records.
    recovery = frontier
    for c in collectors:
        for path in (directory/"completed/train"/c.model).glob("*.json"):
            recovery = max(recovery,int(path.stem.split("-")[1]))
    progress = C.read(directory/"progress.json") if (directory/"progress.json").exists() else {}
    if progress.get("stage") == "train_generation":
        recovery = max(recovery,progress.get("source_stop",frontier))
    saved = C.read(directory/"async_dispatch.json") if (directory/"async_dispatch.json").exists() else {}
    initial_limit = max(min(initial,total),recovery,saved.get("allocated_source_prefix",0))
    dispatch = Dispatch(initial_limit)
    def worker(c):
        try:
            for start,stop in ranges:
                if not dispatch.allow(stop):
                    return
                c.run("train",start,stop)
                dispatch.notify()
        except BaseException as e:
            dispatch.fail(e)
    threads = [threading.Thread(target=worker,args=(c,),name=c.model) for c in collectors]
    for t in threads:
        t.start()
    retained = 0
    def process_prefix():
        nonlocal retained
        summary = clean_fn(directory,"train",sizer)
        retained = summary["unique_valid_changed_pairs"]
        limit = required_horizon(frontier,retained,total,target=target,initial=initial,
             max_pairs_per_source=len(collectors)*p["candidates"],size=size)
        if limit < dispatch.limit:
            raise ValueError("More sources were dispatched than can be proved necessary")
        record = {"stage":"train_generation_async","completed_source_prefix":frontier,
            "allocated_source_prefix":limit,"unique_valid_changed_pairs":retained,"at":C.utc(),
            "scheduler":VERSION,"stop_check_uses":"five-model complete source prefix only"}
        # Persist authorization before waking workers, including after a restart.
        C.dump(directory/"async_dispatch.json",record)
        C.dump(directory/"progress.json",record)
        dispatch.advance(limit)
        timing_fn(directory)
    try:
        if frontier >= recovery:
            process_prefix()
        for start,stop in ranges:
            if stop <= frontier:
                continue
            if frontier >= min(initial,total) and retained >= target:
                break
            with dispatch.cv:
                while not all(c.ready("train",start,stop) for c in collectors):
                    dispatch.check()
                    dispatch.cv.wait(timeout=1)
                dispatch.check()
            for c in collectors:
                c.publish("train",start,stop)
            frontier = stop
            if frontier >= recovery:
                process_prefix()
        dispatch.check()
        if frontier != dispatch.limit:
            raise ValueError("Final model source coverage is not aligned")
    finally:
        dispatch.stop()
        for t in threads:
            t.join()
    dispatch.check()
    record = {"train_sources":frontier,"unique_valid_changed_pairs":retained,
        "target_met":retained>=target,"pool_exhausted":frontier==total,
        "scheduler":VERSION,"at":C.utc()}
    C.dump(complete,record,immutable=True)
    return record

def holdout_worker(collector):
    # Split generation depends solely on its frozen manifest and request parameters.
    # No reference scoring or data cleaning is done by these workers.
    for split in ("dev","test"):
        collector.run(split,0,500)
        collector.publish(split,0,500)

def await_generated(collectors,pending,split):
    wake = threading.Event()
    while not all(c.completion_path(c.published_directory,split,0,500).exists() for c in collectors):
        for job in pending:
            if job.done():
                job.result()
        wake.wait(1)

def run(output):
    import torch
    torch.set_num_threads(4)
    output = Path(output)
    amendment = C.read(output/"scheduling_amendments"/(VERSION+".json"))
    if amendment["scheduler_sha256"] != C.file_hash(__file__):
        raise ValueError("Scheduling implementation changed after freeze")
    sizer = C.InputSizer()
    for task in C.TASKS:
        d = output/task
        if (d/"collection_complete.json").exists():
            continue
        if not (d/"pilot_complete.json").exists():
            raise ValueError("Finish and audit all four pilots before asynchronous collection")
        p = verify_protocol(d,full=True)
        if amendment["protocol_hashes"][task] != C.file_hash(d/"protocol.json"):
            raise ValueError("Original data protocol changed")
        C.dump(d/"service_health.json",{"at":C.utc(),"checks":health(p)})
        # Initialize HF tokenizers sequentially (its lazy module imports are not thread safe).
        collectors = [StagedCollector(d,m) for m in p["models"]]
        report = train(d,collectors,sizer)
        # Overlap reference-blind holdout generation with Train scoring/export.
        with futures.ThreadPoolExecutor(len(collectors)) as pool:
            pending = [pool.submit(holdout_worker,c) for c in collectors]
            C.dump(d/"progress.json",{"stage":"train_scoring_with_holdout_generation",**report})
            P.score(d,"train")
            P.export(d,"train")
            await_generated(collectors,pending,"dev")
            C.dump(d/"progress.json",{"stage":"dev_processing","at":C.utc()})
            P.clean(d,"dev",sizer)
            P.score(d,"dev")
            P.export(d,"dev")
            for job in pending:
                job.result()
        sealed_audit(d)
        report.update(task=task,test="sealed",finished_at_utc=C.utc())
        C.dump(d/"collection_complete.json",report,immutable=True)
        C.dump(d/"progress.json",{"stage":"complete" if report["target_met"] else "source_pool_exhausted",**report})
        C.summarize_timing(d)
        print(C.canonical(report),flush=True)
        del collectors
    C.dump(output/"campaign_complete.json",status(output),immutable=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=C.OUT)
    args = parser.parse_args()
    with (args.output/"campaign.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(args.output)
        except BaseException as e:
            C.dump(args.output/"async_failure.json",{"at":C.utc(),"type":type(e).__name__,"message":str(e)})
            raise

if __name__ == "__main__":
    main()
