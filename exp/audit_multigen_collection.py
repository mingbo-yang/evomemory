"""Read-only contract audit of collected Train/Dev trajectories and source manifests.

Test raw data, labels and cleaned-pair statistics are deliberately inaccessible here.
"""
from __future__ import annotations
import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import core
from baseline_core.tasks import get_adapter
from baseline_core.types import TaskExample
from core.pipeline import build_refine_prompt
from core.refinement_instructions import INSTRUCTIONS
from multigen_data import common as C
from multigen_data.collect import Collector, verify_protocol
from multigen_data.processing import clean, score, export, evaluation_rows

def check_record(record):
    assert record["record_hash"] == C.digest({k:v for k,v in record.items() if k != "record_hash"}), "Record checksum failed"

def audit_pilot(directory, count=128):
    directory = Path(directory)
    p = verify_protocol(directory, full=True)
    task = p["task"]
    manifest = list(itertools.islice(C.jsonl(directory/"manifests/train.sources.jsonl"), count))
    by_id = {r["source_id"]:r for r in manifest}
    adapter = get_adapter(task)
    results = {}
    for model in p["models"]:
        name = model["name"]
        completion_path = directory/"completed/train"/name/f"{0:08d}-{count:08d}.json"
        completion = C.read(completion_path)
        assert completion["sources"] == count
        ids = set()
        counters = Counter()
        for entry in completion["raw_records"]:
            path = Path(entry["path"])
            assert C.file_hash(path) == entry["sha256"]
            r = C.read(path)
            check_record(r)
            m = by_id[r["source_id"]]
            assert r["source"] == m["source"] and r["cluster_id"] == m["cluster_id"]
            assert r["generator"] == name and r["split"] == "train" and r["dataset"] == task
            ids.add(r["source_id"])
            example = TaskExample(index=m["position"],source=m["source"],reference="",task=task)
            initial_messages = [
                {"role":"system","content":adapter.system_prompt()},
                {"role":"user","content":adapter.initial_prompt(example)}]
            assert r["initial"]["request"]["messages"] == initial_messages
            assert r["initial"]["seed"] == C.seed_for(task,"train",r["source_id"],name,"initial")
            assert r["initial"]["temperature"] == .1
            counters["source_only_initial_requests"] += 1
            if r["status"] == "complete":
                candidates = r["candidates"]
                assert len(candidates) == 4
                assert len({x["seed"] for x in candidates}) == 4
                assert [x["seed"] for x in candidates] == C.candidate_seeds(task,"train",r["source_id"],name)
                assert [x["temperature"] for x in candidates] == [.1,.1,.7,.7]
                assert r["retrieval"]["context_hash"] == C.digest(r["retrieval"]["text"])
                revision_messages = [
                    {"role":"system","content":adapter.system_prompt()},
                    {"role":"user","content":build_refine_prompt(adapter,example,r["initial"]["text"],
                        INSTRUCTIONS[task],experience_block=r["retrieval"]["text"],renderer="v2")}]
                assert all(x["request"]["messages"] == revision_messages for x in candidates)
                counters["fixed_current_fixed_context_four_seed_groups"] += 1
            else:
                assert r["status"] == "initial_invalid" and not r["candidates"] and r["retrieval"] is None
                counters["invalid_initial_no_revision_groups"] += 1
            for g in [r["initial"]]+r["candidates"]:
                check_record(g)
                assert g["request_hash"] == C.digest(g["request"])
                assert g["source_id"] == r["source_id"] and g["generator"] == name
                saved = C.read(directory/g["raw_record_location"])
                assert saved == g, "Provenance does not resolve to the exact immutable request"
                counters["resolvable_request_occurrences"] += 1
        assert ids == set(by_id)
        # Exercise the real completed-shard resume path: no generator or retrieval exists on this object.
        resumed = Collector.run(SimpleNamespace(directory=directory,model=name),"train",0,count)
        assert resumed == completion
        results[name] = dict(counters, completed_shard_resume_without_model_calls=True)
    denied = []
    for fn in (clean,score,export,evaluation_rows):
        try:
            fn(directory,"test")
        except PermissionError:
            denied.append(fn.__name__)
        else:
            raise AssertionError("A sealed Test entry point did not reject access")
    return {"task":task,"split":"train","pilot_sources":count,"by_generator":results,
        "sealed_test_entrypoints_denied":denied,"test_quality_data_read":False,
        "protocol_sha256":C.file_hash(directory/"protocol.json")}

def audit_source_hashes(output):
    """Inspect frozen manifest membership, never Test generations or quality signals."""
    seen = set()
    counts = {}
    for task in C.TASKS:
        counts[task] = {}
        for split in ("train","dev","test"):
            n = 0
            for r in C.jsonl(Path(output)/task/"manifests"/(split+".sources.jsonl")):
                keys = {r["source_hash"],r["reference_hash"]}
                assert not (seen & keys), "Cross-source/task/split overlap"
                assert r["task"] == task and r["split"] == split and r["position"] == n
                assert r["source_hash"] == C.text_key(r["source"])
                assert r["cluster_id"] == r["source_hash"]
                seen.update(keys)
                n += 1
            counts[task][split] = n
    return {"manifest_source_counts":counts,"unique_source_reference_hashes":len(seen),
            "cross_task_and_split_overlap":0,"test_generation_or_quality_read":False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=C.OUT)
    parser.add_argument("--source-isolation",action="store_true")
    args = parser.parse_args()
    result = {"schema":C.SCHEMA,"at":C.utc(),"audit_code_sha256":C.file_hash(__file__),
              "pilots":[audit_pilot(args.output/t) for t in C.TASKS]}
    if args.source_isolation:
        result["source_isolation"] = audit_source_hashes(args.output)
    path = args.output/"audits"/("contract-"+result["at"].replace(":","-")+".json")
    C.dump(path,result,immutable=True)
    print(C.canonical({"audit":str(path),"tasks":len(result["pilots"]),"test_quality_read":False,
        "source_isolation":result.get("source_isolation")}))

if __name__ == "__main__":
    main()
