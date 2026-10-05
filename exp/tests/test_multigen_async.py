import threading
import time
from pathlib import Path
import pytest
from multigen_data import common as C
from run_multigen_async import (VERSION, StagedCollector, shards, required_horizon,
                               common_prefix, train)

def test_required_horizon_only_dispatches_mathematically_necessary_shards():
    import random
    rng = random.Random(20261004)
    total,initial,size,target = 1300,256,64,10000
    contributions = [rng.randrange(21) for _ in range(total)]
    prefix = [0]
    for n in contributions:
        prefix.append(prefix[-1]+n)
    ranges = shards(total,initial,size)
    oracle = next((stop for start,stop in ranges if stop>=initial and prefix[stop]>=target),total)
    previous = 0
    for frontier in [128]+[b for a,b in ranges if b<=oracle]:
        horizon = required_horizon(frontier,prefix[frontier],total,
                     initial=initial,size=size,target=target)
        assert previous <= horizon <= oracle
        previous = horizon
    assert required_horizon(1128,7598,100000) == 6000

def test_required_horizon_handles_target_and_source_exhaustion():
    assert required_horizon(5000,100000,100000) == 5000
    assert required_horizon(128,0,1000) == 1000
    with pytest.raises(ValueError):
        required_horizon(128,0,1000,max_pairs_per_source=0)

def test_staging_is_invisible_until_immutable_publication_and_resume_reuses_record(tmp_path):
    c = StagedCollector.__new__(StagedCollector)
    c.published_directory = tmp_path
    c.directory = tmp_path/"staging"/VERSION/"fast"
    c.model = "fast"
    key = "ab"+"c"*62
    source = c.directory/"raw/train/fast/ab"/(key+".json")
    raw = {"source_id":"coedit_gec/"+key,"source":"source","split":"train"}
    raw["record_hash"] = C.digest(raw)
    C.dump(source,raw,immutable=True)
    marker = c.completion_path(c.directory,"train",128,129)
    C.dump(marker,{"start":128,"stop":129,"split":"train","generator":"fast","sources":1,
                  "raw_records":[{"path":str(source),"sha256":C.file_hash(source)}]},immutable=True)
    assert not list((tmp_path/"raw").rglob("*.json"))
    assert c.ready("train",128,129)
    result = c.publish("train",128,129)
    published = Path(result["raw_records"][0]["path"])
    assert published.exists() and published.read_bytes()==source.read_bytes()
    assert c.one({"source_id":raw["source_id"],"split":"train"}) == str(published)
    assert c.publish("train",128,129) == result

class FakeCollector:
    def __init__(self,directory,name,events,fast_at_132,slow=False):
        self.published_directory = directory
        self.model = name
        self.events = events
        self.fast_at_132 = fast_at_132
        self.slow = slow
        self.generated = set()
        self.calls = []
        self.lock = threading.Lock()
    def run(self,split,start,stop):
        self.calls.append((start,stop))
        self.events.append((self.model,"start",start,stop))
        time.sleep(.05 if self.slow else .002)
        with self.lock:
            self.generated.add((start,stop))
        if not self.slow and stop>=132:
            self.fast_at_132.set()
    def ready(self,split,start,stop):
        with self.lock:
            return (start,stop) in self.generated
    def publish(self,split,start,stop):
        C.dump(self.published_directory/"completed/train"/self.model/f"{start:08d}-{stop:08d}.json",
               {"sources":stop-start},immutable=True)

def exercise_pipeline(tmp_path,target):
    p = {"pool_counts":{"train":136},"initial_train_sources":132,
         "target_unique_valid_changed":target,"shard_sources":2,"candidates":4}
    C.dump(tmp_path/"protocol.json",p)
    events = []
    done = threading.Event()
    collectors = [FakeCollector(tmp_path,"fast",events,done),
                  FakeCollector(tmp_path,"slow",events,done,slow=True)]
    first = True
    def clean(directory,split,sizer):
        nonlocal first
        assert split=="train"
        events.append(("clean","start"))
        if first:
            # Generation of future shards must advance DURING the first clean.
            assert done.wait(timeout=3),"GPU workers were serialized behind cleaning"
            assert not list((tmp_path/"raw").rglob("*.json"))
            assert not list((tmp_path/"completed").rglob("*.json"))
            first = False
        f = common_prefix(tmp_path,["fast","slow"],shards(136,132,2))
        events.append(("clean","end"))
        return {"unique_valid_changed_pairs":(f-128)*8}
    result = train(tmp_path,collectors,None,clean_fn=clean,timing_fn=lambda d:None)
    return result,collectors,events

def test_generation_runs_during_cleaning_and_models_end_on_the_same_source_prefix(tmp_path):
    result,collectors,events = exercise_pipeline(tmp_path,100)
    assert result["train_sources"]==136 and result["pool_exhausted"]
    assert not result["target_met"]
    assert collectors[0].calls==collectors[1].calls
    assert events.index(("fast","start",130,132)) < events.index(("clean","end"))
    previous = [list(c.calls) for c in collectors]
    assert train(tmp_path,collectors,None)==result
    assert [c.calls for c in collectors]==previous

def test_no_speculative_suffix_is_generated_beyond_the_first_target_shard(tmp_path):
    result,collectors,events = exercise_pipeline(tmp_path,32)
    assert result["train_sources"]==132 and result["target_met"]
    assert all(c.calls==[(128,130),(130,132)] for c in collectors)
