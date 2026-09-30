"""Isolated current-policy data collection, on physical GPU 0 only."""
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse,json,hashlib,random,unicodedata,gc
from pathlib import Path
from dataclasses import asdict
from core.manifest import _iter_wmt_train_pairs,SampleRef,read_manifest,sha256_text
from core.experience import load_experiences,save_experiences,set_experience_root
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/bert_onpolicy_20260925'

def normalized(s):
    return ''.join(unicodedata.normalize('NFKC',s).casefold().split())
def dump(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'isolation.json').exists(): return
    blocked=set()
    for p in (ROOT/'data/manifests').glob('*.jsonl'):
        for r in read_manifest(p): blocked.update([normalized(r.source),normalized(r.reference)])
    history=json.loads((ROOT/'runs/bert_feedback_study/source_split.json').read_text())
    blocked.update(normalized(s) for ss in history.values() for s in ss)
    main=set()
    for task in ['wmt19_en_zh','wmt19_zh_en']:
        for r in read_manifest(ROOT/f'data/manifests/{task}__test.jsonl'):
            main.update([normalized(r.source),normalized(r.reference)])
    seen=set(); pool=[]; excluded=0
    for index,source,reference in _iter_wmt_train_pairs('en_zh'):
        if index>=20000: break
        keys={normalized(source),normalized(reference)}
        if not all(keys) or keys&blocked or keys&seen:
            excluded+=1;continue
        seen.update(keys);pool.append((index,source,reference))
    random.Random(20260925).shuffle(pool)
    assert len(pool)>=1536
    folds={}; offset=0
    for fold,n in [('train',1024),('validation',256),('test',256)]:
        refs=[SampleRef(sample_id=f'wmt19_en_zh/onpolicy_{fold}/{i:06d}',task='wmt19_en_zh',split='accumulation',row_index=i,source=s,reference=r,source_hash=sha256_text(s),provenance='wmt19 train first20000; normalized source AND target exclusion; seed20260925; on-policy study') for i,s,r in pool[offset:offset+n]]
        offset+=n;folds[fold]=refs
        (OUT/f'{fold}_manifest.jsonl').write_text(''.join(json.dumps(asdict(r),ensure_ascii=False)+'\n' for r in refs))
    allkeys={normalized(v) for refs in folds.values() for r in refs for v in (r.source,r.reference)}
    assert not allkeys&blocked
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not {normalized(x.source) for x in folds[a]}&{normalized(x.source) for x in folds[b]}
    lib=load_experiences(ROOT/'experience/contrastive/qwen3-8b/wmt19_en_zh/initial.jsonl')
    clean=[e for e in lib if normalized(e.source_input) not in main|allkeys]
    dest=OUT/'library/qwen3-8b/wmt19_en_zh/initial.jsonl';save_experiences(clean,dest)
    assert clean
    profile=json.loads((ROOT/'configs/optimized_feedback_v1.json').read_text());profile['device']='cuda'
    dump(OUT/'collection_profile.json',profile)
    dump(OUT/'isolation.json',{'fold_sizes':{f:len(v) for f,v in folds.items()},'pool_size':len(pool),'excluded_rows':excluded,'normalization':'NFKC + casefold + remove whitespace; compare both source and reference','overlap_main_test':0,'overlap_previous_aux_and_historical_bert_sources':0,'cross_fold_source_overlap':0,'initial_library_before':len(lib),'initial_library_after':len(clean),'removed_library_main_or_study_sources':len(lib)-len(clean),'manifest_sha256':{f:hashlib.sha256((OUT/f'{f}_manifest.jsonl').read_bytes()).hexdigest() for f in folds}})
    dump(OUT/'protocol.json',{'version':1,'scope':'current-policy en-zh feedback pilot, not main benchmark','gpu':0,'generator':'qwen3-8b','collection':'unchanged full_online BERT policy and hybrid retrieval, 3 rounds, batch16, same library reset for EACH fold; all generated candidates retained including rejected/ties','data':'1024 train /256 calibration /256 final evaluation sources; exclude previous auxiliaries and all historical BERT sources','labels':'current Scorer.primary /100, computed after generation; joint classifier margin .005 for improve/degrade, otherwise tie','models':'base pretrained mBERT absolute MSE versus joint source-before-after three-class acceptance model; legacy and previous absolute checkpoint as frozen controls','epochs':5,'checkpoint_selection':'maximum validation changed-candidate improvement AUC; test used only after selection','calibration':'fixed grids; validation accepted>=20 and precision>=.80, maximize coverage; no qualifying threshold disables acceptance','class_input':'source, Before+After; reference never model input','rollout_gate':'new full_online run only if calibrated heldout nonzero acceptance, precision>=.80 and positive net metric gain','bootstrap':'source-level paired bootstrap; changed candidates deduplicated within source; report coverage and precision'})
    print(json.dumps(json.loads((OUT/'isolation.json').read_text()),indent=2),flush=True)

def collect():
    import torch
    torch.set_num_threads(4)
    import run_experiment as runner
    from baseline_core.llm import LLMClient
    from core import resolve_model_config,MAX_MODEL_LEN
    from core.optimization_config import resolve_optimization
    set_experience_root(OUT/'library')
    settings=resolve_optimization(OUT/'collection_profile.json','wmt19_en_zh')
    settings['study_manifest_hashes']=json.loads((OUT/'isolation.json').read_text())['manifest_sha256']
    llm=LLMClient(resolve_model_config('qwen3-8b'),backend='vllm',gpu='0',gpu_memory_utilization=.34,max_model_len=MAX_MODEL_LEN,enforce_eager=True)
    for fold in ['train','validation','test']:
        refs=read_manifest(OUT/f'{fold}_manifest.jsonl')
        runner.read_manifest=lambda path,selected=refs:list(selected)
        print(f'COLLECT {fold} {len(refs)}',flush=True)
        result=runner.run_model_task(arm='full_online',model='qwen3-8b',task='wmt19_en_zh',seed=42,gpu='0',split='accumulation',tag=fold,limit=None,max_rounds=3,alpha=.5,k=4,draft_source='cached',draft_cache=str(OUT/f'{fold}_drafts.jsonl'),batch_size=16,llm=llm,out_dir=OUT/'collection',renderer='v2',advice_mode='summary',retrieval_excludes_own_source=True,optimization=dict(settings))
        dump(OUT/f'{fold}_collection_summary.json',result)
        gc.collect();torch.cuda.empty_cache()
        print('COMPLETED '+fold,flush=True)
    print('COLLECTION COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    prepare()
    if not args.prepare_only: collect()
