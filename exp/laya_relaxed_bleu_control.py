"""Exploratory second new checkpoint, same frozen inputs and operating cutoffs."""
import copy
import json
import torch
from laya_relaxed_flow import OUT,MODEL_OUT,Generator,Verifier,Flow,prepare,read_manifest,status
from semantic_label_common import dump


def main():
    torch.set_num_threads(4)
    original=prepare();p=copy.deepcopy(original)
    control=json.loads((OUT/'bleu_control_protocol.json').read_text())
    p['policy']=control['policy']
    refs=read_manifest(OUT/'test_manifest.jsonl');drafts=json.loads((OUT/'shared_drafts.json').read_text())
    status('loading_bleu_control')
    gen=Generator(p);verifier=Verifier(p);flow=Flow(p,gen,verifier)
    flow.run('laya_bleu_static',refs,drafts)
    dump(OUT/'laya_bleu_static/execution_audit.json',{'uncached_generation_calls':gen.calls,'cache_hits':gen.hits,
             'unique_verifier_pairs':verifier.scored,'same_initials':True,'same_operating_thresholds':True})
    status('bleu_control_generation_complete')


if __name__=='__main__':main()
