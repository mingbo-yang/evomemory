"""Secondary audit only: no checkpoint, calibration or threshold selection."""
from relabel_semantic_training_api import collect
from semantic_label_common import OUT,read_jsonl,combine,dump
from semantic_label_pilot import write_jsonl


def run():
    for fold in ['development','temperature_calibration','threshold_calibration','test']:
        rows=read_jsonl(OUT/'inputs'/f'{fold}.jsonl')
        calls=collect(rows,OUT/f'compact_{fold}_calls.jsonl',f'compact_{fold}_status.json')
        labeled=[{'id':r['id'],'label':combine(calls[r['id'],False],calls[r['id'],True])} for r in rows]
        write_jsonl(OUT/'labels_compact'/f'{fold}.jsonl',labeled)
        dump(OUT/f'compact_{fold}_status.json',{'state':'complete','n':len(rows)})
    print('FORMAT_AUDIT_LABELS_COMPLETE',flush=True)


if __name__=='__main__':run()
