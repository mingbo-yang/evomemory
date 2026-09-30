"""Deterministic, outcome-blind compression of public paired editor evidence."""
import json
from .core import digest

VERSION='editor-budget-v1'
MAX_BYTES=16000


def prepare(policy,evidence):
    source={'policy':{k:policy[k] for k in ('key','when','do')},'evidence':evidence}
    def encoded(x):return len(json.dumps(x,ensure_ascii=False,sort_keys=True).encode())
    for limit in (None,1024,512,256,128,64):
        traces={}
        def clip(text):
            if limit is None or not isinstance(text,str) or len(text)<=limit:return text
            return text[:limit//2]+' [...omitted...] '+text[-limit//2:]
        def trace(rows,extra=()):
            # Keep beginning, end, and first branch divergence; no score-based selection.
            indices=list(range(len(rows))) if limit is None else sorted({i for i in (0,len(rows)-2,len(rows)-1,*extra) if 0<=i<len(rows)})
            view=[{'index':i,**{k:clip(rows[i][k]) for k in (('action','observation','admissible') if limit is None else ('action','observation')) if k in rows[i]}} for i in indices]
            obj={'original_steps':len(rows),'selected':view,'omitted_steps':len(rows)-len(indices)}
            key=digest(obj);traces.setdefault(key,obj);return key
        compact=[]
        for e in evidence:
            pairs=[]
            for pair in e['pairs']:
                a=pair['expose']['continuation'];b=pair['mask']['continuation']
                divergence=next((i for i in range(min(len(a),len(b))) if a[i]!=b[i]),min(len(a),len(b)))
                pairs.append({branch:{'score':pair[branch]['score'],'done':pair[branch]['done'],
                    'trace':trace(pair[branch]['continuation'],(divergence,))} for branch in ('expose','mask')})
            compact.append({'goal':clip(e['goal']),'history':trace(e['history']),
                'pairs':pairs,'mean_delta':e['mean_delta'],'direction':e['direction']})
        payload={'policy':source['policy'],'evidence':compact,'traces':traces,
            'compression':{'version':VERSION,'text_character_limit':limit,'note':'All pairs/outcomes retained; traces deduplicated. Missing/truncated text is unknown, never negative evidence. Trace selection is independent of outcome sign.'}}
        if encoded(payload)<=MAX_BYTES:
            return payload,{'version':VERSION,'source_hash':digest(source),'original_bytes':encoded(source),'compressed_bytes':encoded(payload),'text_character_limit':limit}
    raise ValueError('editor evidence metadata exceeds fixed input budget; no evidence pairs dropped')
