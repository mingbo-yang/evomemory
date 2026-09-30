"""Score completed trajectories; no policy selection or model update occurs here."""
from collections import Counter
import json
import numpy as np
from sacrebleu.metrics import BLEU,CHRF
import sacrebleu
from laya_relaxed_flow import OUT,ROOT,MODEL_OUT,read_manifest,Scorer,TASK
from semantic_label_common import dump


def feedback_stats(traces,refs):
    scorer=Scorer(TASK);accepted=[];candidates=[];invalid=Counter();rounds=0;logical_tokens=0;online_hits=0
    for tr,ref in zip(traces,refs):
        assert tr['sample_id']==ref.sample_id
        assert len(tr['rounds'])<=3
        current=tr['initial']
        for rd in tr['rounds']:
            assert rd['before']==current,'Broken trajectory state replay'
            assert len(rd['candidates'])==4
            rounds+=1;online_hits+=rd['online_retrieved'];before=scorer.primary(ref.reference,rd['before'])
            for c in rd['candidates']:
                if c['finish_reason']!='context_length_exceeded':logical_tokens+=c['input_tokens']+c['output_tokens']
                if not c['valid']:invalid[c['validity_reason']]+=1
                if c['valid'] and c['changed']:
                    candidates.append(scorer.primary(ref.reference,c['text'])-before)
            if rd['accepted']:
                chosen=rd['candidates'][rd['selected_slot']];current=chosen['text']
                accepted.append({'sample_id':tr['sample_id'],'round':rd['round'],'source':tr['source'],
                   'before':rd['before'],'after':current,'probabilities':chosen['probabilities'],
                   'proxy_delta':scorer.primary(ref.reference,current)-before})
            else:assert rd is tr['rounds'][-1],'A rejected round must stop the task'
        assert current==tr['final']
    d=np.array([r['proxy_delta'] for r in accepted]);n=len(d)
    return {'attempted_rounds':rounds,'candidate_slots':rounds*4,'accepted_revisions':n,
       'acceptance_per_round':n/rounds if rounds else 0.,
       'tasks_with_acceptance':sum(any(rd['accepted'] for rd in tr['rounds']) for tr in traces),
       'final_changed_tasks':sum(tr['initial']!=tr['final'] for tr in traces),
       'invalid_candidates':dict(invalid),'online_memory_retrieval_hits':online_hits,
       'logical_revision_tokens':logical_tokens,
       'accepted_feedback':{'semantics':'legacy sentence-metric proxy, not independent semantic truth; thresholds +/-1 point',
          'better_gt1':int((d>1).sum()),'tie_within1':int((abs(d)<=1).sum()),'worse_lt_minus1':int((d<-1).sum()),
          'precision_gt1':float((d>1).mean()) if n else None,'positive_any':int((d>1e-8).sum()),
          'precision_gt0':float((d>1e-8).mean()) if n else None,'mean_delta':float(d.mean()) if n else None},
       'accepted_pairs':accepted,'valid_changed_candidate_slots':len(candidates)}


def main():
    protocol=json.loads((OUT/'protocol.json').read_text());refs=read_manifest(OUT/'test_manifest.jsonl')
    drafts=json.loads((OUT/'shared_drafts.json').read_text());gold=[r.reference for r in refs]
    outputs={'initial':[d['text'] for d in drafts]};traces={}
    for name in ['laya_static','unfiltered_static','laya_online','laya_bleu_static']:
        path=OUT/name/'results.json'
        if path.exists():
            tr=json.loads(path.read_text());assert len(tr)==len(refs)
            assert all(t['initial']==d['text'] for t,d in zip(tr,drafts))
            traces[name]=tr;outputs[name]=[t['final'] for t in tr]
    metrics={'sacrebleu':BLEU(tokenize='zh'),'chrfpp':CHRF(word_order=2)}
    scores={a:{} for a in outputs};stats={a:{} for a in outputs}
    for name,metric in metrics.items():
        for arm,hyps in outputs.items():
            scores[arm][name]=float(metric.corpus_score(hyps,[gold]).score)
            st=metric._extract_corpus_statistics(hyps,[gold]);stats[arm][name]=np.asarray(st)
            reconstructed=metric._compute_score_from_stats(np.sum(st,axis=0).tolist()).score
            assert abs(reconstructed-scores[arm][name])<1e-8,'Bootstrap sufficient-statistic parity failed'
    n=len(refs);indices=np.random.default_rng(20260927).integers(n,size=(2000,n));boots={a:{} for a in outputs}
    for arm in outputs:
        for name,metric in metrics.items():
            aggregated=stats[arm][name][indices].sum(1)
            boots[arm][name]=np.array([metric._compute_score_from_stats(row.tolist()).score for row in aggregated])
    comparisons={}
    pairs=[(a,'initial') for a in traces]
    pairs.append(('laya_static','unfiltered_static'))
    if 'laya_online' in traces:pairs.append(('laya_online','laya_static'))
    if 'laya_bleu_static' in traces:pairs.append(('laya_bleu_static','laya_static'))
    for a,b in pairs:
        comparisons[f'{a}_minus_{b}']={m:{'delta':scores[a][m]-scores[b][m],
              'ci95':np.quantile(boots[a][m]-boots[b][m],[.025,.975]).tolist()} for m in metrics}
    details={a:feedback_stats(tr,refs) for a,tr in traces.items()}
    for a,d in details.items():dump(OUT/a/'accepted_feedback.json',d['accepted_pairs'])
    report={'sources':n,'initial_valid':sum(t['draft_valid'] for t in traces['laya_static']),
       'sacrebleu_version':sacrebleu.__version__,'scores':scores,'paired_source_bootstrap':{'replicates':2000,'comparisons':comparisons},
       'arms':{a:{k:v for k,v in d.items() if k!='accepted_pairs'} for a,d in details.items()},
       'policy':protocol['policy'],'continuation':json.loads((OUT/'static_continuation.json').read_text()),
       'audit':{'same_initials_all_arms':True,'accepted_state_replay':True,'no_feedback_in_acceptance':True,
                'no_test_threshold_retuning':True,'statistics_match_public_metric_apis':True},
       'limitations':['256 fresh sources from WMT train, not the official final main benchmark.',
          'Threshold relaxed at user request using a prior 48-pair calibration fold; calibration precision is not guaranteed for best-of-4 rollouts.',
          'Corpus metrics are reference based. Per-revision feedback is a weak lexical proxy, not human correctness.',
          'Online continuation is conditional on positive static point estimate and is exploratory.',
          'Online feedback is available for all generated candidates in this benchmark; this assumes counterfactual candidate feedback.',
          'Generation reuse keeps matching requests identical; wall time with caching is not a method latency comparison.']}
    if 'laya_bleu_static' in traces:report['secondary_bleu_control']=json.loads((OUT/'bleu_control_protocol.json').read_text())
    for arm in traces:
        audit_path=OUT/arm/'accepted_semantic_audit.json'
        if audit_path.exists():report['arms'][arm]['semantic_audit']={k:v for k,v in json.loads(audit_path.read_text()).items() if k!='rows'}
    blind=OUT/'accepted_blind_review_result.json'
    if blind.exists():report['ai_blind_review']={k:v for k,v in json.loads(blind.read_text()).items() if k!='rows'}
    if 'laya_online' in traces:
        events=[json.loads(p.read_text()) for p in sorted((OUT/'laya_online').glob('admission_*.json'))]
        report['memory']={'initial':protocol['initial_memory_size'],'final':events[-1]['bank_size'],
                         'admitted':sum(x['added'] for x in events),'update_batches':sum(x['added']>0 for x in events)}
    dump(OUT/'report.json',report)
    lines=['# 放宽接受门槛后的 Laya 真实流程试验','',
       f"Qwen3-8B + 新训练的 Laya multilingual（语义标签主试验、BLEU 标签附加对照）；全新 {n} 个 EN→ZH 原文。所有方法共享逐字相同的初稿与初始经验库。",
       '', '接受策略：p(Better) ≥ 0.275 且 p(Worse) ≤ 0.4；最多 3 轮、每轮 4 个候选，拒绝即停止。阈值只用旧校准集确定，不用本次测试反馈调参。',
       '', '| 方法 | corpus BLEU | 相对初稿 ΔBLEU [95% CI] | chrF++ | 接受修改数 | 最终改变任务数 |',
       '|---|---:|---|---:|---:|---:|']
    for arm,score in scores.items():
        d=details.get(arm,{})
        change='—'
        if arm!='initial':
            c=comparisons[f'{arm}_minus_initial']['sacrebleu'];change=f"{c['delta']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
        lines.append(f"| {arm} | {score['sacrebleu']:.4f} | {change} | {score['chrfpp']:.4f} | {d.get('accepted_revisions',0)} | {d.get('final_changed_tasks',0)} |")
    if 'laya_bleu_static' in traces:
        lines+=['', 'laya_static 使用语义标签版本；laya_bleu_static 使用本次另一份 BLEU 标签版本。后者在主试验完成后追加，阈值与主方法相同，未根据其测试表现调参；该附加对照的校准集在相同阈值下只接受 2 对（1 对相当、1 对不确定），没有声称它达到了 50% 校准精度。']
    lines+=['', '无判别器修订每轮选择第一个通过基本有效性检查且有改动的候选；Laya 选择满足阈值的最高 p(Better) 候选。生成温度、检索、候选数与最大轮数一致，实际轮数随停止策略变化。', '',
       '实际接受修改的反馈（当前工程句级代理分数；不是人工语义准确率）：','']
    for arm,d in details.items():
        a=d['accepted_feedback'];precision='未定义' if a['precision_gt1'] is None else f"{a['precision_gt1']:.1%}"
        mean='未定义' if a['mean_delta'] is None else f"{a['mean_delta']:+.4f}"
        lines.append(f"- {arm}：改善 {a['better_gt1']}、基本相当 {a['tie_within1']}、退化 {a['worse_lt_minus1']}；改善比例 {precision}，平均反馈变化 {mean}。")
    static_delta=comparisons['laya_static_minus_initial']['sacrebleu']
    lines+=['',f"固定库的 BLEU 点估计{'提升' if static_delta['delta']>0 else '未提升'}；95% 区间{'高于零' if static_delta['ci95'][0]>0 else '未完全高于零'}。这里报告真实得分变化，不再以 80% 分类精度作为禁止运行的条件。",
       '',f"在线库阶段：{'已运行' if 'laya_online' in traces else '未运行，固定库未满足预先声明的正收益条件'}。"]
    if 'memory' in report:
        c=comparisons['laya_online_minus_laya_static']['sacrebleu'];mem=report['memory']
        lines += [f"经验库 {mem['initial']} → {mem['final']}，新增 {mem['admitted']} 条；后续检索命中新经验 {details['laya_online']['online_memory_retrieval_hits']} 次。",
             f"在线相对固定库 BLEU：{c['delta']:+.4f}，95% CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]。",
             '入库在整批任务结束后使用正反馈决定，与候选是否被 Laya 接受分离；生成器和 Laya 参数始终冻结。']
    for arm,detail in report['arms'].items():
        if 'semantic_audit' in detail:
            audit=detail['semantic_audit']
            lines += ['', f"{arm} 全部 {audit['n']} 条已接受修改的参考答案盲评：本地 Qwen 教师计数 {audit['counts']}；被该教师确认改善的比例 {audit['confirmed_improvement_lower']:.1%}。这是与训练标签相同教师的辅助模型评审，不是独立人工金标，也不是全体候选分类准确率。"]
    if 'ai_blind_review' in report:
        lines += [f"Codex 对语义标签版随机 A/B 顺序的全部 14 条已接受修改复核：{report['ai_blind_review']['counts']}。复核隐藏逐条参考、分数和教师标签，但已知总体统计；它与教师对相当/退化的判断有分歧，不能当成人工验证。"]
    lines+=['', '局限：小规模、单次运行；参考指标不能覆盖全部语义质量；本次仅验证固定库流程。预设的在线分支若运行，会用基准参考答案模拟所有候选的事后反馈，这要求场景支持未采纳候选的反馈。当前结论不能替代正式主实验或人工评价。', '',
       f"语义标签模型：{protocol['policy']['checkpoint']}。BLEU 标签模型：{MODEL_OUT / 'bleu/best'}。两份权重均未在本次测试中更新。原始轨迹、接受修改、检索 ID、生成缓存、协议与完整指标均保存在本目录。"]
    text='\n'.join(lines)+'\n';(OUT/'RESULTS_ZH.md').write_text(text)
    (ROOT/'reports/LAYA_RELAXED_FLOW_RESULTS.md').write_text(text)
    dest=MODEL_OUT/'semantic/relaxed_flow_v1';dest.mkdir(exist_ok=True)
    (dest/'RESULTS_ZH.md').write_text(text);dump(dest/'report.json',report)
    if 'laya_bleu_static' in traces:
        bleu_dest=MODEL_OUT/'bleu/relaxed_flow_v1';bleu_dest.mkdir(exist_ok=True)
        (bleu_dest/'RESULTS_ZH.md').write_text(text);dump(bleu_dest/'report.json',report)
    dump(OUT/'status.json',{'stage':'complete','report':str(OUT/'RESULTS_ZH.md'),'static_bleu_delta':static_delta['delta'],'online_ran':'laya_online' in traces})
    print(json.dumps({'scores':scores,'comparisons':comparisons,'arms':report['arms'],'memory':report.get('memory')},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
