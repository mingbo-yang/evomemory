"""CPU-only report; use base Python (which includes matplotlib)."""
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/bert_onpolicy_20260925'

def main():
    data=json.loads((OUT/'data.json').read_text());rows=data['test'];stats=json.loads((OUT/'data_statistics.json').read_text())
    names=sorted(set(r[0] for r in rows));lookup={s:i for i,s in enumerate(names)};g=np.array([lookup[r[0]] for r in rows]);n=len(names)
    delta=np.array([r[4]-r[3] for r in rows]);changed=np.array([r[1]!=r[2] for r in rows]);eligible=np.array([bool(r[2].strip()) and r[1]!=r[2] and len(r[2])<=1.02*max(1,len(r[1]))+8 for r in rows]);counts=np.bincount(g,weights=changed,minlength=n)
    draws=np.random.default_rng(20260925).integers(0,n,size=(2000,n));reports={};ci={}
    group_draw_counts=np.array([np.bincount(d,minlength=n) for d in draws])
    arms=['legacy','historical_absolute','absolute','joint']
    for arm in arms:
        e=json.loads((OUT/f'{arm}_evaluation.json').read_text());reports[arm]=e
        scores=np.load(OUT/f'{arm}_predictions.npz')['test'];t=e['calibrated_threshold']
        accept=eligible&(scores>t) if t is not None else np.zeros(len(rows),dtype=bool)
        accepted=np.bincount(g,weights=accept,minlength=n)[draws].sum(1)
        good=np.bincount(g,weights=accept&(delta>1e-6),minlength=n)[draws].sum(1)
        gain=np.bincount(g,weights=accept*delta,minlength=n)[draws].sum(1)/np.maximum(counts[draws].sum(1),1)
        precision=np.divide(good,accepted,out=np.full(2000,np.nan),where=accepted>0)
        ci[arm]={'calibrated_precision_95':np.nanquantile(precision,[.025,.975]).tolist() if accept.any() else None,'calibrated_gain_95':np.quantile(gain,[.025,.975]).tolist()}
        default_accept=eligible & (scores>(.5 if arm=='joint' else .01))
        dn=np.bincount(g,weights=default_accept,minlength=n)[draws].sum(1);dg=np.bincount(g,weights=default_accept & (delta>1e-6),minlength=n)[draws].sum(1)
        dp=np.divide(dg,dn,out=np.full(2000,np.nan),where=dn>0)
        ci[arm]['default_precision_95']=np.nanquantile(dp,[.025,.975]).tolist() if default_accept.any() else None
        mask=changed & (np.abs(delta)>1e-6);order=np.argsort(scores[mask],kind='stable');ss=scores[mask][order];yy=delta[mask][order]>0;gg=g[mask][order]
        weights=group_draw_counts[:,gg];starts=np.r_[0,np.flatnonzero(np.diff(ss))+1]
        pw=np.add.reduceat(weights*yy,starts,axis=1);nw=np.add.reduceat(weights*(~yy),starts,axis=1)
        numerator=(pw*(np.cumsum(nw,axis=1)-.5*nw)).sum(1);denominator=pw.sum(1)*nw.sum(1)
        auc=np.divide(numerator,denominator,out=np.full(2000,np.nan),where=denominator>0)
        ci[arm]['auc_95']=np.nanquantile(auc,[.025,.975]).tolist()
    (OUT/'cluster_bootstrap.json').write_text(json.dumps({'source_count':n,'replicates':2000,'seed':20260925,'results':ci},indent=2))
    validation_rows=data['validation'];vd=np.array([r[4]-r[3] for r in validation_rows]);ve=np.array([bool(r[2].strip()) and r[1]!=r[2] and len(r[2])<=1.02*max(1,len(r[1]))+8 for r in validation_rows])
    threshold_scan={}
    for arm in arms:
        scores=np.load(OUT/f'{arm}_predictions.npz')['validation'];options=[]
        for threshold in np.unique(scores[ve]):
            accepted=ve & (scores>=threshold);count=int(accepted.sum())
            if count>=20: options.append({'threshold_inclusive':float(threshold),'accepted':count,'precision':float((vd[accepted]>1e-6).mean()),'net_gain_sum':float(vd[accepted].sum())})
        threshold_scan[arm]=max(options,key=lambda r:(r['precision'],r['accepted'])) if options else None
    (OUT/'validation_threshold_diagnostic.json').write_text(json.dumps({'scope':'Post-hoc validation-only threshold scan; no test thresholds or model weights changed','best_precision_with_at_least_20_acceptances':threshold_scan},indent=2))
    isolate=json.loads((OUT/'isolation.json').read_text())
    lines=['# 当前策略数据上的 BERT 验收实验','', '## 结论','', '本轮实际完成 1536 条隔离原文的 full_online 采集、两个新 BERT 各 5 轮训练和四个冻结模型的同候选评估。在线库更新、后续检索、反馈时序与成本审计均通过；验收效果未通过预定门槛，因此未替换正式配置，也未触发新策略的后续在线运行。', '', '新绝对评分模型在独立评估中的 AUC 为 0.490（原文聚类 95% 区间约 0.396–0.577），联合三分类为 0.451（0.374–0.528）。默认阈值下，新绝对评分接受 47 对，其中 16 对提升、27 对下降、4 对持平；联合模型接受 0 对。全部阈值的校准集事后扫描也未找到精度达到 80% 且接受至少 20 对的区间，失败不能只归因于预设阈值网格。', '', '本轮只能说明这些训练方案尚不能可靠承担细微修改的最终验收，不能证明 BERT 架构无效。有效训练修改对只有 961 对，且训练标签是单参考 BLEU 反馈；样本量、标签可预测性、分类容差和优化方式仍未分离。新数据方案并未展示优于历史训练模型的结果，不能把同分布采样本身当作已证实的解决办法。', '', '## 实验范围','', 'GPU 0，Qwen3-8B，英译中。先使用现有 full_online 策略实际生成轨迹，再训练和复评新验收模型；固定候选上的成绩不是新策略完整在线运行的成绩。', '', '训练/校准/最终评估分别为 1024/256/256 条原文，从 WMT 训练语料前 20000 行中确定性抽取。NFKC、大小写和空白归一化后排除主测试、既有辅助集和旧 BERT 训练原文，同时检查源文与译文重叠。三个分区重置为相同初始库；仅允许本分区中较早批次反馈带来的在线入库。', '', f"实验初始库从 {isolate['initial_library_before']} 条清理为 {isolate['initial_library_after']} 条。新模型均从预训练 mBERT 初始化，训练 5 轮；绝对评分使用 MSE，joint 同时输入原文和修改前后文本，三分类预测提升/持平/下降，训练使用 0.005 分数差容差和交换顺序增强。参考答案只生成标签，不输入模型。", '', '## 数据量','', '| 分区 | 原文 | 修改轮次 | 唯一候选对 | 有变化的唯一候选对 | 下降/持平/提升训练标签 |','|---|---:|---:|---:|---:|---|']
    for fold,s in stats.items():lines.append(f"| {fold} | {s['sources']} | {s['refine_rounds']} | {s['unique_pairs']} | {s['changed_unique_pairs']} | {'/'.join(map(str,s['class_counts']))} |")
    lines += ['', '## 在线采集流程审计', '', '| 分区 | 状态/反馈/成本审计 | 新增经验 | 后续检索命中在线经验次数 | 初稿 corpus BLEU | 最终 corpus BLEU |', '|---|---|---:|---:|---:|---:|']
    for fold in ['train','validation','test']:
        audit=json.loads((OUT/f'{fold}_flow_audit.json').read_text())
        lines.append(f"| {fold} | {audit['validation']}/{audit['cost_check']} | {audit['new_memory']} | {audit['counts'].get('online_retrieval_hits',0)} | {audit['initial_score']:.3f} | {audit['final_score']:.3f} |")
    lines += ['', '上述 corpus BLEU 来自用于采集数据的旧策略真实运行，不能归因于后续新训练的模型，也不能单独证明在线库优于固定库。']
    lines += ['', f"最终评估候选池有 {int(changed.sum())} 个有变化的唯一候选对，其中 {int((changed & (delta>1e-6)).sum())} 个提高句级指标；通过共享长度闸门且提高指标的有 {int((eligible & (delta>1e-6)).sum())} 个。这是候选池诊断，不是可实现的在线轨迹收益上限。"]
    lines+=['', '权重按校准集有变化候选的提升 AUC 选择。四种模型使用相同的保守长度上限：1.02 × 修改前字符数 + 8；空答案和完全相同答案强制拒绝。默认阈值为绝对分差 0.01、joint 提升概率 0.5。校准要求至少接受 20 个唯一候选、提升精度 ≥80%、净增量为正；在固定阈值网格内选接受量最多者，失败则关闭接受。', '', '## 独立评估：默认阈值','', '| 模型 | AUC | 接受数 | 提升/下降/持平 | 精度 | 每个变化候选净增量 |','|---|---:|---:|---|---:|---:|']
    def percent(x):return '—' if x is None else f'{x:.1%}'
    for arm,e in reports.items():
        m=e['test_default'];lines.append(f"| {arm} | {m['auc_changed_non_tie']:.3f} | {m['accepted']} | {m['improve']}/{m['degrade']}/{m['tie']} | {percent(m['precision'])} | {m['net_gain_per_changed_pair']:.5f} |")
    lines+=['', '## 独立评估：仅由校准集选定阈值','', '| 模型 | 阈值 | 接受数 | 精度 | 接受比例 | 净增量 | 满足运行新策略的门槛 |','|---|---:|---:|---:|---:|---:|---|']
    for arm,e in reports.items():
        m=e['test_calibrated'];lines.append(f"| {arm} | {m['threshold']} | {m['accepted']} | {percent(m['precision'])} | {percent(m['coverage_changed'])} | {m['net_gain_per_changed_pair']:.5f} | {e['rollout_gate_passed']} |")
    lines += ['', '作为额外诊断，只在校准集上遍历全部不同分数阈值（不修改已冻结的测试阈值），检查失败是否只是固定网格太粗。接受至少 20 个候选时的最高校准精度：' + '；'.join(arm+'='+percent(value['precision'] if value else None) for arm,value in threshold_scan.items()) + '。这些是校准集上的事后诊断，不能视为独立评估成绩。']
    fit=json.loads((OUT/'fit_diagnostic.json').read_text())
    lines += ['', '## 训练拟合诊断', '', '| 新模型 | 训练集验收排序 AUC | 校准集 AUC（选中轮次） | 独立评估 AUC |', '|---|---:|---:|---:|']
    for arm in ['absolute','joint']:
        selected=json.loads((OUT/f'{arm}_selection.json').read_text())
        lines.append(f"| {arm} | {fit[arm]['train']['auc_changed_non_tie']:.3f} | {selected['validation_auc']:.3f} | {reports[arm]['test_default']['auc_changed_non_tie']:.3f} |")
    lines += ['', '训练损失下降不等价于验收排序学会了：联合模型在训练集上的该指标也约为随机水平。因此目前不能将问题仅解释成测试分布偏移或过拟合，更不能根据本轮直接宣布 BERT 架构无效。该诊断不改变权重与阈值；排序按实际正负反馈统计，训练三分类另有 0.005 容差，二者任务差异已保留在协议中。']
    lines+=['', 'legacy 为原有权重；historical_absolute 为上一轮历史数据实验的绝对评分权重；absolute/joint 为本轮新训练权重。本轮原文与这些旧模型已知训练原文隔离，因此旧模型可作迁移参照；但它们的训练量、标签和预算不同，不能把差异单独归因于数据来源。absolute 与 joint 在本轮数据和轮数相同，输入结构、损失及增强不同。', '', '本轮标签使用当前流程句级 BLEU 代理指标（0–1），不是人工质量判定，也不是 corpus BLEU。完全相同的修改前后组合在同一原文下只算一次；剩余多候选仍有原文内相关性。置信区间按原文聚类 bootstrap 2000 次，详见 `cluster_bootstrap.json`。', '', '![learning curves](bert_onpolicy_learning_curves.png)', '', '## 产物','', '全部清单、隔离记录、原始在线轨迹、初稿缓存、训练数据、权重、逐轮曲线、阈值搜索与最终预测保存在 `../runs/bert_onpolicy_20260925/`。原有主实验配置及权重不变。']
    report=ROOT/'reports/BERT_ONPOLICY_STUDY.md';report.write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.7))
    for arm in ['absolute','joint']:
        h=json.loads((OUT/f'{arm}_learning_curve.json').read_text());epochs=[r['epoch'] for r in h]
        axes[0].plot(epochs,[r['validation']['auc_changed_non_tie'] for r in h],marker='o',label=arm)
        axes[1].plot(epochs,[np.nan if r['validation']['precision'] is None else r['validation']['precision'] for r in h],marker='o',label=arm)
    axes[0].set_title('Validation improvement AUC');axes[1].set_title('Default-threshold precision')
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'reports/bert_onpolicy_learning_curves.png',dpi=160);plt.close(fig)
    print(report)
if __name__=='__main__':main()
