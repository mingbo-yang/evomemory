"""Source-clustered uncertainty and a reviewable experiment report."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent / 'runs/bert_feedback_study'
def dump(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))

def main():
    data=json.loads((ROOT/'data.json').read_text())
    rows=data['test']; source_names=sorted(set(r[0] for r in rows)); lookup={s:i for i,s in enumerate(source_names)}
    group=np.asarray([lookup[r[0]] for r in rows]); dy=np.asarray([r[4]-r[3] for r in rows]); ng=len(source_names)
    counts=np.bincount(group,minlength=ng)
    rng=np.random.default_rng(2026); draws=rng.integers(0,ng,size=(2000,ng))
    boot={}; intervals={}; evaluations={}
    for arm in ['absolute','pairwise','legacy']:
        evaluations[arm]=json.loads((ROOT/f'{arm}_evaluation.json').read_text())
        pred=np.load(ROOT/f'{arm}_predictions.npz')['test']; accept=pred[:,1]-pred[:,0]>.01
        n=np.bincount(group,weights=accept,minlength=ng); positive=np.bincount(group,weights=accept&(dy>1e-6),minlength=ng); gain=np.bincount(group,weights=accept*dy,minlength=ng)
        numerator=positive[draws].sum(1); denominator=n[draws].sum(1)
        precision=np.divide(numerator,denominator,out=np.full(2000,np.nan),where=denominator>0)
        net=gain[draws].sum(1)/counts[draws].sum(1)
        boot[arm]={'precision':precision,'mean_gain_per_pair':net}
        intervals[arm]={k:np.nanquantile(v,[.025,.975]).tolist() for k,v in boot[arm].items()}
    intervals['pairwise_minus_absolute']={k:np.nanquantile(boot['pairwise'][k]-boot['absolute'][k],[.025,.975]).tolist() for k in boot['absolute']}
    dump(ROOT/'cluster_bootstrap.json',{'unit':'source', 'n_sources':ng, 'replicates':2000,'seed':2026,'fixed_threshold':.01,'percentile_95_intervals':intervals})
    external=json.loads((ROOT/'external_evaluation.json').read_text())
    manifest=json.loads((ROOT/'data_manifest.json').read_text())
    lines=['# BERT 反馈模型诊断实验','', '## 结论','', '本轮不支持简单增加训练轮数或直接用这次配对损失替换旧 BERT。训练损失不断下降，但后期验证验收能力退步。历史留出测试中，配对目标未取得稳定优势；两个新模型经阈值校准可达到较高精度，但接受率仅约 4–5%。在当前 full_online 独立候选上，两者默认阈值的净收益仍为负，校准阈值则接受 0 个候选。这提示训练样本与当前候选分布不匹配、细粒度增量监督不足；尚不能据此判定 BERT 架构无效。', '', '建议下一步从与主实验测试原文隔离的开发/累积集，使用当前生成器和实际修改流程收集候选，构建提升/持平/下降配对监督，再校准验收和停止阈值。当前试验权重不接入正式主实验。', '', '## 范围与协议','', '英译中小规模诊断实验，使用 GPU 1。每种新模型训练 5 轮，训练/验证/测试分别使用 6000/2000/2000 对有变化的相邻轮答案；按原文 70%/15%/15% 划分，无原文交叉。两种新模型均从同一本地预训练 BERT 初始化，架构、随机种子、批量和学习率相同。绝对评分使用 MSE；配对模型在 MSE 上增加排序损失。参考答案只用于生成监督标签，不输入模型。', '', '两个新模型均按最低验证集增量 MSE 选权重。验收阈值只在验证集上校准，要求至少接受 30 对且精度达到 80%；未达标则标记校准失败，不把“全部拒绝”当成成功。', '', f"清洗隔离了 {manifest['quarantined_source_answer_keys']} 个标签冲突的原文-答案键，共 {sum(v['quarantined_pairs'] for v in manifest['counts'].values())} 对修改。", '', '**旧权重曾见过该历史数据的原文，因此其留出测试指标仅作污染条件下的诊断参照，不能与干净新模型作公平排名。** 本轮新权重也仅供诊断：历史训练数据与现有主实验测试原文存在重叠，未经主实验隔离不能直接部署为正式主实验模型。', '', '## 留出测试集（固定阈值 0.01）','', '| 模型 | 选中轮次 | 增量相关性 | AUC | 接受数 | 提升/下降/持平 | 接受精度 | 每对平均净增量 | 校准成功 |','|---|---:|---:|---:|---:|---|---:|---:|---|']
    for arm in ['absolute','pairwise','legacy']:
        e=evaluations[arm]; m=e['test_fixed']; epoch='旧权重' if arm=='legacy' else str(json.loads((ROOT/f'{arm}_selection.json').read_text())['epoch'])
        lines.append(f"| {arm} | {epoch} | {m['gain_spearman']:.3f} | {m['improvement_auc']:.3f} | {m['accepted']} | {m['accepted_improve']}/{m['accepted_degrade']}/{m['accepted_tie']} | {m['accept_precision']:.1%} | {m['mean_gain_per_pair']:.5f} | {e['calibration_passed']} |")
    lines += ['', '标签为旧训练使用的字符级 sentence BLEU（0–1），不等价于人工质量或主实验 corpus BLEU。', '', '## 验证集校准后的测试表现','', '| 模型 | 阈值 | 接受数 | 接受精度 | 每对平均净增量 |','|---|---:|---:|---:|---:|']
    for arm,e in evaluations.items():
        m=e['test_calibrated']; precision='—' if m['accept_precision'] is None else f"{m['accept_precision']:.1%}"
        lines.append(f"| {arm} | {m['threshold']} | {m['accepted']} | {precision} | {m['mean_gain_per_pair']:.5f} |")
    diff=intervals['pairwise_minus_absolute']
    lines += ['',f"按原文聚类 bootstrap（2000 次），配对减绝对评分的接受精度差 95% 区间：{diff['precision']}；每对净增量差区间：{diff['mean_gain_per_pair']}。", '', '## 当前 full_online 候选外部复查','',f"独立的 {external['sample_count']} 条开发集原文，与整个历史数据无原文重叠。固定旧策略生成的候选，重放各轮真实修改前状态，仅重新评分；不是重新执行在线检索进化，也不是新模型的最终 BLEU。重复轮次有相关性，样本量小。阈值和权重未用此外部数据调整。65 条有变化的轮次中只有 36 个唯一的原文/前答案/后答案组合，不能将其视为 65 个独立样本。", '', '| 模型 | 长度闸门后接受数 | 提升/下降/持平（当前句级指标） | 初稿停止数 |','|---|---:|---|---:|']
    for arm,e in external['results'].items():
        m=e['fixed_threshold_with_length_gate'];lines.append(f"| {arm} | {m['accepted']} | {m['improve']}/{m['degrade']}/{m['tie']} | {e['initial_stop_at_0_6']} |")
    lines += ['', '按历史验证集校准的阈值，absolute=0.1、pairwise=0.075，在此外部候选中均接受 0 次；legacy=0.02 接受 5 次且全部下降。提高阈值避免了部分错误，但没有验证出可用的在线改进收益。', '', '## 学习曲线与限制','', '![learning curves](bert_feedback_learning_curves.png)', '', '这是单一随机种子、有限数据预算的受控诊断，不能证明架构上限，也没有排除更大数据、不同排序目标和超参数的收益。排序目标仍依赖同一 BLEU 标签；标签与真实质量不一致的问题不会被排序损失自动解决。停止阈值 0.6 仅观察，尚未按新权重校准。', '', '原始协议、划分、逐轮曲线、权重、预测、校准表和外部复评见 `../runs/bert_feedback_study/`。']
    report=ROOT.parent.parent/'reports/BERT_FEEDBACK_STUDY.md';report.write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    for arm in ['absolute','pairwise']:
        history=json.loads((ROOT/f'{arm}_learning_curve.json').read_text()); epochs=[h['epoch'] for h in history]
        for ax,key,title in zip(axes,['absolute_mse','gain_spearman','accept_precision'],['Validation absolute MSE','Validation gain Spearman','Acceptance precision (threshold 0.01)']):
            ax.plot(epochs,[h['validation'][key] for h in history],marker='o',label=arm);ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.25)
    axes[2].axhline(.8,color='gray',linestyle='--',label='Calibration target'); axes[0].legend();fig.tight_layout();fig.savefig(report.parent/'bert_feedback_learning_curves.png',dpi=160);plt.close(fig)
    print(report)
if __name__=='__main__':main()
