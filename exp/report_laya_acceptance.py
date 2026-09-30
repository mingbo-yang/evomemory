"""Render frozen results and learning curves; performs no model selection."""
import json
from pathlib import Path

OUT = Path('/mnt/huawei/ymb/model/laya-multilingual-acceptance-enzh-v1')
ROOT = Path(__file__).resolve().parent


def run():
    report = json.loads((OUT/'evaluation_report.json').read_text())
    history = json.loads((OUT/'history.json').read_text())
    base_dev = json.loads((OUT/'base_development.json').read_text())
    diagnostics = json.loads((OUT/'diagnostics.json').read_text())
    lock = json.loads((OUT/'evaluation_lock.json').read_text())
    passed = report['verifier_gate_passed']
    final = report['models']['finetuned']
    policy = {
        'question_type': 'choice', 'label_order': ['Better', 'Tie', 'Worse'],
        'temperature': lock['models']['finetuned']['temperature'],
        'threshold': lock['models']['finetuned']['threshold'],
        'verifier_gate_passed': passed,
        'deployment_status': 'requires_fixed_memory_pipeline_validation' if passed else 'blocked_failed_verifier_gate',
        'eligibility': 'nonidentical, nonempty candidate; NFKC casefold whitespace-free length <= 1.5 * current length + 8',
        'input_fields': ['source', 'current', 'candidate'],
        'feedback_available_to_model': False,
        'label_semantics': 'weak reference-metric feedback; no independent semantic gold',
        'full_static_full_online_validated': False,
    }
    (OUT/'best/acceptance_policy.json').write_text(json.dumps(policy, indent=2, ensure_ascii=False)+'\n')
    lines = [
        '# Laya multilingual 训练与验证结果', '',
        f"4 轮训练和独立验证已完成。使用 convaiinnovations/laya-multilingual，开发集选择第 {report['selected_epoch']} 轮权重。",
        f"验证结论：{'通过 verifier 门槛，仍须固定库完整流程验证' if passed else '未通过 verifier 门槛，暂不能用于主实验'}。", '',
        '训练集为 20,002 个候选比较样本；开发、概率校准、阈值校准和最终测试按原文隔离。',
        f"最终测试覆盖 {report['test_sources']} 个原文、{report['test_pairs']} 个比较，其中 {diagnostics['changed_pairs']} 个确实发生修改。",
        '选择权重和阈值后才解封测试；模型输入仅含原文、当前答案和候选答案。', '',
        '| 模型 | 实际修改样本 Better AUC | 三分类 Macro-F1 | 校准后接受数 | 覆盖率 | 接受精度 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    def f(v): return '无定义' if v is None else f'{v:.4f}'
    for name, title in [('base', '原始 Laya multilingual'), ('finetuned', '微调 Laya multilingual')]:
        m = report['models'][name]; a = m['calibrated_acceptance']
        lines.append(f"| {title} | {f(m['auc_better_changed'])} | {f(m['macro_f1'])} | {a['accepted']} | {a['coverage']:.2%} | {f(a['precision'])} |")
    lines += ['', '三分类指标包含完全相同的答案对，不能代替实际修改样本上的判别能力。',
              '冻结的阈值要求：校准集接受精度至少 80%、覆盖率至少 10%、平均反馈增量为正；测试还要求按原文聚类 bootstrap 的增量置信区间下界大于 0。',
              '没有合格阈值时关闭接受；零接受属于失败，精度无定义，不能报告为高准确率。', '']
    interval = diagnostics['models']['finetuned']['changed_auc_bootstrap']['ci95']
    search = json.loads((OUT/'finetuned_threshold_search.json').read_text())
    feasible_coverage = [r for r in search['grid'] if r['coverage'] >= .1]
    max_precision = max((r['precision'] for r in feasible_coverage), default=None)
    diagnostic_acceptance = final['default_acceptance']
    lines += [f"微调模型实际修改 AUC 的 95% 原文聚类 bootstrap 区间为 [{interval[0]:.4f}, {interval[1]:.4f}]。",
              f"在独立阈值校准集中，满足至少 10% 覆盖率的网格阈值最高精度为 {f(max_precision)}，仍未满足 80% 标准。",
              f"仅作诊断：校准后 pBetter ≥ 0.5 时，测试接受 {diagnostic_acceptance['accepted']} 对，精度 {f(diagnostic_acceptance['precision'])}，平均代理反馈增量 {f(diagnostic_acceptance['mean_delta'])}（0–1 尺度）。这不是一个通过验证的工作阈值。", '']
    if 'bert_supplementary' in diagnostics:
        b = diagnostics['bert_supplementary']; a = b['acceptance']
        lines += [f"补充对照：旧 BERT 在相同测试对上 Better AUC={b['auc_better_changed']:.4f}；原接受策略接受 {a['accepted']} 对，精度 {f(a['precision'])}，平均反馈增量 {f(a['mean_delta'])}。",
                  '该 BERT 对照沿用原有增益阈值与长度限制，不重训、不调参；不是完整在线流程的对比。', '']
    lines += ['## 结果应如何解释', '',
              '本次标签来自原有参考答案指标的分差，ε=0.01；它们是弱反馈标签，不是独立人工语义判断。已有训练集抽查发现同义改写造成较大分差，因此本次实验不能单独证明翻译质量判断能力，也不能把失败直接归因为某个模型架构。',
              '已完成的是分类器训练、校准、测试与保存后的官方接口一致性检查。full_static/full_online 主实验尚未在本模型上验证。', '',
              '## 保存位置', '',
              f'- 最终模型：`{OUT / "best"}`',
              f'- 原始预训练模型：`{OUT.parent / "laya-multilingual"}`',
              '- `training_protocol.json`：训练及评估规则。',
              '- `evaluation_lock.json`：测试前冻结的模型、温度和阈值。',
              '- `evaluation_report.json`：主验证指标和置信区间。',
              '- `diagnostics.json`：修改样本、混淆矩阵及旧 BERT 补充对照。',
              '- `best/acceptance_policy.json`：接受阈值、验证状态及使用限制。',
              '- `history.json`、`training.log`：逐轮指标及训练日志。', '',
              '![开发集学习曲线](learning_curves.png)', '']
    text = '\n'.join(lines)
    (OUT/'RESULTS_ZH.md').write_text(text)
    (ROOT/'reports/LAYA_ACCEPTANCE_RESULTS_ZH.md').write_text(text.replace('(learning_curves.png)', f'({OUT / "learning_curves.png"})'))
    (OUT/'best/README.md').write_text('\n'.join([
        '# Laya multilingual EN→ZH acceptance', '',
        'Base: convaiinnovations/laya-multilingual (jhu-clsp/mmBERT-base).',
        'This directory contains the development-selected, temperature-calibrated model in official Laya Agent format.',
        f'Selected epoch: {report["selected_epoch"]}. Verifier gate passed: {passed}.',
        f'Deployment status: {policy["deployment_status"]}.', '',
        'Read acceptance_policy.json before applying the output. A null threshold disables acceptance and means calibration failed to find a usable operating point.',
        'Training targets are weak reference-metric differences, not independently annotated semantic quality. Only source/current/candidate are model inputs.',
        'No fixed-memory or online-memory task rollout has been validated with this checkpoint.',
        'See ../RESULTS_ZH.md and ../evaluation_report.json for all evaluation details.', '',
    ]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    axes[0].plot(epochs, [h['ce'] for h in history], 'o-', label='Training CE')
    axes[0].plot(epochs, [h['development']['nll'] for h in history], 's-', label='Development NLL')
    axes[0].set_title('Learning curves'); axes[0].set_ylabel('Loss'); axes[0].legend()
    axes[1].plot([0]+epochs, [base_dev['auc_better_changed']]+[h['development']['auc_better_changed'] for h in history], 'o-')
    axes[1].axhline(.5, color='gray', ls='--', label='Random ranking')
    axes[1].set_ylim(0, 1); axes[1].set_title('Development: changed pairs'); axes[1].set_ylabel('Better AUC'); axes[1].legend()
    axes[2].plot([0]+epochs, [base_dev['top10pct']['precision']]+[h['development']['top10pct']['precision'] for h in history], 'o-')
    axes[2].axhline(.8, color='gray', ls='--', label='Operating precision target')
    axes[2].set_ylim(0, 1); axes[2].set_title('Development: top 10% eligible rank'); axes[2].set_ylabel('Better precision'); axes[2].legend()
    for ax in axes:
        ax.set_xlabel('Epoch (0 = pretrained)'); ax.set_xticks(range(5)); ax.grid(alpha=.2)
    fig.suptitle('Laya multilingual EN→ZH acceptance — weak feedback targets')
    fig.savefig(OUT/'learning_curves.png', dpi=180)
    fig.savefig(OUT/'learning_curves.pdf')
    plt.close(fig)
    print(OUT/'RESULTS_ZH.md')


if __name__ == '__main__':
    run()
