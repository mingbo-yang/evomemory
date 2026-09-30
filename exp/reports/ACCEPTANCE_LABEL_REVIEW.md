# Acceptance data: label preparation review

All 80 presampled training pairs were reviewed by the Codex assistant. This is AI-assisted qualitative review, not human annotation, not blinded, and not an independent gold evaluation. Twenty pairs came from each absolute-delta bin; the sample is not population-proportional. Do not extrapolate its agreement counts into whole-dataset label precision.

## Findings

Assistant judgments: {'Tie': 63, 'Uncertain': 4, 'Better': 8, 'Worse': 5}.

Simple lexical edits can move the existing sentence feedback by more than 0.05 and even 0.68. Enlarging epsilon does not establish reliable semantic labels. Shared translation errors do not imply a relative improvement/degradation; several pairs preserve the same incorrect translation.

Three training sources have a clearly misaligned or incomplete reference. Their source/reference fields were checked against the original parquet and match exactly; no collector index error was found. All 11 associated pairs were quarantined in the derived dataset. Seven other invalid pairs were filtered. Originals are preserved. This is not an exhaustive reference-quality audit of all sources.

## Candidate-margin diagnostics

Counts below are descriptive and do not freeze a label rule. Review agreement excludes four Uncertain judgments and three misaligned-reference cases, leaving 73 pairs.

| ε | Train Better | Train Tie | Train Worse | Review matching labels / 73 | Review matching directional labels / predicted directional |
|---:|---:|---:|---:|---:|---:|
| 0 | 5166 | 6978 | 7858 | 11/73 | 6/68 |
| 0.005 | 4063 | 9581 | 6358 | 20/73 | 6/59 |
| 0.01 | 3363 | 10996 | 5643 | 22/73 | 6/55 |
| 0.02 | 2521 | 12710 | 4771 | 30/73 | 5/43 |
| 0.03 | 2003 | 13958 | 4041 | 34/73 | 4/37 |
| 0.05 | 1298 | 15769 | 2935 | 42/73 | 1/20 |

The greater overall match for a larger margin mainly reflects converting cases into Tie. It should not be mistaken for evidence of high-quality Better/Worse supervision. A weak-feedback export can still support a proxy-objective experiment if explicitly chosen; a semantic verifier needs a better justified label source.

## Per-pair notes

### 0: wmt19_en_zh/acceptance_v1/train/029871

- AI-assisted label: Tie; delta=-0.00193025; reference issue=None
- Source: But this does not guarantee that future drops will have a similar outcome.
- Before: 但这并不能保证未来下跌将产生类似的结果。
- Candidate: 但这并不能保证未来下跌将有相似的结果。
- Reference (review only): 但这并不能保证未来的房价下跌会带来同样的后果。
- Reason: 相似/类似、有/产生为近义改写，未修正实质信息。

### 1: wmt19_en_zh/acceptance_v1/train/009982

- AI-assisted label: Tie; delta=0.00009912; reference issue=None
- Source: For example, by blocking reforms to International Monetary Fund governance that were agreed in 2010, Congress has damaged, perhaps irreparably, the legitimacy and relevance of the Bretton Woods institutions.
- Before: 例如，通过阻止2010年达成的国际货币基金组织治理改革，国会可能已经严重损害，甚至不可挽回地损害了布雷顿森林体系机构的合法性与相关性。
- Candidate: 例如，通过阻止2010年达成的国际货币基金组织治理改革，国会可能已严重损害，甚至不可挽回地损害了布雷顿森林体系机构的合法性与相关性。
- Reference (review only): 比如，国会阻挠在2010年就形成一致的国际货币基金组织（IMF）治理改革，这可能对布雷顿森林机构的合法性和重要性造成了不可挽回的伤害。
- Reason: 已经缩成已，内容和语法基本等价。

### 2: wmt19_en_zh/acceptance_v1/train/000194

- AI-assisted label: Tie; delta=0.00261192; reference issue=None
- Source: Eurozone policymakers, by contrast, rejected monetary stimulus and implemented fiscal austerity measures, while ignoring the deepening distress of their banks.
- Before: 欧元区政策制定者相比之下拒绝采取货币刺激措施，实施财政紧缩政策，同时忽视了其银行日益加剧的困境。
- Candidate: 欧元区政策制定者相比之下拒绝采取货币刺激措施，转而实施财政紧缩政策，同时忽视了其银行日益加剧的困境。
- Reference (review only): 相反，欧洲决策者拒绝采取货币刺激而实施了财政紧缩，无视其银行压力的加剧。
- Reason: 加入转而，未改变货币与财政政策的主要信息。

### 3: wmt19_en_zh/acceptance_v1/train/017001

- AI-assisted label: Tie; delta=-0.00000118; reference issue=None
- Source: People wanted to know whether board members were actually doing their jobs or just filling seats and collecting a nice fee.
- Before: 人们想知道董事们究竟是真正履行职责，还是仅仅占据席位并领取一笔不菲的报酬。
- Candidate: 人们想知道董事们究竟是真正履职尽责，还是仅仅占据席位并领取高额报酬。
- Reference (review only): 人们希望了解董事会成员是否实际完成了他们的任务，抑或只是尸位素餐，领到一大笔费用。
- Reason: 履行职责/履职尽责、不菲/高额为措辞变化。

### 4: wmt19_en_zh/acceptance_v1/train/033129

- AI-assisted label: Tie; delta=0.00637413; reference issue=None
- Source: The US already conducts more military exercises with India than with any other country.
- Before: 美国目前与印度进行的军事演习数量已经超过了与其他任何国家的演习数量。
- Candidate: 美国目前与印度举行的军事演习数量，已超过与其他任何国家的演习数量。
- Reference (review only): 美国与印度的联合军演比其他任何国家都要多。
- Reason: 进行改举行并删重复成分，未见实质优劣。

### 5: wmt19_en_zh/acceptance_v1/train/018316

- AI-assisted label: Tie; delta=0.00000009; reference issue=mismatch
- Source: If the US recession were – as most likely - to be over at the end of the year, as is likely, it will have been three times as long and about fives times as deep – in term of the cumulative decline in output – as the previous two.
- Before: 如果美国经济衰退——正如最有可能的情况——在年底结束，那么它持续的时间将是此前两次的三倍，产出累计下降的程度则将是此前两次的五倍。
- Candidate: 如果美国经济衰退——正如最有可能的情况——在年底结束，那么其持续时间将是此前两次的三倍，产出累计下降的程度则将是此前两次的五倍。
- Reference (review only): 时间的跨度是24个月。 美国的衰退也将拖累整个世界的经济。
- Reason: 它持续的时间改其持续时间，候选语义未变；参考遗漏三倍、五倍及年底等核心信息并加入世界经济内容。

### 6: wmt19_en_zh/acceptance_v1/train/030766

- AI-assisted label: Tie; delta=0.00000000; reference issue=None
- Source: Only if Nexus (like the Nexus-6 replicants in the 1982 film Blade Runner) achieves that leap will “he” have earned the “right” to be thought of as distinct from the harvester he operates.
- Before: 只有当尼克斯（如同1982年电影《银翼杀手》中的Nexus-6复制人）实现这一飞跃，“他”才配拥有被视作与他操作的收割者有所区别的“权利”。
- Candidate: 只有当尼克斯（如同1982年电影《银翼杀手》中的Nexus-6仿生人）实现这一飞跃，“他”才配拥有被视作与他操作的收割者有所区别的“权利”。
- Reference (review only): 仅当Nexus（就像1982年的电影《银翼杀手》中的Nexus-6克隆人）实现这一飞越 ， “ 他”才能获得可视为不同于他所操作的收割机的“权利 ” 。
- Reason: 复制人/仿生人是译名选择，当前证据不足以视为实质质量差异。

### 7: wmt19_en_zh/acceptance_v1/train/026704

- AI-assisted label: Tie; delta=0.00000089; reference issue=None
- Source: Governments will need to distance themselves from the companies that fail to do their part.
- Before: 政府将需要与那些未能履行自身职责的公司保持距离。
- Candidate: 政府需要与未能履行自身职责的公司保持距离。
- Reference (review only): 政府必须与不履行义务的企业坚决疏远。
- Reason: 将需要/需要的轻微时态变化，不足以稳定判定方向。

### 8: wmt19_en_zh/acceptance_v1/train/028944

- AI-assisted label: Tie; delta=0.00005162; reference issue=None
- Source: There is no analogous set of floodlights being deployed to erase the shadow that is currently being cast by the lesser depression.
- Before: 目前并没有类似的聚光灯被用来消除由较弱的低压系统所投下的阴影。
- Candidate: 目前尚无类似的聚光灯被用来消除由较弱低压系统所投下的阴影。
- Reference (review only): 如今，小萧条也留下了阴影，但我们却找不到类似的强光。
- Reason: 两者均把经济萧条误译为低压系统，候选未修复这一共同错误。

### 9: wmt19_en_zh/acceptance_v1/train/006354

- AI-assisted label: Uncertain; delta=0.00336732; reference issue=None
- Source: Such investments could generate an estimated five million “green-collar” jobs, provide a shot in the arm for the construction and engineering industries, and get America back into the equally serious business of combating climate change and achieving energy security.
- Before: 此类投资可能创造约五百万个“绿色岗位”，为建筑业和工程行业注入新的活力，并使美国重新致力于应对气候变化和实现能源安全这一同样严肃的事业。
- Candidate: 此类投资可能创造约五百万个“绿色岗位”，为建筑业和工程行业注入新的活力，并使美国重新投入应对气候变化和实现能源安全这一同样严峻的事业。
- Reference (review only): 这些投资可以产生大约五百万“绿领”工作，给建筑和工程产业注入一剂强心针并且把美国带回到气候变化以及实现能源安全的同样重要的任务上来。
- Reason: 严肃改严峻可能偏离serious business，但措辞本身均不理想，净改善不确定。

### 10: wmt19_en_zh/acceptance_v1/train/004479

- AI-assisted label: Tie; delta=0.00000000; reference issue=None
- Source: Industrialization alone cannot resolve the migration crisis, but it can address one root cause, by creating jobs in the countries of origin.
- Before: 单靠工业化无法解决移民危机，但它可以解决其中一个根源问题——在移民来源国创造就业机会。
- Candidate: 单靠工业化无法解决移民危机，但它可以解决其中一个根源问题，即在移民来源国创造就业机会。
- Reference (review only): 仅靠工业化无法解决移民危机，但却可以通过在原籍国创造就业机会的方式来化解其中一个根本原因。
- Reason: 破折号改即，均保留把创造就业解释成根源问题的共同表达缺陷。

### 11: wmt19_en_zh/acceptance_v1/train/001648

- AI-assisted label: Tie; delta=0.00000000; reference issue=None
- Source: LONDON – Reading Barack Obama’s Dreams from My Father, the US president’s beautifully written reflections on his early life and identity, most people are struck by his cool and intellectual approach.
- Before: 伦敦——阅读巴拉克·奥巴马的《我父亲的梦想》，这本美国总统关于他早年生活和身份的优美写照，大多数人会被他冷静而富有智慧的思维方式所打动。
- Candidate: 伦敦——阅读巴拉克·奥巴马的《我父亲的梦想》，这本美国总统关于他早年生活和身份的优美作品，大多数人会被他冷静而富有智慧的思维方式所打动。
- Reference (review only): 发自伦敦 — — 当读到《我父亲的梦想》这本美国总统奥巴马用优美的文笔叙述自己早年生活以及个人身份定位的自传时，许多人都会惊讶于他那种冷静而理智的做事方式。
- Reason: 写照改作品，未清楚修复reflections的含义，倾向等价措辞。

### 12: wmt19_en_zh/acceptance_v1/train/021273

- AI-assisted label: Tie; delta=-0.00279635; reference issue=None
- Source: It helped that as ASEAN was getting off the ground in the late 1960s and early 1970s, the strategic interests of America, China, and the bloc’s members converged.
- Before: 东盟在20世纪60年代末和70年代初刚刚起步时，美国、中国以及该组织成员国的战略利益趋于一致，这在一定程度上帮助了局势的发展。
- Candidate: 东盟在20世纪60年代末至70年代初刚刚起步时，美国、中国以及该组织成员国的战略利益趋于一致，这在一定程度上促进了局势的发展。
- Reference (review only): 东盟在20世纪60年代末、70年代初起航的时机也恰到好处，美国、中国和域内成员国的战略利益形成了趋同。
- Reason: 帮助发展/促进发展、和/至主要是表达变化。

### 13: wmt19_en_zh/acceptance_v1/train/014758

- AI-assisted label: Tie; delta=0.00228162; reference issue=extra_context
- Source: Monetary and fiscal tightening aimed at curtailing credit expansion runs the risk of undermining GDP growth and reducing Chinese demand for commodities.
- Before: 旨在抑制信贷扩张的紧缩货币政策和财政政策可能会损害GDP增长，并降低中国对大宗商品的需求。
- Candidate: 旨在抑制信贷扩张的紧缩性货币政策和财政政策，可能拖累GDP增长，并降低中国对大宗商品的需求。
- Reference (review only): 唯一实际的国际溢出效应来自抑制信贷扩张的货币政策和财政紧缩政策有可能降低GDP增长，从而抑制中国对大宗商品的需求。
- Reason: 损害增长改拖累增长、补紧缩性，未见明确质量方向；参考另加国际溢出效应说明。

### 14: wmt19_en_zh/acceptance_v1/train/016257

- AI-assisted label: Tie; delta=-0.00414634; reference issue=None
- Source: In the US, for example, George Washington, Abraham Lincoln, and Franklin D. Roosevelt succeeded in strengthening their country because they recognized the need to address internal divisions in the light of America’s core values, global position, and long-term goals.
- Before: 例如在美国，乔治·华盛顿、亚伯拉罕·林肯和富兰克林·D·罗斯福成功地加强了本国实力，因为他们认识到，根据美国的核心价值观、国际地位和长远目标，必须解决国内分歧。
- Candidate: 例如在美国，乔治·华盛顿、亚伯拉罕·林肯和富兰克林·D·罗斯福成功增强了国家实力，因为他们认识到，依据美国的核心价值观、国际地位和长远目标，必须解决国内分歧。
- Reference (review only): 比如，在美国，华盛顿、林肯和富兰克林·罗斯福都成功地处理了危机并让美国变得更强大，因为他们都能够认识到解决内部分歧必须兼顾国家核心价值观、国家在全球的地位、以及国家长期发展目标。
- Reason: 加强本国实力/增强国家实力、根据/依据近义。

### 15: wmt19_en_zh/acceptance_v1/train/001732

- AI-assisted label: Tie; delta=0.00000000; reference issue=None
- Source: This narrative – the standard Keynesian line, modified for a debt crisis – is the one to which most government officials, central bankers, and Wall Street economists have subscribed, and needs little elaboration.
- Before: 这一叙事——即针对债务危机而调整的标准凯恩斯主义观点——已成为大多数政府官员、央行行长和华尔街经济学家所接受的共识，几乎无需进一步阐述。
- Candidate: 这一叙事——即针对债务危机调整后的标准凯恩斯主义观点——已成为大多数政府官员、央行行长和华尔街经济学家所接受的共识，几乎无需进一步阐述。
- Reference (review only): 这种观点是标准的凯恩斯主义，只是根据债务危机做了些许修正。 该观点受到了大部分政府官员、中央银行和华尔街经济学家的支持，已无需再赘述了。
- Reason: 针对债务危机而调整/调整后的结构等价。

### 16: wmt19_en_zh/acceptance_v1/train/038943

- AI-assisted label: Tie; delta=-0.00949269; reference issue=None
- Source: California is the most diverse US state; more than half of its 37 million people are non-white.
- Before: 加利福尼亚州是美国人口最多样化的州；其3700万人口中，超过一半是非白人。
- Candidate: 加利福尼亚州是美国人口最多样化的州；其3700万人口中，超过半数为非白人。
- Reference (review only): 加州是美国最多元化的州; 其3700万人口中非白人占了一半以上。
- Reason: 超过一半是/超过半数为，事实与数字完全一致。

### 17: wmt19_en_zh/acceptance_v1/train/035364

- AI-assisted label: Tie; delta=0.00000000; reference issue=None
- Source: Let’s consider them in turn.
- Before: 让我们逐一考虑它们。
- Candidate: 让我们依次考虑它们。
- Reference (review only): 现在让我们轮流来检视这几点：
- Reason: 逐一/依次意思一致。

### 18: wmt19_en_zh/acceptance_v1/train/000535

- AI-assisted label: Better; delta=-0.00813817; reference issue=None
- Source: Thus, Putin is more reluctant than ever to offer territorial concessions, lest it tarnish his domestic image as a staunch defender of Russian national interests.
- Before: 因此，普京比以往更加不愿做出领土让步，以免损害其在国内作为坚定维护俄罗斯国家利益捍卫者的形象。
- Candidate: 因此，普京比以往更加不愿做出领土让步，以免损害其在国内作为坚定维护俄罗斯国家利益的捍卫者的形象。
- Reference (review only): 因此，普京更加不愿意在领土问题上让步，唯恐玷污其俄罗斯国家利益坚定捍卫者的国内形象。
- Reason: 补上的修复作为……捍卫者结构中的缺词，使句法更完整，幅度较小。

### 19: wmt19_en_zh/acceptance_v1/train/028491

- AI-assisted label: Worse; delta=0.00586483; reference issue=None
- Source: By contrast, adherents of the Dr. Doom school warned that global imbalances were an accident waiting to happen.
- Before: 相比之下，德鲁伊博士学派的追随者警告称，全球失衡只是迟早会发生的问题。
- Candidate: 相比之下，德鲁伊博士学派的追随者警告称，全球失衡迟早会发生。
- Reference (review only): 相反，悲观派警告说，全球失衡是一场随时都有可能发生的事故。
- Reason: 从失衡迟早引发问题滑向失衡本身尚未发生，进一步偏离原文事故隐喻。

### 20: wmt19_en_zh/acceptance_v1/train/031066

- AI-assisted label: Tie; delta=-0.01230461; reference issue=None
- Source: Biopharming has much to offer us.
- Before: 生物制药对我们有很多益处。
- Candidate: 生物制药对我们有许多好处。
- Reference (review only): 生物制药能给我们带来很大的裨益。
- Reason: 很多益处/许多好处等价。

### 21: wmt19_en_zh/acceptance_v1/train/035239

- AI-assisted label: Tie; delta=-0.02462630; reference issue=None
- Source: The UK’s demand to weigh in on EU decisions can and should be accommodated, with experts from the UK included in the committees that decide trade policy.
- Before: 英国要求参与欧盟决策的需求是可以并且应当得到满足的，英国专家应被纳入决定贸易政策的委员会。
- Candidate: 英国要求参与欧盟决策的需求可以并应当得到满足，英国专家应被纳入制定贸易政策的委员会。
- Reference (review only): 英国参与欧盟决定的要求应该真诚以对，让来自英国的专家加入决定贸易政策的委员会。
- Reason: 压缩措辞，决定/制定贸易政策未形成清晰语义质量差异。

### 22: wmt19_en_zh/acceptance_v1/train/026440

- AI-assisted label: Tie; delta=-0.01377459; reference issue=None
- Source: Indeed, sharing-economy data can help governments identify those workers at greatest risk and support their retraining.
- Before: 确实，共享经济数据可以帮助政府识别风险最大的工人，并支持他们的再培训。
- Candidate: 确实，共享经济数据可帮助政府识别风险最高的劳动者，并支持其再培训。
- Reference (review only): 事实上，共享经济数据有助于政府识别风险最高的工人，为他们提供再培训支持。
- Reason: 工人/劳动者、可以/可、他们/其是表达选择。

### 23: wmt19_en_zh/acceptance_v1/train/031908

- AI-assisted label: Better; delta=-0.01432544; reference issue=None
- Source: That model was transferred to many other countries, and became the bedrock on which modern financial stability was built.
- Before: 该模式被转移到许多其他国家，并成为现代金融稳定的基石。
- Candidate: 该模式被引入许多其他国家，并成为现代金融稳定的基石。
- Reference (review only): 那种模式传播到许多国家，成为近代财政稳健的基石。
- Reason: 模式被引入比被转移更符合制度传播的语境和汉语搭配。

### 24: wmt19_en_zh/acceptance_v1/train/012116

- AI-assisted label: Worse; delta=-0.02938366; reference issue=None
- Source: Although Myanmar is an overwhelmingly Buddhist country, dozens of religious and ethnic communities have long lived there side by side.
- Before: 尽管缅甸是一个以佛教为主的国家，但数十个宗教和民族群体长期以来一直和睦共处。
- Candidate: 尽管缅甸是一个佛教国家，但数十个宗教和民族群体长期以来和睦共处。
- Reference (review only): 尽管缅甸是个佛教占压倒性优势的国家，但长期以来，数十种宗教和民族一直共处着。
- Reason: 删除以……为主，削弱overwhelmingly表达的人口占多数限定。

### 25: wmt19_en_zh/acceptance_v1/train/013159

- AI-assisted label: Tie; delta=-0.01219642; reference issue=None
- Source: So why have the Europeans been unable to sustain improvements in both foundations of growth – employment and productivity – at the same time?
- Before: 那么，为什么欧洲无法同时维持就业和生产率这两个增长基础的改善呢？
- Candidate: 那么，为什么欧洲无法同时维持就业和生产率这两个增长支柱的改善呢？
- Reference (review only): 那么，为何欧洲无法同时在增长的两大基础、也就是就业和生产效率上维持增长呢？
- Reason: 增长基础/支柱是等价比喻。

### 26: wmt19_en_zh/acceptance_v1/train/032431

- AI-assisted label: Tie; delta=0.01284274; reference issue=missing_sentence
- Source: In the United States, the 30-year Treasury bond yield reached a record low (since the Federal Reserve series began in 1972) of 2.25% on January 30. The yield on the United Kingdom's 30-year government bond fell to 2.04% on the same day.
- Before: 在美国，30年期美国国债收益率于1月30日降至自联邦储备系统于1972年开始记录以来的最低水平2.25%。同一天，英国30年期政府债券收益率也降至2.04%。
- Candidate: 在美国，30年期国债收益率于1月30日降至自联邦储备系统1972年开始记录以来的最低水平2.25%。同日，英国30年期政府债券收益率也降至2.04%。
- Reference (review only): 在美国，30年国债收益率在1月30日创下了2.25%的历史低点（自1972年美联储系列债券诞生以来 ） 。
- Reason: 删重复美国、同一天改同日，事实不变；参考完全缺少英国收益率一句。

### 27: wmt19_en_zh/acceptance_v1/train/003060

- AI-assisted label: Better; delta=-0.01230569; reference issue=None
- Source: To be sure, the EU recognized its need for a coherent strategy, and attempted to resolve it by establishing the European External Action Service and the position of High Representative for Foreign Affairs.
- Before: 可以肯定的是，欧盟意识到了自身需要一个协调一致的策略，并试图通过设立欧洲外务行动署和欧盟外交与安全政策高级代表职位来解决这一问题。
- Candidate: 可以肯定的是，欧盟意识到了需要一个协调一致的策略，并试图通过设立欧洲对外行动署和欧盟外交与安全政策高级代表职位来加以解决。
- Reference (review only): 平心而论，欧盟认识到需要统一的战略，也试图通过设立欧洲对外行动署（European External Action Service，EEAS）和对外事务高级代表解决这一问题。
- Reason: 欧洲外务行动署改欧洲对外行动署，修复机构名称表达。

### 28: wmt19_en_zh/acceptance_v1/train/021122

- AI-assisted label: Tie; delta=-0.02701443; reference issue=None
- Source: The first round of QE was unambiguously beneficial, because it minimized, or even eliminated, the tail risk of a global depression after the collapse of Lehman Brothers in September 2008.
- Before: 量化宽松（QE）的第一轮作用明确积极，因为它在2008年9月雷曼兄弟破产后，最大限度地降低了甚至消除了全球经济大萧条的尾部风险。
- Candidate: 量化宽松（QE）首轮作用明确积极，因其在2008年9月雷曼兄弟破产后最大限度地降低了甚至消除了全球性经济大萧条的尾部风险。
- Reference (review only): 第一轮QE毫无疑问利大于弊，因为它最小化甚至消除了2008年9月雷曼兄弟公司倒闭后全球萧条的尾部风险。
- Reason: 第一轮/首轮、因为/因其、全球/全球性均未改变事实。

### 29: wmt19_en_zh/acceptance_v1/train/010882

- AI-assisted label: Tie; delta=-0.01544747; reference issue=None
- Source: The vulnerability caused by higher public debt over the next few decades is simply ignored.
- Before: 未来几十年由公共债务增加所带来的脆弱性被简单地忽视了。
- Candidate: 未来几十年公共债务上升带来的脆弱性被简单忽视了。
- Reference (review only): 而未来数十年公债增加所引起的脆弱性完全不在他们的考虑范围内。
- Reason: 债务增加/上升、删结构助词，基本等价。

### 30: wmt19_en_zh/acceptance_v1/train/011145

- AI-assisted label: Tie; delta=-0.01366536; reference issue=None
- Source: Other countries have done so.
- Before: 其他国家也这样做过。
- Candidate: 其他国家也这么做了。
- Reference (review only): 其他国家已经找到了这样的道路。
- Reason: 做过/这么做了均可表达have done so；上下文不足以确定时态差别的优劣。

### 31: wmt19_en_zh/acceptance_v1/train/024354

- AI-assisted label: Tie; delta=0.02539430; reference issue=None
- Source: Russia, despite Siberia’s massive oil and gas riches, isn’t even investing enough to support healthy growth in its energy industries, much less human development in the country’s impoverished areas (including hapless Siberia).
- Before: 尽管西伯利亚拥有丰富的石油和天然气资源，俄罗斯甚至未能投入足够的资金来支持能源产业的健康发展，更不用说在该国贫困地区的社会发展（包括无助的西伯利亚）了。
- Candidate: 俄罗斯尽管拥有西伯利亚丰富的石油和天然气资源，却仍未投入足够资金支持能源产业的健康发展，更不用说在该国贫困地区的社会发展（包括处境艰难的西伯利亚）了。
- Reference (review only): 俄罗斯尽管有着西伯利亚的巨大石油和天然气储量，却连支撑其能源工业健康发展的投资水平也保证不了，更别说在该国贫困地区（包括不幸的西伯利亚）的人力资源发展上的投资了。
- Reason: 调整语序、无助改处境艰难，轻微润色；human development的共同不精确译法仍在。

### 32: wmt19_en_zh/acceptance_v1/train/034218

- AI-assisted label: Tie; delta=0.01762192; reference issue=None
- Source: Peer-to-peer lending has expanded faster than elsewhere in Europe.
- Before: 点对点借贷在欧洲扩张速度超过其他地区。
- Candidate: 点对点借贷在欧洲的扩张速度超过了其他地区。
- Reference (review only): 点对点贷款的扩张速度快于欧洲其他地区。
- Reason: 加的和了，仍保留把欧洲内部比较误解为欧洲与其他地区比较的共同问题。

### 33: wmt19_en_zh/acceptance_v1/train/035677

- AI-assisted label: Tie; delta=-0.01747313; reference issue=extra_sentence
- Source: Agreement is also possible on giving national parliaments a greater role in guiding EU legislation.
- Before: 也可以达成协议，赋予国家议会更大的作用，以指导欧盟立法。
- Candidate: 也可以达成协议，赋予国家议会更大作用，以指导欧盟立法。
- Reference (review only): 在给予国家议会更大的欧盟立法指导作用方面也有可能达成一致，尽管卡梅伦提出的允许国家议会对欧盟法律亮“红牌”的建议走得太远了。
- Reason: 删一个的无实质变化；参考多出卡梅伦红牌建议的完整分句。

### 34: wmt19_en_zh/acceptance_v1/train/032556

- AI-assisted label: Worse; delta=-0.01087499; reference issue=None
- Source: Although interest-rate spreads for Italian and Spanish ten-year bonds relative to German bonds briefly jumped 30-50 basis points after the results were announced, they then eased to 300-350 basis points, compared to 500-600 basis points before the ECB’s decision to establish its “outright monetary transactions” program.
- Before: 尽管意大利和西班牙十年期债券相对于德国债券的利率利差在公布结果后短暂上升了30至50个基点，但随后回落至300至350个基点，与欧洲央行决定启动“直接货币交易”计划前的500至600个基点相比有所下降。
- Candidate: 尽管意大利和西班牙十年期债券相对于德国债券的利率利差在结果公布后短暂上升了30至50个基点，随后回落至300至350个基点，与欧洲央行决定启动“定向货币交易”计划前的500至600个基点相比有所回落。
- Reference (review only): 尽管意大利和西班牙十年期债券相对德国的利差在结果公布后扩大了30—50个基点，但随后减小至300—350个基点，而在欧洲央行的“直接货币交易”计划决定之前，利差高达500—600个基点。
- Reason: 直接货币交易改定向货币交易，偏离outright的术语含义。

### 35: wmt19_en_zh/acceptance_v1/train/017221

- AI-assisted label: Tie; delta=-0.01949528; reference issue=None
- Source: After all, Asia has been on such a roll in recent years that far too many believe that the region can shrug off almost anything that the rest of the world dishes out.
- Before: 毕竟，近年来亚洲发展势头强劲，以至于许多人相信该地区几乎可以承受世界其他地方施加的任何压力。
- Candidate: 毕竟，近年来亚洲发展势头强劲，以至于许多人认为该地区几乎可以承受世界其他地方施加的任何压力。
- Reference (review only): 毕竟，亚洲在最近几年表现出色，太多的人相信，不管世界其他地区情况怎样，亚洲地区都能顶住。
- Reason: 相信/认为没有实质内容差异。

### 36: wmt19_en_zh/acceptance_v1/train/025540

- AI-assisted label: Tie; delta=0.02649379; reference issue=None
- Source: The report also recognizes the need to ensure that “wages are linked to rising productivity, so that prosperity is shared with all.”
- Before: 报告还承认了确保“工资与不断上升的生产率挂钩，从而使繁荣惠及所有人”的必要性。
- Candidate: 报告也认识到，需确保“工资与不断上升的生产率挂钩，从而使繁荣惠及所有人”。
- Reference (review only): 报告还认识到需要确保“工资与生产率的提高挂钩，从而让繁荣被所有人所共享。
- Reason: 承认必要性改认识到需，句式变化，核心义务仍保留。

### 37: wmt19_en_zh/acceptance_v1/train/031098

- AI-assisted label: Tie; delta=-0.01466421; reference issue=None
- Source: Already, the market capitalization of cryptocurrencies amounts to nearly one tenth the value of the physical stock of official gold, with the capability to handle significantly larger payment operations, owing to low transaction costs.
- Before: 目前，加密货币的市值已接近官方黄金实物存量的十分之一，由于交易成本低廉，其具备处理更大规模支付交易的能力。
- Candidate: 目前，加密货币的市值已接近官方黄金实物存量的十分之一，凭借低廉的交易成本，其具备处理更大规模支付交易的能力。
- Reference (review only): 目前，私密货币的市值已经接近官方实物黄金存量价值的十分之一，拜其交易成本低的特点所赐，私密货币能够处理规模大得多的支付操作。
- Reason: 由于/凭借，原因关系和信息未变。

### 38: wmt19_en_zh/acceptance_v1/train/008776

- AI-assisted label: Tie; delta=-0.01619664; reference issue=None
- Source: Indeed, white-collar knowledge workers – that is, most workers in advanced economies – are now tethered to their workplaces essentially 24 hours a day, seven days a week, a reality that is not reflected in the official statistics.
- Before: 事实上，白领知识工作者——即大多数发达国家的劳动者——如今基本上全天候、每周七天都与工作场所紧密相连，而这一现实并未在官方统计数据中得到反映。
- Candidate: 事实上，白领知识工作者——也就是大多数发达国家的劳动者——如今基本上全天候、每周七天都与工作场所紧密相连，而这一现实并未在官方统计数据中得到反映。
- Reference (review only): 事实上，白领知识工人 — — 即发达经济体的大部分工人 — — 如今需要每天24小时、每周七天随时待命，这完全没有体现在官方数字中。
- Reason: 即/也就是是等价解释连接词。

### 39: wmt19_en_zh/acceptance_v1/train/009053

- AI-assisted label: Tie; delta=-0.02033844; reference issue=None
- Source: According to exit polls, Trump received 53% of the votes of those 45 and older, 42% of those 30-44, and just 37% of voters 18-29.
- Before: 根据出口民调显示，特朗普获得了45岁及以上选民的53%选票，30至44岁选民的42%，以及18至29岁选民的37%。
- Candidate: 根据出口民调显示，特朗普获得了45岁及以上选民53%的选票，30至44岁选民42%的选票，以及18至29岁选民37%的选票。
- Reference (review only): 据出口民调，特朗普赢得53%的45岁及以上选民，42%的30—44岁选民，而18—29岁选民他只赢得了37 % 。
- Reason: 补齐各百分比的选票宾语，信息在原稿已明确，属于小幅格式润色。

### 40: wmt19_en_zh/acceptance_v1/train/039878

- AI-assisted label: Tie; delta=0.04420941; reference issue=None
- Source: If Europe wants things to remain as they are, things will have to change dramatically.
- Before: 如果欧洲希望一切维持现状，就必须发生巨大变化。
- Candidate: 如果欧洲希望保持现状，就必须发生巨大变化。
- Reference (review only): 如果欧洲希望保住现有的地位，就必须进行大规模改革。
- Reason: 一切维持现状/保持现状等价。

### 41: wmt19_en_zh/acceptance_v1/train/004149

- AI-assisted label: Tie; delta=0.03360608; reference issue=None
- Source: One key factor limiting firms’ growth is the fragmented nature of the African market, which currently comprises mostly small economies with only limited economic and political linkages.
- Before: 限制企业发展的关键因素之一是非洲市场的碎片化特征，目前该市场主要由众多小型经济体组成，这些经济体之间经济和政治联系有限。
- Candidate: 限制企业发展的关键因素之一是非洲市场的碎片化特征，目前该市场主要由众多小型经济体构成，这些经济体之间经济和政治联系有限。
- Reference (review only): 限制企业增长的一个关键因素是非洲市场四分五裂，现有的非洲市场主要由经济和政治联系有限的小型经济体构成。
- Reason: 组成/构成等价，未见质量变化。

### 42: wmt19_en_zh/acceptance_v1/train/021418

- AI-assisted label: Worse; delta=0.04000357; reference issue=None
- Source: For starters, tensions between several Southeast Asian countries and China are on the rise, owing partly to the fact that China, under President Xi Jinping (its most autocratic leader since Deng Xiaoping), has been acting increasingly assertively in staking its contested territorial claims in the region’s waters.
- Before: 首先，由于中国在习近平总统（自邓小平以来最专制的领导人）领导下，日益积极地主张其在该地区水域的争议性领土主张，东南亚一些国家与中国之间的紧张关系正在上升。
- Candidate: 首先，由于中国在习近平主席（自邓小平以来最具权威性的领导人）领导下，日益积极地主张其在该地区海域的争议性领土主张，东南亚一些国家与中国之间的紧张关系正在上升。
- Reference (review only): 首先，一些东盟国家与中国的紧张关系正在升级，部分原因在于在习近平主席（邓小平以来最独裁的领导人）的领导下，中国在东南亚海域的争议性领土主张越来越咄咄逼人。
- Reason: 虽然修正了主席称谓，但把autocratic改最具权威性改变了关键评价含义。

### 43: wmt19_en_zh/acceptance_v1/train/011842

- AI-assisted label: Tie; delta=0.03564761; reference issue=None
- Source: Moreover, though the US unemployment rate has fallen, this largely reflects an alarming decline in labor-force participation, with more than 6.5 million Americans since 2006 having given up looking for work.
- Before: 此外，尽管美国的失业率已下降，但这一下降在很大程度上反映了劳动力参与率的令人担忧的下降，自2006年以来，已有超过650万人放弃寻找工作。
- Candidate: 此外，尽管美国失业率已下降，但这一下降在很大程度上反映了劳动力参与率的令人担忧的下降，自2006年以来，已有超过650万人放弃寻找工作。
- Reference (review only): 此外，尽管美国失业率有所下降，但这大部分反映了值得警惕的劳动力参与率的下降，自2006年以来超过650万美国人放弃寻找工作。
- Reason: 只删除美国的失业率中的的。

### 44: wmt19_en_zh/acceptance_v1/train/039874

- AI-assisted label: Tie; delta=0.03929225; reference issue=None
- Source: Or are we prepared to undergo a painful decline brought on by nationalist infighting and complacency?
- Before: 我们是否准备承受由民族主义内斗和自满情绪引发的痛苦衰退？
- Candidate: 我们是否准备承受因民族主义内斗和自满情绪而带来的痛苦衰退？
- Reference (review only): 我们是否打算经历一场由民族主义内讧和自满所带来的痛苦的经济衰退？
- Reason: 由……引发/因……而带来等价。

### 45: wmt19_en_zh/acceptance_v1/train/025457

- AI-assisted label: Tie; delta=0.04060622; reference issue=None
- Source: Unlike older industrial robots, newer technologies can interact safely and efficiently with humans, who sometimes need to train them and will increasingly have to work seamlessly with algorithms and machines. For example, a doctor’s practice will be greatly enhanced by diagnostic algorithms.
- Before: 与早期的工业机器人不同，新技术能够安全而高效地与人类互动，有时人类还需要对其进行培训，并且将越来越多地与算法和机器无缝协作。例如，诊断算法将大大增强医生的诊疗实践。
- Candidate: 与早期的工业机器人不同，新技术能够安全而高效地与人类互动，人类有时还需要对其进行培训，并将越来越多地与算法和机器无缝协作。例如，诊断算法将大大增强医生的诊疗实践。
- Reference (review only): 与旧的工业机器人不同，新的技术可以与人类安全有效地进行交互，而人类有时需要对机器进行培训，并将越来越多地与算法和机器密切协作，比如诊断算法就将大大提升医生的行医水平。
- Reason: 换词序并删重复主语，保留相同核心信息。

### 46: wmt19_en_zh/acceptance_v1/train/037053

- AI-assisted label: Tie; delta=-0.03445167; reference issue=None
- Source: It is estimated that some 1.1 million two-parent families will lose an average of £2,770 per year, while working single parents will lose an average of £1,350 per year.
- Before: 据估计，约有110万个双亲家庭每年将平均损失2770英镑，而单亲在职父母每年将平均损失1350英镑。
- Candidate: 据估计，约有110万个双亲家庭每年将平均减少收入2770英镑，而单亲在职父母每年将平均减少收入1350英镑。
- Reference (review only): 据估计约有110万双亲家庭会因此年均少收入2770英镑，而有工作的单亲家庭年均损失1350英镑。
- Reason: 损失/减少收入在这一句中表达相近结果，参考也混用二者。

### 47: wmt19_en_zh/acceptance_v1/train/027866

- AI-assisted label: Uncertain; delta=-0.03462170; reference issue=None
- Source: And if the liquidation rate for creditors’ common claims is defined as no lower than that at the time the draft restructuring plan was submitted for approval, compensation must be considered in the event that payment is delayed.
- Before: 如果清偿率被定义为不低于提交重整计划草案时的水平，那么在付款延迟的情况下，必须考虑补偿。
- Candidate: 如果清偿率被定义为不低于提交重整计划草案时的水平，则在付款延迟的情况下，应考虑补偿。
- Reference (review only): 而如果债权人一般要求的清盘赔偿率被界定为不低于重组计划初稿提交审批时的赔偿率的话，那么在支付延迟的情况下，就要考虑给予赔偿金了。
- Reason: 必须改应可能弱化must，但汉语规范性应也可表示义务，不能稳定判定。

### 48: wmt19_en_zh/acceptance_v1/train/003260

- AI-assisted label: Better; delta=0.04852089; reference issue=None
- Source: US President John F. Kennedy once warned that “every man, woman, and child lives under a nuclear sword of Damocles, hanging by the slenderest of threads, capable of being cut at any moment.”
- Before: 美国总统约翰·F·肯尼迪曾警告说：“每个人、每个妇女和儿童都生活在达摩克利斯之剑的核阴影之下，悬于一根最纤细的丝线上，随时可能被切断。”
- Candidate: 美国总统约翰·F·肯尼迪曾警告说：“每个人、每个妇女和儿童都生活在核时代的达摩克利斯之剑之下，悬于一根最纤细的丝线上，随时可能被切断。”
- Reference (review only): 美国总统肯尼迪曾警告“每个人，不管男人、女人还是儿童，都生活在核达摩克利斯之剑之下，千钧之重系于一发，随时随地都有可能发断剑落。
- Reason: 将核阴影表述改为核时代的剑之下，更接近原句核剑隐喻；其余悬挂指代仍不佳。

### 49: wmt19_en_zh/acceptance_v1/train/032166

- AI-assisted label: Better; delta=-0.04600665; reference issue=None
- Source: From the outside, a boardroom is all too often viewed as a kind of bubble where big decisions that influence thousands of lives are made by faceless people.
- Before: 从外部来看，董事会往往被视为一个泡泡，里面那些对数千人生活产生影响的重大决策，是由一些无名之辈做出的。
- Candidate: 从外部来看，董事会往往被视为一个封闭空间，那些对数千人生活产生重大影响的决策，是由一些无名之人做出的。
- Reference (review only): 从外部看来，董事会常常像是一个透明罩，在这个罩子里，平庸刻板之士做出影响成千上万人生活的重大决定。
- Reason: 封闭空间比泡泡更自然地表达boardroom bubble，无名之人也减少无名之辈的贬义。

### 50: wmt19_en_zh/acceptance_v1/train/024398

- AI-assisted label: Better; delta=0.04625005; reference issue=None
- Source: SDRs are highly complicated and difficult to grasp, but they boil down to the international creation of money.
- Before: 特别提款权具有高度复杂性和难以把握的特点，但其本质是国际创造货币。
- Candidate: 特别提款权具有高度复杂性和难以把握的特点，但其本质是国际货币创造。
- Reference (review only): 特别提款权高度复杂并难以掌握，但他们都属于国际货币扩张行为。
- Reason: 国际货币创造较国际创造货币的名词结构更清楚，主要是句法与搭配改善。

### 51: wmt19_en_zh/acceptance_v1/train/037990

- AI-assisted label: Uncertain; delta=0.04088948; reference issue=None
- Source: With the military, which had held exclusive power from 1962, retaining some 25% of the seats, there were fears that the election would be a façade.
- Before: 自1962年以来一直掌握独裁权力的军队仍保留了大约25%的席位，人们担心这次选举只是一场表面现象。
- Candidate: 自1962年以来一直掌握权力的军队仍保留约25%的席位，人们担心此次选举只是表面文章。
- Reference (review only): 军方（自1962年以来一直大权独揽）保留了25%的席位，因此人们担心此次选举只是一场闹剧。
- Reason: 表面文章修复façade表达，但删除exclusive power的独占信息，收益与损失并存。

### 52: wmt19_en_zh/acceptance_v1/train/028312

- AI-assisted label: Tie; delta=0.04630840; reference issue=None
- Source: Might this mean that the Fed should slow the pace of its tapering of QE?
- Before: 这是否意味着美联储应放缓量化宽松政策缩减的步伐？
- Candidate: 这是否意味着美联储应放缓缩减量化宽松政策的节奏？
- Reference (review only): 这有可能意味着美联储应该放慢退出QE的节奏吗？
- Reason: 缩减的步伐/缩减的节奏，语序变化不构成明显质量差别。

### 53: wmt19_en_zh/acceptance_v1/train/035418

- AI-assisted label: Tie; delta=-0.03735198; reference issue=None
- Source: All were portrayed as representatives of another world, with no relationship to the lives of ordinary British people.
- Before: 所有人物都被描绘为另一个世界中的代表，与普通英国人的生活毫无关联。
- Candidate: 所有人皆被描绘为另一个世界中的代表，与普通英国人的生活毫无关联。
- Reference (review only): 所有人都被描绘成另一个世界的代表，那个世界丝毫无关于普通英国民众的生活。
- Reason: 所有人物/所有人皆，缺乏上下文无法确认人物一词是否错误，实质内容相同。

### 54: wmt19_en_zh/acceptance_v1/train/019668

- AI-assisted label: Better; delta=-0.03091400; reference issue=None
- Source: That means that the system’s focus must shift from speculative and proprietary trading to lending and job creation, which implies reforms of financial-sector regulation, and of anti-trust and corporate-governance laws, together with adequate enforcement to ensure that markets do not become rigged casinos.
- Before: 这意味着该系统的重点必须从投机性和专营交易转向放贷和就业创造，这暗示着需要对金融监管、反垄断法和公司治理法律进行改革，并辅以充分的执行力度，以确保市场不会变成被操控的赌场。
- Candidate: 这意味着该系统的重点必须从投机性交易和自营交易转向放贷和就业创造，这预示着需要对金融监管、反垄断法和公司治理法律进行改革，并辅以充分的执行力度，以确保市场不会沦为被操控的赌场。
- Reference (review only): 这意味着金融系统的中心应该从投机和自营交易转向借贷和就业创造，而这又要求金融部门监管、反垄断和公司治理法律等方面的改革和充分强化，以保证市场不会变成被操纵的赌场。
- Reason: 专营交易改自营交易，修复proprietary trading；其余改写并未逆转这项修正。

### 55: wmt19_en_zh/acceptance_v1/train/032002

- AI-assisted label: Uncertain; delta=-0.03310076; reference issue=None
- Source: As should be clear, the claim of “decentralization” is a myth propagated by the pseudo-billionaires who control this pseudo-industry.
- Before: 正如应该清楚的，所谓“去中心化”的说法是那些控制这一伪产业的伪亿万富翁们散布的神话。
- Candidate: 正如所言，所谓“去中心化”的说法，是那些控制这一伪产业的伪亿万富翁们散布的神话。
- Reference (review only): 应该清楚，所谓的“去中心化”根本就是控制着这个伪行业的伪亿万富翁所宣传的神话。
- Reason: 正如所言弱化should be clear，但旧句也生硬；净效果存在取舍。

### 56: wmt19_en_zh/acceptance_v1/train/022481

- AI-assisted label: Tie; delta=-0.03475734; reference issue=None
- Source: Otherwise, according to Zhou Xiaochuan, the governor of the People’s Bank of China, the country might face a “Minsky moment,” in which excessive optimism and debt-financed investment culminate in financial crisis.
- Before: 否则，根据中国人民银行行长周小川的说法，该国可能会面临“明斯基时刻”，即过度乐观和以债务融资的投资最终导致金融危机。
- Candidate: 否则，根据中国人民银行行长周小川的说法，该国可能面临“明斯基时刻”，即过度乐观与债务驱动的投资最终引发金融危机。
- Reference (review only): 否则，中国人民银行行长周小川说，中国将面临“明斯基时刻 ” ， 即在过度乐观和债务融资投资的高潮中迎来金融危机。
- Reason: 债务融资/债务驱动与导致/引发总体近似，缺乏足够依据判定实质优劣。

### 57: wmt19_en_zh/acceptance_v1/train/006519

- AI-assisted label: Tie; delta=-0.04418955; reference issue=None
- Source: Slow growth has silenced the monetarists on the ECB’s Governing Council. This will change, however, once the economic pick-up is confirmed.
- Before: 缓慢的增长已使欧洲央行管理委员会中的货币主义者保持沉默。然而，一旦经济复苏得到确认，这一局面将有所改变。
- Candidate: 缓慢增长已使欧洲央行管理委员会中的货币主义者保持沉默。然而，一旦经济复苏得到确认，这一局面将有所改变。
- Reference (review only): 缓慢的增长速度堵住了欧洲央行监管理事会货币专家们的幽幽众口，但只要经济复苏得到确认，所有这些情况都会有所改变。
- Reason: 删除缓慢的增长中的的。

### 58: wmt19_en_zh/acceptance_v1/train/011298

- AI-assisted label: Tie; delta=-0.03568645; reference issue=None
- Source: If, say, the Bank of China and the Federal Reserve decided to teach speculators a lesson by pushing the dollar’s value relative to the yuan up by 20% for a month, they could do so, bankrupting many financial institutions with short positions.
- Before: 例如，如果中国银行和美联储决定通过在一个月内将美元相对于人民币的价值推高20%来给投机者一个教训，他们可以这么做，从而让许多持有空头头寸的金融机构破产。
- Candidate: 例如，如果中国银行和美联储决定通过在一个月内将美元相对人民币升值20%来给投机者一个教训，他们可以这么做，从而让许多持有空头头寸的金融机构破产。
- Reference (review only): 比如说，如果中国银行和美联储决定教训教训投机者，将美元与人民币的比价推高20 % ， 并保持一个月，它们这样做是完全可能的，能让大量做空仓的金融机构破产。
- Reason: 推高美元价值/美元升值表达相近；for a month的持续期误读两者均未修正。

### 59: wmt19_en_zh/acceptance_v1/train/031568

- AI-assisted label: Better; delta=0.03667222; reference issue=None
- Source: I am often taken aback by well-meaning and economically comfortable environmentalists who cavalierly suggest that gasoline prices should be doubled or electricity exclusively sourced from high-cost green sources.
- Before: 我常常对那些本意良好且经济状况舒适的环保主义者感到震惊，他们轻率地建议应将汽油价格翻倍，或仅使用高成本的绿色能源来供电。
- Candidate: 我常常对那些本意良好且经济状况优越的环保主义者感到震惊，他们轻率地建议应将汽油价格翻倍，或仅使用高成本的绿色能源来供电。
- Reference (review only): 经济条件优越的环保主义者虽然用心是好的，但他们提出汽油价格应该翻番或者只允许使用高成本绿色能源的所谓骑士精神却常常令我目瞪口呆。
- Reason: 经济状况优越比经济状况舒适更符合汉语搭配，仍表达经济宽裕。

### 60: wmt19_en_zh/acceptance_v1/train/030298

- AI-assisted label: Tie; delta=-0.16616674; reference issue=None
- Source: (It was Yahoo, not Google, that gave personal information to China’s government, which then jailed a blogger.)
- Before: （是雅虎，而不是谷歌，向中国政府提供了个人资料，随后该博主被拘捕。）
- Candidate: 是雅虎，而非谷歌，向中国政府提供了个人资料，随后该博主被拘捕。
- Reference (review only): （是雅虎，而不是谷歌，曾把个人信息提供给中国政府，而致使一名博客作者入狱。 ）
- Reason: 删括号、不是改非，命题内容一致。

### 61: wmt19_en_zh/acceptance_v1/train/030040

- AI-assisted label: Tie; delta=-0.11357271; reference issue=None
- Source: Governments and major pharmaceutical companies must be sure that poorer nations receive the medical supplies they need.
- Before: 政府和主要制药公司必须确保较贫穷国家获得所需的医疗物资。
- Candidate: 政府和主要制药公司必须确保较贫穷国家获得所需医疗物资。
- Reference (review only): 政府和大型制药企业必须确保贫困国家也能得到所需的医疗物资。
- Reason: 仅删除所需的中的的。

### 62: wmt19_en_zh/acceptance_v1/train/026724

- AI-assisted label: Tie; delta=-0.06720214; reference issue=None
- Source: The EU had indicated that performance on human rights would be a key factor in determining whether Turkey would be admitted as a member.
- Before: 欧盟曾表示，人权表现将是决定土耳其是否能加入欧盟的关键因素之一。
- Candidate: 欧盟曾表示，人权状况将是决定土耳其是否能加入欧盟的关键因素之一。
- Reference (review only): 欧盟曾表态说人权问题上的表现将是是否承认土耳其为成员国的关键决定因素。
- Reason: 人权表现/人权状况在当前句中无可靠优劣。

### 63: wmt19_en_zh/acceptance_v1/train/039669

- AI-assisted label: Tie; delta=0.06272708; reference issue=None
- Source: Emerging-market economies have not yet shown the kind of global leadership that suggests an affirmative answer to this question.
- Before: 新兴市场国家尚未展现出表明该问题答案为肯定的全球领导力。
- Candidate: 新兴市场国家尚未展现出足以表明该问题答案为肯定的全球领导力。
- Reference (review only): 而事实上，这些新兴经济体在这方面还未能表现出足以令人信服的领导力。
- Reason: 补足以，未改变全球领导力与肯定答案的关系。

### 64: wmt19_en_zh/acceptance_v1/train/013992

- AI-assisted label: Tie; delta=0.07570810; reference issue=None
- Source: Work has already begun on construction of a land-based interceptor and radar site in Romania.
- Before: 罗马尼亚已经开始建设陆基拦截器和雷达站点。
- Candidate: 罗马尼亚已开始建设陆基拦截系统及雷达站点。
- Reference (review only): 罗马尼亚已动工建设陆基拦截系统和雷达阵列。
- Reason: 拦截器/拦截系统是表述范围选择，仅当前句不足以确认质量提升。

### 65: wmt19_en_zh/acceptance_v1/train/021825

- AI-assisted label: Tie; delta=-0.05734179; reference issue=None
- Source: For example, Indian and Chinese members held different positions about the role of China’s “Belt and Road” infrastructure projects.
- Before: 例如，印度和中国成员对中国“一带一路”基础设施项目的角色持不同立场。
- Candidate: 例如，印度与中国成员对“一带一路”基础设施项目的作用持不同立场。
- Reference (review only): 比如，印度和中国成员对于中国“一带一路”基础设施项目的作用持有不同观点。
- Reason: 印度和中国/印度与中国、角色/作用近义，中国一带一路的归属仍清楚。

### 66: wmt19_en_zh/acceptance_v1/train/011065

- AI-assisted label: Tie; delta=0.20904782; reference issue=ambiguous_translation
- Source: Obama’s Underachieving Foreign Policy
- Before: 奥巴马表现平平的外交政策
- Candidate: 奥巴马外交政策的成效欠佳
- Reference (review only): 奥巴马外交政策差强人意
- Reason: 表现平平/成效欠佳均可传达未达预期；参考差强人意本身存在语义歧义，需谨慎。

### 67: wmt19_en_zh/acceptance_v1/train/026131

- AI-assisted label: Tie; delta=-0.13701609; reference issue=None
- Source: Were the US Senate to fail to ratify New START, the treaty’s proponents argue that the US would lose predictability about Russia’s nuclear activities, resulting in greater distrust and risk of miscalculation, making both sides less secure.
- Before: 如果美国参议院未能批准新START条约，该条约的支持者认为，美国将难以预测俄罗斯的核活动，从而加剧不信任，增加误判风险，使双方安全处境更加严峻。
- Candidate: 如果美国参议院未能批准新START条约，该条约的支持者认为，美国将失去对俄罗斯核活动的可预测性，进而导致不信任加剧、误判风险上升，使双方安全处境更加严峻。
- Reference (review only): 新条约的支持者们宣称，如果参议院未能通过的话，美国将难以预测俄罗斯未来的核活动，并以此产生更多的不信任和误判，最终令双方都更不安全。
- Reason: 难以预测/失去可预测性为改写，没有明确事实修复。

### 68: wmt19_en_zh/acceptance_v1/train/011455

- AI-assisted label: Tie; delta=-0.06351172; reference issue=None
- Source: More important, even though the severe sanctions regime led by the United States is bound to be imperfect – it only hardens further Iran’s resistance to “America’s designs.”
- Before: 更重要的是，尽管以美国为首的严厉制裁制度注定不完美——它只会进一步强化伊朗对“美国图谋”的抵抗。
- Candidate: 更重要的是，尽管以美国主导的严厉制裁体系注定存在缺陷——它只会进一步强化伊朗对“美国图谋”的抵触。
- Reference (review only): 更重要的是，即使美国领导的严厉制裁制度​​注定并不完美 — — 它只能进一步增强伊朗对“美国阴谋”的抵触情绪。
- Reason: 主要是主导/为首、缺陷/不完美等近义替换。

### 69: wmt19_en_zh/acceptance_v1/train/030707

- AI-assisted label: Tie; delta=-0.68036147; reference issue=None
- Source: This has always been stunning to me.
- Before: 这一直让我感到惊叹。
- Candidate: 我一直为之惊叹。
- Reference (review only): 这一直让我感到震惊。
- Reason: 这一直让我感到惊叹/我一直为之惊叹命题等价，巨大评分差来自词面。

### 70: wmt19_en_zh/acceptance_v1/train/024634

- AI-assisted label: Tie; delta=-0.08867489; reference issue=None
- Source: Russia’s leaders view Sino-American competition as a welcome addition to their country’s strategic weight, which, unlike China’s, is not being augmented by robust economic growth.
- Before: 俄罗斯领导人将中美竞争视为对本国战略影响力的有益补充，而与中国的不同之处在于，俄罗斯的战略影响力并非由强劲的经济增长所增强。
- Candidate: 俄罗斯领导人视中美竞争为增强本国战略影响力的有益补充，而与中国的不同之处在于，俄罗斯的战略影响力并非依靠强劲的经济增长来提升。
- Reference (review only): 俄罗斯领导人将中美竞争视为增加俄罗斯战略分量的良机，与中国不同，俄罗斯的地位并没有因经济增长而增加。
- Reason: 视为有益补充/增强影响力的补充，核心关系相同。

### 71: wmt19_en_zh/acceptance_v1/train/007701

- AI-assisted label: Tie; delta=0.13654826; reference issue=None
- Source: Many began to wonder whether Hollande was aware of the scope of the crisis that the recent downturn might trigger.
- Before: 许多人开始怀疑，奥朗德是否意识到近期经济下滑可能引发的危机的规模。
- Candidate: 许多人开始怀疑，奥朗德是否意识到近期经济衰退可能引发的危机规模。
- Reference (review only): 许多人开始怀疑奥朗德是否对不久前经济衰退可能引发的危机规模有着清醒的认识。
- Reason: 下滑/衰退在此语境可有程度差异，现有信息不足以据此判为明确改善。

### 72: wmt19_en_zh/acceptance_v1/train/001918

- AI-assisted label: Worse; delta=-0.16298477; reference issue=None
- Source: But its rising value also makes it an attractive asset class for investors, because further price increases are expected.
- Before: 但其价值的上升也使它成为投资者青睐的资产类别，因为预计价格还会进一步上涨。
- Candidate: 但其升值也使它成为投资者青睐的资产类别，因进一步价格上涨预期。
- Reference (review only): 但土地价值的上升让它们成为令投资者垂涎不已的资产类别，因为可以预期未来价格还会上涨。
- Reason: 因进一步价格上涨预期缺乏自然的谓语结构，比因为预计价格还会上涨更生硬。

### 73: wmt19_en_zh/acceptance_v1/train/028717

- AI-assisted label: Tie; delta=0.09707284; reference issue=None
- Source: And even if a UBI succeeds in Kenya over the next 12 years, it is not a solution to pressing problems in the US economy today.
- Before: 即使在接下来的12年内，肯尼亚的全民基本收入（UBI）取得成功，它也不是解决美国当前经济紧迫问题的方案。
- Candidate: 即使肯尼亚在未来12年内成功实施全民基本收入（UBI），它也并非解决美国当前经济紧迫问题的方案。
- Reference (review only): 即使全民基本收入项目能在未来12年中在肯尼亚取得成功，也并非解决当前美国经济中紧迫问题的办法。
- Reason: 调整肯尼亚和成功的语序、不是/并非，核心限定均保留。

### 74: wmt19_en_zh/acceptance_v1/train/033535

- AI-assisted label: Tie; delta=-0.10417492; reference issue=None
- Source: Today, counter-cyclical policies are not an option; there simply isn’t enough fiscal or monetary space.
- Before: 如今，逆周期政策已不再是可选方案；财政或货币政策空间已所剩无几。
- Candidate: 如今，逆周期政策已不再是可选方案；财政和货币政策空间已所剩无几。
- Reference (review only): 如今，反周期政策不再是选项之一；根本没有足够的财政或货币空间。
- Reason: 否定语境下财政或货币/财政和货币总体仍指两类政策空间不足，未见显著变化。

### 75: wmt19_en_zh/acceptance_v1/train/011142

- AI-assisted label: Tie; delta=0.07038038; reference issue=None
- Source: The United States seems to reel from one mass gun killing to another – roughly one a month this year alone.
- Before: 美国似乎正接连陷入一起又一起大规模枪击事件——仅今年就已发生每月一起此类事件。
- Candidate: 美国似乎接连发生一起又一起大规模枪击事件——仅今年就已发生每月一起。
- Reference (review only): 美国似乎陷入了大规模枪击杀人的怪圈，光是今年，几乎每个月都要发生一起。
- Reason: 删除部分重复词，但每月一起的表达缺陷仍在，未达到明确净改善。

### 76: wmt19_en_zh/acceptance_v1/train/039903

- AI-assisted label: Tie; delta=-0.10344928; reference issue=None
- Source: The EU’s institutions must exercise their creativity to the fullest – conventional thinking will not be enough to save the Union.
- Before: 欧盟的机构必须充分发挥其创造力——传统的思维方式不足以拯救联盟。
- Candidate: 欧盟机构必须充分施展创造力——传统思维不足以挽救联盟。
- Reference (review only): 欧盟机构必须充分发挥创造力 — — 传统的思维方式救不了欧盟。
- Reason: 充分发挥/充分施展、拯救/挽救均为近义改写。

### 77: wmt19_en_zh/acceptance_v1/train/023825

- AI-assisted label: Tie; delta=-0.06219582; reference issue=None
- Source: They did not realize that the money they had made would melt away like snow in a warm spring.
- Before: 他们没有意识到，他们赚到的钱会像春日暖阳下的雪一样消融殆尽。
- Candidate: 他们没有意识到，他们赚到的钱会像春日融雪般消融殆尽。
- Reference (review only): 他们没有意识到，他们所赚到的钱会如同阳光下的冰雪一样融化殆尽。
- Reason: 暖阳下的雪/春日融雪保留同一比喻，细节删减不足以稳定判向。

### 78: wmt19_en_zh/acceptance_v1/train/011455

- AI-assisted label: Tie; delta=-0.08233385; reference issue=None
- Source: More important, even though the severe sanctions regime led by the United States is bound to be imperfect – it only hardens further Iran’s resistance to “America’s designs.”
- Before: 更重要的是，尽管以美国为首的严厉制裁制度注定不完美——它只会进一步强化伊朗对“美国图谋”的抵抗。
- Candidate: 更重要的是，尽管由美国主导的严厉制裁体系注定存在缺陷——它只会进一步强化伊朗对“美国图谋”的抵制。
- Reference (review only): 更重要的是，即使美国领导的严厉制裁制度​​注定并不完美 — — 它只能进一步增强伊朗对“美国阴谋”的抵触情绪。
- Reason: 主导/为首、抵制/抵抗等近义改写，原句信息仍在。

### 79: wmt19_en_zh/acceptance_v1/train/032558

- AI-assisted label: Tie; delta=-0.06415112; reference issue=None
- Source: In the US, economic performance improved only marginally in 2012, with annual GDP rising by 2.3%, up from 1.8% in 2011.
- Before: 在美国，2012年的经济表现仅略有改善，全年GDP增长了2.3%，高于2011年的1.8%。
- Candidate: 在美国，2012年经济表现仅小幅改善，全年GDP增长2.3%，高于2011年的1.8%。
- Reference (review only): 在美国，2012年的经济表现只是略有好转，GDP增长率从2011年的1.8%增至2.3 % 。
- Reason: 略有/小幅、删的了，事实及数字全部保留。
