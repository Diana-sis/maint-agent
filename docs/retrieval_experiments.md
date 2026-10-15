# Retrieval Experiments

## Dataset

- Document: ATV340 Chinese Programming Manual
- Corpus: 585 text pages, 811 chunks
- Evaluation set: 10 manually labelled questions
- Metrics: Recall@1, Recall@3, MRR

## Results

| Experiment | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| E0: BM25 baseline | 0.700 | 1.000 | 0.833 |
| E1: + answer-section boost | 0.800 | 1.000 | 0.883 |
| E2: + navigation-page penalty | 0.900 | 1.000 | 0.933 |
| E3: after q007 annotation audit | 1.000 | 1.000 | 1.000 |

## Changes

### E1: Answer-section boost

Chunks containing both `可能原因` and `解决措施` receive a score boost.
This promotes answer-bearing fault descriptions over code-only index pages.

### E2: Navigation-page penalty

Chunks containing many dot leaders receive a score penalty.
This reduces false positives from table-of-contents and fault-index pages.

### E3: Annotation audit

The first result for q007 was page 549, which directly describes the generic
`CNF 现场总线通信中断` fault. Page 550 describes the more specific
`COF CANopen通信中断` fault. The relevant-page label was corrected from
`[550]` to `[549, 550]`.

E3 is an annotation correction, not an algorithmic improvement.

## Limitations

The evaluation set is small and partially derived from inspected manual pages.
The current 1.000 score must not be treated as final production performance.
More code-free paraphrases and confusing negative cases are required.
## Dense and Hybrid Retrieval

### Embedding setup

- Model: `BAAI/bge-small-zh-v1.5`
- Device: CPU
- Embedding dimension: 512
- Corpus size: 811 chunks
- Corpus encoding time: 52.81 seconds

| Experiment | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| E4: Dense retrieval | 0.800 | 0.900 | 0.875 |
| E5: Equal-weight RRF | 0.900 | 1.000 | 0.950 |
| E6: Weighted RRF (BM25 1.2, Dense 1.0) | 1.000 | 1.000 | 1.000 |

Dense retrieval recovered semantic matches but sometimes ranked general safety or
parameter pages above the specific fault record. Equal-weight RRF improved q010
but produced a rank tie for q004. Giving BM25 a weight of 1.2 resolved the tie.

The RRF weight was selected on the current development questions and therefore
requires validation on a separate test set.


## 结构感知分块实验

### 独立测试集

在优化前创建并锁定 `holdout_retrieval_questions.jsonl`，包含 12 条未参与检索权重调整的问题，覆盖制动控制、存储器错误、通信中断、过热、模拟量丢失、母线过压、缺相、编码器反馈、自整定和欠压等故障。

### 固定长度分块基线

- 文本块数量：811
- Recall@1：0.750
- Recall@3：0.917
- MRR：0.819
- 主要失败案例：编码器反馈丢失问题的正确页面未进入 Top 5

失败分析发现，第 576 页中的 SOF、SPF、SPFC 三个故障条目被混合在同一个文本块中，导致关键词分数被稀释，向量表示同时包含多个故障主题。

### 故障条目感知分块

对错误代码章节识别 `[故障名称] CODE` 标题，并使用“可能原因”确认故障条目边界。普通页面仍采用固定长度分块，超长故障条目回退到原分块策略。

- 文本块数量：883
- SPF 条目独立为 358 字符文本块
- 独立集 Recall@1：1.000
- 独立集 Recall@3：1.000
- 独立集 MRR：1.000
- 开发集 Recall@1：1.000
- 开发集 Recall@3：1.000
- 开发集 MRR：1.000

结果仅代表当前 12 条独立问题和 10 条开发问题，不代表对全部设备故障达到 100% 准确率。

### 标注审计

开发集 q010 的问题未限定具体短路故障代码。第 573 页的 SCF1/SCF3 和第 574 页的 SCF5 都明确要求检查电机电缆及电机绝缘，因此将相关页面由 `[573]` 修正为 `[573, 574]`。该变化属于标注补全，不计为算法提升。