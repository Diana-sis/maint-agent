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