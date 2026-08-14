# Five-subject B0 delta and failure analysis

## Transitions

| Transition | Win | Loss | Mixed | Tie |
|---|---:|---:|---:|---:|
| within_subject_bm25 -> within_subject_dense | 52 | 75 | 8 | 196 |
| within_subject_dense -> within_subject_hybrid_rrf | 65 | 23 | 7 | 236 |
| within_subject_hybrid_rrf -> within_subject_hybrid_reranker | 66 | 30 | 1 | 234 |
| within_subject_hybrid_reranker -> within_subject_parent_aware_b0 | 44 | 0 | 1 | 286 |
| global_bm25 -> global_dense | 55 | 80 | 9 | 187 |
| global_dense -> global_hybrid_rrf | 66 | 26 | 9 | 230 |
| global_hybrid_rrf -> global_hybrid_reranker | 67 | 27 | 2 | 235 |
| global_hybrid_reranker -> global_parent_aware_b0 | 44 | 0 | 3 | 284 |

## Residual parent-aware top-5 failures

### within_subject_parent_aware_b0

Total: **20**

| ID | Subject | Category | Difficulty | First gold rank |
|---|---|---|---|---:|
| hcm202_0037 | HCM202 | multi_chunk | hard | 1 |
| hcm202_0039 | HCM202 | multi_hop_cross_chapter | hard | — |
| hcm202_0052 | HCM202 | synthesis | hard | — |
| hcm202_0057 | HCM202 | multi_chunk | hard | 1 |
| hcm202_0058 | HCM202 | synthesis | medium | 2 |
| hcm202_0081 | HCM202 | multi_chunk | medium | 2 |
| mln111_0031 | MLN111 | multi_hop_cross_chapter | hard | 1 |
| mln111_0049 | MLN111 | comparison_relationship | hard | 3 |
| mln122_0005 | MLN122 | comparison_relationship | hard | 1 |
| mln122_0009 | MLN122 | synthesis | hard | 1 |
| mln122_0036 | MLN122 | synthesis | hard | 1 |
| mln122_0044 | MLN122 | multi_hop_cross_chapter | hard | 1 |
| mln131_0002 | MLN131 | direct | easy | — |
| mln131_0010 | MLN131 | multi_chunk | medium | 1 |
| mln131_0023 | MLN131 | multi_hop_cross_chapter | hard | — |
| mln131_0028 | MLN131 | multi_chunk | hard | 1 |
| mln131_0035 | MLN131 | multi_chunk | hard | 1 |
| mln131_0044 | MLN131 | multi_hop_cross_chapter | hard | 1 |
| mln131_0054 | MLN131 | comparison_relationship | hard | — |
| vnr202_0005 | VNR202 | synthesis | hard | 2 |

### global_parent_aware_b0

Total: **26**

| ID | Subject | Category | Difficulty | First gold rank |
|---|---|---|---|---:|
| hcm202_0005 | HCM202 | comparison_relationship | medium | — |
| hcm202_0032 | HCM202 | multi_hop_cross_chapter | hard | 1 |
| hcm202_0037 | HCM202 | multi_chunk | hard | 1 |
| hcm202_0039 | HCM202 | multi_hop_cross_chapter | hard | 6 |
| hcm202_0057 | HCM202 | multi_chunk | hard | 1 |
| hcm202_0058 | HCM202 | synthesis | medium | 1 |
| hcm202_0069 | HCM202 | synthesis | medium | — |
| hcm202_0081 | HCM202 | multi_chunk | medium | 4 |
| mln111_0030 | MLN111 | synthesis | hard | 2 |
| mln111_0031 | MLN111 | multi_hop_cross_chapter | hard | 1 |
| mln111_0049 | MLN111 | comparison_relationship | hard | 3 |
| mln122_0009 | MLN122 | synthesis | hard | 1 |
| mln122_0036 | MLN122 | synthesis | hard | 1 |
| mln131_0002 | MLN131 | direct | easy | — |
| mln131_0004 | MLN131 | explanation | medium | 6 |
| mln131_0010 | MLN131 | multi_chunk | medium | 1 |
| mln131_0023 | MLN131 | multi_hop_cross_chapter | hard | — |
| mln131_0028 | MLN131 | multi_chunk | hard | 1 |
| mln131_0035 | MLN131 | multi_chunk | hard | 1 |
| mln131_0037 | MLN131 | multi_hop_cross_chapter | hard | 3 |
| mln131_0044 | MLN131 | multi_hop_cross_chapter | hard | 1 |
| mln131_0076 | MLN131 | multi_chunk | medium | 1 |
| vnr202_0003 | VNR202 | comparison_relationship | hard | 3 |
| vnr202_0005 | VNR202 | synthesis | hard | 4 |
| vnr202_0019 | VNR202 | synthesis | hard | 2 |
| vnr202_0078 | VNR202 | explanation | hard | — |

