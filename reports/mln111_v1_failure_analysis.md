# MLN111 B0 Residual Failure Analysis

## Scope

This report manually audits the five development questions that fail child-level Full Evidence@5
under baseline B0. B0 is structured-child BM25 + Qwen3 dense retrieval + RRF + Qwen3 reranking.
The benchmark is not modified. Hidden-test data is not accessed.

The audit distinguishes two failure levels:

- **child-level miss:** a required gold child ID is absent from the top five;
- **production-context miss:** parent expansion also fails to supply the source passage containing
  the required aspect.

This distinction is essential because production generation receives parent passages, not isolated
child IDs.

## Summary

| Question | Child-level result | Parent-aware result | Action |
|---|---|---|---|
| `mln111_0002` | No gold child in top 5 | Genuine context miss | Targeted retry candidate |
| `mln111_0021` | Natural-origin child at rank 7 | Covered by parent of rank-1 child | No retry |
| `mln111_0057` | Supplemental child missing | Covered by parent of rank-1 child | No retry |
| `mln111_0082` | List-of-principles child missing | Covered by parent of rank-1 child | No retry |
| `mln111_0095` | Engels child missing | Covered by parent of rank-2 child | No retry |

Therefore, five child-level failures reduce to **one genuine production-context failure** after
parent expansion.

## `mln111_0002`

**Question:** Ý thức là gì theo quan điểm duy vật biện chứng?

**Required aspects**

- `definition_reflection`: consciousness is the reflection of the real world by the human brain;
- `highest_form`: consciousness is the highest form of reflection of the material world.

**Gold evidence group**

- `g1`, `child_03e5028e78e625dddaa5`, PDF pages 87–88.

**Retrieved evidence**

- rank 1 `child_71def1e7021001d9c769`: the brain is the material organ of consciousness and
  consciousness is its function;
- rank 2 `child_c276e01ad2f1d3130911`: section introduction;
- remaining top-five passages discuss the matter–consciousness relation, epistemic reflection, and
  chapter objectives.

**Covered:** material basis and brain function.

**Missing:** the two explicit definitional propositions in the gold child.

**Why missing:** semantic-neighbour retrieval favors explanatory context about the brain and
consciousness, but not the adjacent passage containing the exact definition.

**Expected recovery query:** `Ý thức là hình thức phản ánh nào của thế giới vật chất và phản ánh
hiện thực bằng cơ quan nào?`

**Failure type:** `true_context_gap / semantic_neighbour`.

## `mln111_0021`

**Question:** Phân tích đồng thời nguồn gốc tự nhiên và xã hội của ý thức.

**Required aspects**

- `natural_origin`: human brain plus objective-world effects;
- `social_origin`: labour, language, and social practice.

**Child-level result:** social origin is rank 1; strict natural-origin gold child is rank 7.

**Parent-aware result:** both gold children share `parent_99b193033aafa93122a1`. Expanding the
rank-1 social-origin child supplies both required aspects.

**Missing after parent expansion:** none.

**Expected recovery query:** none.

**Failure type:** `child_id_gap / resolved_by_parent_expansion`.

## `mln111_0057`

**Question:** Sản xuất vật chất đóng vai trò là cơ sở của sự tồn tại và phát triển xã hội như thế
nào?

**Required aspects**

- `material_livelihood`: production creates material means of life;
- `social_relations`: production is the premise of history and foundation of social relations.

**Child-level result:** the main child is rank 1; supplemental gold child is absent from top 5.

**Parent-aware result:** both gold children share `parent_df449336628d52ca7b68`; rank 1 therefore
expands to the complete argument.

**Missing after parent expansion:** none.

**Expected recovery query:** none.

**Failure type:** `sibling_child_gap / resolved_by_parent_expansion`.

## `mln111_0082`

**Question:** Học thuyết hình thái kinh tế - xã hội bao gồm những quan điểm cơ bản nào và đóng vai
trò gì tại Việt Nam hiện nay?

**Required aspects**

- `theory_components`: the four core theoretical relationships/processes;
- `vietnam_role`: scientific basis for Vietnam's socialist transition path.

**Child-level result:** the contextual/Vietnam child is rank 1; the adjacent list of theoretical
components is absent from top 5.

**Parent-aware result:** both gold children share `parent_0999fee139446c2e8d92`. Rank-1 parent
expansion restores the list; ranks 2–3 additionally retrieve Vietnam-specific application evidence.

**Missing after parent expansion:** none.

**Expected recovery query:** none.

**Failure type:** `sibling_child_gap / resolved_by_parent_expansion`.

## `mln111_0095`

**Question:** Quan điểm của chủ nghĩa duy tâm về vận động là gì và Ph. Ăngghen cùng V.I. Lênin đã
bác bỏ quan điểm đó ra sao?

**Required aspects**

- `engels_argument`: material properties are disclosed through motion;
- `idealist_claim_and_lenin`: motion without matter and Lenin's criticism.

**Child-level result:** the Lenin/idealist child is rank 2; the preceding Engels child is absent
from top 5.

**Parent-aware result:** both gold children share `parent_535a46827fddb3b5c359`. Expanding rank 2
supplies the complete two-author argument.

**Missing after parent expansion:** none.

**Expected recovery query:** none.

**Failure type:** `sibling_child_gap / resolved_by_parent_expansion`.

## Design consequence

The current evidence does **not** justify activating an agent for all five child-level failures.
An Evidence Judge must operate on expanded parent context and trigger recovery only for a genuine
missing aspect. On this development set, the initial positive specification contains one true
recovery case and four mandatory non-activation cases.
