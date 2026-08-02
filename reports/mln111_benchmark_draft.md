# MLN111 Benchmark Draft

## Composition

- Total questions: **30**
- Definition: **10**
- Explanation: **5**
- Paraphrase: **5**
- Multi-hop: **3**
- False-premise: **3**
- Out-of-domain: **4**
- Development split: **18**
- Test split: **12**
- Answerable: **23**
- Expected refusal/correction: **7**
- Gold source spans: **29**

All source spans were resolved against the versioned MLN111 page artifact and their text
was verified to occur on the referenced page. IDs are deterministic. The test split must
not be used for chunking, model, top-k, or threshold selection.

## Review state

All 30 items remain `draft_review`. The project owner must review the Vietnamese
question, gold answer, and cited evidence in `benchmark/mln111_review.md`. Only approved
items may be used for reported retrieval or answer-quality metrics.

**Day 07 automated preparation: PASS. Human academic review: PENDING.**
