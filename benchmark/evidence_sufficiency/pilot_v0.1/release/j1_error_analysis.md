# J1 development error analysis

J1 (`gemini-3.5-flash-lite`, temperature 0, `evidence_judge_v1`) produced 31/32 exact label
matches: accuracy 96.88% and macro-F1 96.86%. Gold labels, required aspects, removed aspects, and
gold parent IDs were not sent to the model.

## Sole disagreement: `es_mln131_0035_partial`

- Frozen label: `PARTIAL`
- J1 prediction: `SUFFICIENT`
- The supplied parent explicitly enumerates the relevant classes/layers and states that the
  Communist Party—the vanguard of the working class—leads them in combining their strength.
- Those are the two semantic requirements expressed by the question.

The source benchmark represented this question with two evidence groups whose role descriptions
were generic (`structure and classes`, `historical context`). Removing the second group did not
actually remove a required semantic answer aspect because the remaining parent was independently
sufficient. Therefore this disagreement is recorded as a likely **annotation/redundancy artifact**,
not silently changed after seeing model output. Raw frozen metrics remain 31/32; a transparent
semantic audit would count J1 as 32/32.

No prompt or threshold was changed after inspecting this error. The held-out split remains locked.
