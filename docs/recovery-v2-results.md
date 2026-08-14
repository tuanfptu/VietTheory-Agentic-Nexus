# Evidence-Guided Recovery V2

Recovery V2 was designed from the 20 residual Full Evidence@5 failures left by the frozen
five-subject B0 retrieval pipeline. Hidden Natural QA data was not accessed.

## Diagnosis

- 3/20 failures contained the missing gold parent around ranks 6–7.
- 17/20 did not contain all required evidence in the stored candidate ranking.
- A gold-assisted diagnostic ceiling recovered 7/20 by querying explicit missing concepts. This
  was used only to establish retrieval headroom and is not reported as agentic performance.

## V2.0: unrestricted joint reranking

Gemini judged the top-five B0 contexts for all 331 answerable public development questions and
activated on 33 (9.97%). Targeted queries were bounded to two. Recovered parents and B0 parents
were jointly reranked against the original question.

| System | Full Evidence@5 | Wins | Losses | Decision |
|---|---:|---:|---:|---|
| Frozen B0 | 93.96% | — | — | baseline |
| Recovery V2.0 | 92.15% | 2 | 8 | rejected |

The unrestricted reranker damaged already sufficient evidence when the Judge activated falsely.

## V2.1: conservative support-gated insertion

V2.1 retains the B0 ordering. A recovered parent may replace only the tail of top-five when the
local Qwen reranker scores it higher for the targeted missing-aspect query than every current B0
top-five parent. The support margin was frozen at zero before the run.

| System | Full Evidence@5 | Wins | Losses | Ties |
|---|---:|---:|---:|---:|
| Frozen B0 | 93.96% | — | — | — |
| Recovery V2.1 | **94.26%** | **1** | **0** | 330 |

V2.1 inserted evidence for 6/331 questions and fully recovered `mln131_0010`. It passes the
predefined development gate (wins > losses and at most one regression). The gain is only +0.30
percentage points, so the component remains opt-in and is not claimed as hidden-validated.

Enable the demo candidate with `VIETTHEORY_AGENTIC=1`. The default remains frozen B0 until a new
candidate freeze and an explicitly authorized hidden evaluation.
