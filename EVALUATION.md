# Evaluation plan

Create independently reviewed examples before making quality claims. For each example, save a small project ZIP, expected observations and their exact file/line references, likely hypotheses, and what the data does not establish. Score (1) citation existence, (2) evidence relevance, (3) unsupported claims, (4) quality of controlled test, (5) clarity. Use the app's automated citation check only for (1); human review is needed for semantic correctness. Include both adequate and missing-evidence cases, and report failures, sample size and uncertainty. This is a plan, not an evaluation already performed.

Toloka can optionally help obtain human assessments or curate reference data, depending on available tools and pricing. Do not upload private user projects without permission. No Toloka API integration is included.

## Initial labeled cases for manual review

These labels describe expected observable evidence, not verified root causes. The cases are synthetic and have not yet been independently scored. Use the case IDs in `evaluate_agent.py` and review the full output and cited lines.

| Case | Expected evidence | What must not be asserted as fact |
| --- | --- | --- |
| `error_only` | `error.log:2` reports a NaN error | The source of missing values or the exact estimator is known |
| `f1_only` | `run.log:1` reports validation F1 of 0.42 | A particular cause or baseline score is established |
| `generalization_gap` | `run.log:1-2` reports 0.96 train and 0.70 validation accuracy | The gap proves overfitting or identifies its cause |
| `no_metrics` | No supported metric or error is supplied | The outcome or failure cause is known |
| `scaling_hypothesis` | Scaling is not shown in the excerpt; `run.log:1-2` reports 0.55 and 0.53 | Scaling is absent from the whole run or must improve it |

For each case record pass, needs improvement, or incorrect, plus exact unsupported claims and reviewer notes. This small checklist is not an aggregate accuracy benchmark or proof of human validation.
