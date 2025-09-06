# Mini-Report (≤1 page)

## Prompt design choices (with few-shot cues)
- **Policy stack**: clear label rules for *Positive / Negative / Neutral* with edge handling — sarcasm → Negative; negation (“not bad” → weak Positive, “not great” → weak Negative); 4–5★ or 1–2★ steer Positive/Negative; 3★ defaults to Neutral unless text leans; off‑topic → Neutral.
- **Evidence rules**: extract **0–5** short **verbatim** spans (≤10 words) that must be literal substrings of the review (prevents made‑up evidence).
- **Confidence rubric**: bins mapped to **[0,1]** to avoid ad‑hoc scores.
- **Modes**: **Strict** prefers Neutral on weak/mixed cues; **Lenient** picks a mild lean over Neutral.
- **Rating hint**: regex pulls `N/5` or “N stars” and passes `RATING_HINT:` into the prompt so explicit ratings inform the decision.
- **Few-shot (compact)**:
  - “Loved it … **5/5**” → **Positive** (high confidence)
  - “Story is dull … bored” (+ minor praise) → **Negative**
  - “**Not bad** … some jokes land” → weak **Positive**
  - “3★ … just okay” → **Neutral**
- **Budget control & parsing**: normalize whitespace, head+tail cap (~1600 chars), *first JSON blob* extraction, and sanitization (label whitelist, clipped confidence, ≤2‑sentence explanation, dedupbed evidence).

## Failure cases & mitigations
- **Overconfidence on mixed cues** → **Audit & nudge**: second, compact audit prompt rates evidence sufficiency; when first‑pass confidence falls in a mode band (Strict: **0.50–0.85**, Lenient: **0.55–0.80**), shrink toward 0.5 and blend (**20% weight**).
- **Hallucinated evidence** → enforce **substring‑only** evidence with length + dedupe checks before display.
- **Sarcasm/contrast misreads** → explicit policy lines + sarcastic few‑shot.
- **Negation pitfalls** → “not bad/not great” encoded in policy + examples.
- **Ratings ignored** → `RATING_HINT` line makes numeric stars/`N/5` actionable.

## Mini metrics (test set)
- **Accuracy**: **96.67%** (29 / 30)
- **Confusion matrix** (rows = Actual, cols = Predicted):

| label    | Positive | Neutral | Negative |
|:---------|---------:|--------:|---------:|
| Positive | **11**   | 0       | 0        |
| Neutral  | 0        | **9**   | 0        |
| Negative | 0        | 1       | **9**    |

*Single error: one **Negative → Neutral**.*

![Metrics screenshot](sandbox:/mnt/data/metrics.png)
