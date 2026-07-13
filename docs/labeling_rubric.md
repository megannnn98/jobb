# Review Labeling Rubric

Task: label whether the author describes work or employment experience with the company in `name` as positive, negative, or unclear.

## Labels

### `positive_about_work`

Use when the review clearly says working for, applying to, or cooperating with the company is good.

Signals:
- recommends the employer or job;
- praises salary, payments, management, team, schedule, working conditions, career growth, or fair treatment;
- describes a successful employment or contractor experience.

### `negative_about_work`

Use when the review clearly says working for, applying to, or cooperating with the company is bad.

Signals:
- warns not to work there or not to apply;
- complains about unpaid salary, deception, illegal employment, toxic management, bad conditions, fines, turnover, overwork, or dismissal;
- describes the employer as unreliable or unsafe for job seekers.

### `manual_review`

Use when the correct class is not clear enough for training.

Signals:
- text is not about work or employment despite `is_about_work=1`;
- only a vacancy repost, news item, joke, quote, or third-party fragment;
- sentiment is mixed and the final recommendation is unclear;
- too little context, broken text, parser artifact, or contradictory evidence;
- the author discusses the company/product generally, not work experience.

## Confidence

Use a decimal from `0.00` to `1.00`.

- `0.90-1.00`: explicit recommendation or warning with strong evidence.
- `0.70-0.89`: likely class, but evidence is less direct or text is noisy.
- `<0.70`: mark `manual_review` or set `needs_audit=1`.

## Audit Fields

- `label_source`: how the label was produced. Expected values: `llm`, `heuristic`, `human`, or `imported`.
- `needs_audit`: `1` if the row should be reviewed by a human, else `0`.
- `audit_reason`: short reason, for example `low_confidence`, `mixed_sentiment`, `not_about_work`, `parser_noise`, or `random_audit`.

## Decision Rule

Prefer `manual_review` over a forced positive/negative label when the text would confuse a supervised model or when a reasonable reviewer could disagree.
