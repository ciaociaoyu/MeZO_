# OPT Result Audit

OPT-1.3B rows are used only as cross-architecture sanity checks. They are not presented as a reproduction of the original MeZO benchmark or as SOTA.

All available tasks in the processed comparison are retained, including TREC. Status bins use fixed delta thresholds:

- `near-default`: |delta| <= 0.01
- `moderate gap`: 0.01 < |delta| <= 0.05
- `substantial gap/failure`: |delta| > 0.05
