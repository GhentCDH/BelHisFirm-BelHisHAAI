# IAA NER — Inter-Annotator Agreement for NER

Script for calculating inter-annotator agreement (IAA) on NER annotations exported from Label Studio.

No external dependencies — only the Python standard library.

## Input

A Label Studio JSON export where each record has the fields:
`id`, `text`, `label`, `annotator`, `annotation_id`, `created_at`, `updated_at`, `lead_time`.

Tasks annotated by multiple annotators (same `id`, different `annotator`) are used for IAA calculations.

## Metrics

- **Span-level F1** — a span only counts as a match if `start`, `end`, and `label` are all identical (exact boundary + exact label). Macro-averaged across tasks.
- **Cohen's Kappa** — character-level agreement, treating each character position as a classification decision (`O` or a label name).
- **Agreement classification** per task: `full` (identical span sets), `partial` (some overlap), or `none` (no overlap).

## Usage

```bash
# Full report with per-label breakdown
python iaa_ner.py path/to/export.json

# Filter all metrics to a single label type
python iaa_ner.py export.json --label person

# Compare only two specific annotators (by their numeric ID)
python iaa_ner.py export.json --annotators 5 15

# List tasks where all annotators fully agree
python iaa_ner.py export.json --show-full

# List tasks with disagreements and show the diff per annotator
python iaa_ner.py export.json --show-issues

# Only print task IDs instead of full details (works with --show-full and --show-issues)
python iaa_ner.py export.json --show-issues --minimal

# Exclude one or more label types from all metrics
python iaa_ner.py export.json --ignore-label honorific_title
python iaa_ner.py export.json --ignore-label honorific_title --ignore-label number_of_votes

# Export disagreement tasks to a new Label Studio-compatible JSON file
python iaa_ner.py export.json --export-issues issues.json

# Combine flags
python iaa_ner.py export.json --label person --annotators 5 15 --show-issues --minimal
python iaa_ner.py export.json --ignore-label honorific_title --export-issues issues.json
```

## Output example

```
============================================================
  Inter-Annotator Agreement Report
============================================================
  Annotators found         : [5, 15, 16, 17, 18, 19]
  Total tasks              : 697
  Multi-annotated tasks    : 302

  Full agreement           :   63  (20.9%)
  Partial agreement        :  234  (77.5%)
  No agreement             :    5  (1.7%)

  Macro Precision          : 0.7564
  Macro Recall             : 0.7602
  Macro F1                 : 0.7485
  Cohen's Kappa (char)     : 0.8311

  Per-label F1:
  Label                             P      R     F1    TP    FP    FN
  person                        0.936  0.931  0.934  1033    71    76
  ...
```

## Importable functions

The script can also be imported and used programmatically:

```python
from iaa_ner import load_data, calculate_iaa, get_full_agreement_tasks, get_disagreement_tasks

data = load_data("export.json")

stats = calculate_iaa(data)
stats = calculate_iaa(data, label_filter="person")
stats = calculate_iaa(data, annotator_pair=(5, 15))
stats = calculate_iaa(data, ignore_labels={"honorific_title", "number_of_votes"})

full  = get_full_agreement_tasks(data)
full  = get_full_agreement_tasks(data, ignore_labels={"honorific_title"})
diffs = get_disagreement_tasks(data)
diffs = get_disagreement_tasks(data, ignore_labels={"honorific_title"})

# Export disagreement tasks as a Label Studio-compatible JSON
from iaa_ner import export_issues_to_labelstudio
n = export_issues_to_labelstudio(data, "issues.json")
n = export_issues_to_labelstudio(data, "issues.json", ignore_labels={"honorific_title"})
```
