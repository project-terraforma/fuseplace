# FusePlace

Project A: Places Attribute Conflation (CRWN 102, Winter 2026)

FusePlace is an end-to-end starter repository for **attribute-level conflation** on pre-matched place pairs. It is built to match the Project A mission from your slides:

1. Create a golden dataset (manual labels).
2. Build selection logic (rule-based and ML).
3. Evaluate which method performs better.

## Included Data

- `data/project_a_samples.parquet` (copied from your provided sample file)
- `data/readme_project_a_samples.txt`

Each row compares a conflated record (`names`, `phones`, `websites`, etc.) with a base record (`base_names`, `base_phones`, `base_websites`, etc.).

## Repository Structure

```text
fuseplace/
├── data/
├── analysis/inspection/
│   ├── attributes/
│   ├── golden/
│   └── side_by_side/
├── reports/
│   ├── audit/
│   └── conflation/
├── scripts/
│   ├── attributes/
│   ├── conflation/
│   └── utils/
├── tests/
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow

### 1) Inspect and sample the dataset

```bash
python3 -m scripts.inspect_dataset
```

Outputs:
- `analysis/inspection/side_by_side/side_by_side_sample.csv`
- `analysis/inspection/side_by_side/side_by_side_sample.jsonl`
- `analysis/inspection/golden/golden_dataset_template.json`
- `analysis/inspection/attribute_disagreement_rates.csv`

### 2) Run attribute-specific inspection

```bash
python3 -m scripts.attributes.inspect_categories
python3 -m scripts.attributes.inspect_addresses
python3 -m scripts.attributes.inspect_phones
python3 -m scripts.attributes.inspect_websites
```

Outputs land in `analysis/inspection/attributes/`.

### 3) Run data quality audit

```bash
python3 -m scripts.data_audit
```

Outputs land in `reports/audit/`.

### 4) Run rule-based conflation baseline

```bash
python3 -m scripts.conflation.rule_based_selection
```

Outputs:
- `reports/conflation/rule_attribute_decisions.csv`
- `reports/conflation/rule_selected_records.csv`
- `reports/conflation/rule_summary.csv`

### 5) Build ML conflation baseline

Manual labeling step:
- Fill `labels` in `analysis/inspection/golden/golden_dataset_template.json`.
- Use `current` or `base` per attribute when you can decide.
- Leave empty for undecidable records.

Then run:

```bash
python3 -m scripts.conflation.ml_selection
```

If labels are still mostly empty, ML training falls back to proxy labels by default.

Outputs:
- `models/ml_selector.joblib`
- `reports/conflation/ml_training_metrics.json`
- `reports/conflation/ml_attribute_decisions.csv`
- `reports/conflation/ml_selected_records.csv`
- `reports/conflation/ml_summary.csv`

### 6) Evaluate methods against manual labels

```bash
python3 -m scripts.conflation.evaluate_methods
```

Output:
- `reports/conflation/method_evaluation_against_golden.csv`

## Running Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Notes

- Default input parquet path is `data/project_a_samples.parquet`.
- Most scripts accept `--input` if you want to swap datasets.
- This repo is intentionally structured to support both course deliverables and reproducible experimentation.
