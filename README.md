# FusePlace

Project A: Places Attribute Conflation (CRWN 102, Winter 2026)  
Team: Satvik Khanna, Kate Mikhailova

## Project Goal

Different sources often describe the same real-world place with conflicting values (name, website, phone, address, category, etc.).  
Our goal is to build a pipeline that:

1. Creates a golden dataset for manual ground truth.
2. Implements automated attribute selection logic.
3. Compares methods and reports results.

## What We Built

This repository was completed from an initial empty starter and now includes:

1. **Data + project structure**
Added `data/project_a_samples.parquet` and reproducible folder layout for analysis, reports, scripts, and tests.

2. **Inspection pipeline**
Built scripts to inspect schema, missingness, side-by-side base/current pairs, and disagreement rates. Generated a golden dataset template for manual labeling.

3. **Attribute-specific analysis**
Added per-attribute inspection scripts for `categories`, `addresses`, `phones`, and `websites`.

4. **Data quality audit**
Added audit scripts that export missingness, confidence stats, and conflict examples.

5. **Two conflation baselines**
Implemented a **rule-based** selector using confidence + attribute quality heuristics, and an **ML baseline** (logistic regression) using features like confidence, token overlap, missingness, source counts, and quality deltas.

6. **Evaluation tooling**
Added a comparison script for rule vs ML against manual golden labels.

7. **Tests**
Added unit tests for parsing/normalization and rule-decision behavior.

## Current Run Snapshot (from provided sample parquet)

Dataset:
- 2,000 rows
- 22 columns
- 7 core attribute pairs (`names`, `categories`, `websites`, `phones`, `addresses`, `emails`, `socials`)

Observed conflict rates (comparable rows):
- `phones`: 73.35%
- `categories`: 70.60%
- `addresses`: 52.30%
- `names`: 50.85%
- `websites`: 39.60%
- `socials`: 40.65% (on non-missing comparable subset)

ML training status:
- Model currently trains successfully.
- Since manual labels are still empty, current ML metrics are based on proxy labels only.
- Final method comparison requires filling manual labels in the golden file.

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

## How To Reproduce

Run full pipeline:

```bash
python3 -m scripts.inspect_dataset
python3 -m scripts.attributes.inspect_categories
python3 -m scripts.attributes.inspect_addresses
python3 -m scripts.attributes.inspect_phones
python3 -m scripts.attributes.inspect_websites
python3 -m scripts.data_audit
python3 -m scripts.conflation.rule_based_selection
python3 -m scripts.conflation.ml_selection
python3 -m scripts.conflation.evaluate_methods
```

Key outputs:
- `analysis/inspection/golden/golden_dataset_template.json`
- `reports/audit/audit_conflict_rates.csv`
- `reports/conflation/rule_attribute_decisions.csv`
- `reports/conflation/ml_attribute_decisions.csv`
- `reports/conflation/method_evaluation_against_golden.csv`

## Manual Labeling Step (Required for Final Evaluation)

Fill `labels` in:
- `analysis/inspection/golden/golden_dataset_template.json`

Use:
- `current` if conflated value is better
- `base` if base value is better
- leave blank when undecidable

Then rerun:

```bash
python3 -m scripts.conflation.ml_selection
python3 -m scripts.conflation.evaluate_methods
```

## Run Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Notes

- Default input parquet is `data/project_a_samples.parquet`.
- Most scripts support `--input` for alternate parquet files.
- This repo is designed to map directly to Project A deliverables: golden data, selection algorithms, and evaluation report artifacts.
