# FusePlace

Project A: Places Attribute Conflation (CRWN 102, Winter 2026)  
**Team:** Satvik Khanna

## Project Goal

Different sources often describe the same real-world place with conflicting values (name, website, phone, address, category, etc.). This project builds a pipeline that:

1. Creates a golden dataset with ground-truth labels
2. Implements rule-based and ML attribute selection
3. Compares both methods and reports F1 scores

## Approach

### Data

- **Source:** 2,000 pre-matched place pairs from [Overture Maps](https://overturemaps.org/) (Meta + Microsoft)
- **Attributes:** `names`, `categories`, `websites`, `phones`, `addresses`, `emails`, `socials`
- Optional expansion scripts (`build_yelp_pairs.py`, `fetch_overture.py`) can add more pairs for larger training sets

### Golden Labels

Ground truth is built from:

1. **Manual labeling** — human labels in `golden_dataset_template.json`
2. **Proxy labels** — confidence + quality heuristics where manual labels are missing (weak supervision)

### Rule-Based Baseline

Heuristic selection using confidence scores, attribute quality, and source counts. Simple and interpretable.

### ML Pipeline (Random Forest)

A single **Random Forest** classifier trained on all attributes. Features include confidence, quality scores, source counts, token counts, and pair similarity. Evaluated via 80/20 stratified train/test split.

## Results

| Method     | F1   | Precision | Recall | Accuracy |
|------------|------|-----------|--------|----------|
| ML         | 0.83 | 0.87      | 0.79   | 0.81     |
| Rule-based | 0.75 | 0.69      | 0.84   | 0.68     |

ML beats the rule-based baseline by ~8 F1 points. Holdout weighted F1 (20% test set) is ~0.81.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/fuseplace.git
cd fuseplace
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run rule-based + ML conflation, then evaluate
python3 -m scripts.run_conflation
```

## How To Reproduce

```bash
# 1. Inspect data
python3 -m scripts.inspect_dataset
python3 -m scripts.data_audit

# 2. Build golden labels (if needed)
python3 -m scripts.label_golden

# 3. Run both methods and evaluate
python3 -m scripts.run_conflation

# Or run individually:
python3 -m scripts.conflation.rule_based_selection
python3 -m scripts.conflation.ml_selection
python3 -m scripts.conflation.evaluate_methods
```

## Repository Structure

```
fuseplace/
├── data/
│   ├── project_a_samples.parquet    # 2,000 Overture place pairs
│   └── readme_project_a_samples.txt
├── analysis/inspection/
│   ├── golden/
│   │   ├── golden_dataset_template.json
│   │   └── labeling_worksheet.csv
│   ├── attributes/
│   └── side_by_side/
├── reports/
│   ├── audit/
│   └── conflation/
│       ├── ml_training_metrics.json
│       ├── ml_attribute_decisions.csv
│       ├── rule_attribute_decisions.csv
│       └── method_evaluation_against_golden.csv
├── scripts/
│   ├── run_conflation.py           # Run rule-based + ML + evaluation
│   ├── conflation/
│   │   ├── rule_based_selection.py
│   │   ├── ml_selection.py
│   │   └── evaluate_methods.py
│   ├── inspect_dataset.py
│   ├── data_audit.py
│   ├── label_golden.py
│   └── utils/
│       ├── conflation.py           # Features, rules, proxy labels
│       ├── parsing.py
│       └── io.py
├── tests/
├── requirements.txt
└── README.md
```

## Run Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Key Design Decisions

- **Rule-based baseline** — confidence + quality heuristics for a simple, interpretable baseline
- **Single Random Forest** — one shared model for all attributes, suitable for an 8-week project scope
- **Proxy labels** — weak supervision (confidence + quality) where manual labels are sparse
