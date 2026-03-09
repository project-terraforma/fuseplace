# FusePlace

Project A: Places Attribute Conflation (CRWN 102, Winter 2026)  
Team: Satvik Khanna

## Project Goal

Different sources often describe the same real-world place with conflicting values (name, website, phone, address, category, etc.).  
The goal is to build a pipeline that:

1. Creates a golden dataset with ground-truth labels.
2. Implements automated attribute selection logic.
3. Compares methods and reports F1 scores.

## Approach

### Data

- **Source**: 2,000 pre-matched place pairs from [Overture Maps](https://overturemaps.org/) (Meta + Microsoft sources).
- **Expansion**: 1,449 additional pairs created by pairing Overture records with [Yelp Fusion API](https://docs.developer.yelp.com/docs/fusion-intro) data, giving genuine attribute differences between sources.
- **Total**: 3,449 place pairs across 7 attributes (`names`, `categories`, `websites`, `phones`, `addresses`, `emails`, `socials`).

### Golden Dataset & Labeling

Ground-truth labels were built in three layers:

1. **Trivial auto-labeling** — if one side is missing, the other wins.
2. **Yelp verification** — for 1,449 US businesses matched via the Yelp Fusion API, labels were assigned by comparing both Overture sources against the Yelp-verified data (independent third-party ground truth).
3. **Domain heuristic labeling** — for records without Yelp coverage, attribute-specific heuristics (name completeness, address field count, phone country codes, category specificity, URL quality) were used.

Total: **24,143 labeled attribute pairs** across 3,449 records.

### ML Pipeline (Random Forest)

A single **Random Forest** classifier is trained on all attributes using golden labels (manual + proxy). Features include confidence, quality scores, source counts, token counts, and pair similarity. Evaluated via 80/20 stratified train/test split.

### Rule-Based Baseline

A heuristic baseline using confidence scores, attribute quality, and content completeness for comparison.

## Results

**Holdout F1 (20% held-out test set):** Run `python3 -m scripts.conflation.ml_selection` to retrain and see current metrics.

**ML vs Rule-based (against golden labels):**

| Method     | F1   | Precision | Recall | Accuracy |
|------------|------|-----------|--------|----------|
| ML         | 0.83 | 0.87      | 0.79   | 0.81     |
| Rule-based | 0.75 | 0.69      | 0.84   | 0.68     |

## Repository Structure

```text
fuseplace/
├── data/
│   ├── project_a_samples.parquet        # Original 2,000 Overture pairs
│   ├── yelp_verified_dataset.parquet    # 1,449 Overture-vs-Yelp pairs
│   └── merged_dataset.parquet           # Combined dataset (3,449 pairs)
├── analysis/inspection/
│   ├── golden/
│   │   ├── golden_dataset_template.json # Ground-truth labels
│   │   ├── labeling_worksheet.csv       # Labeling export
│   │   └── yelp_lookups.json            # Cached Yelp API responses
│   ├── attributes/                      # Per-attribute pair samples
│   └── side_by_side/                    # Side-by-side comparison samples
├── reports/
│   ├── audit/                           # Data quality audit CSVs
│   └── conflation/
│       ├── ml_training_metrics.json     # ML holdout metrics
│       ├── ml_attribute_decisions.csv   # ML predictions per attribute
│       ├── rule_attribute_decisions.csv # Rule-based predictions
│       └── method_evaluation_against_golden.csv
├── scripts/
│   ├── inspect_dataset.py               # Schema, missingness, golden template
│   ├── data_audit.py                    # Data quality audit
│   ├── label_golden.py                  # Golden dataset labeling helper
│   ├── yelp_verify.py                   # Yelp Fusion API verification
│   ├── build_yelp_pairs.py              # Overture-vs-Yelp pair builder
│   ├── auto_label.py                    # Domain heuristic auto-labeler
│   ├── expand_golden.py                 # Expand golden to all records
│   ├── fetch_overture.py                # Pull extra data from Overture Maps S3
│   ├── run_conflation.py                # Run rule-based + ML + evaluation
│   ├── conflation/
│   │   ├── rule_based_selection.py      # Rule-based conflation
│   │   ├── ml_selection.py              # Random Forest ML training & prediction
│   │   └── evaluate_methods.py          # Rule vs ML evaluation
│   ├── attributes/                      # Per-attribute inspection scripts
│   └── utils/
│       ├── conflation.py                # Features, rules, proxy labels
│       ├── parsing.py                   # Normalization (names, phones, URLs, etc.)
│       └── io.py                        # Parquet loading, CSV/JSON writing
├── tests/
│   ├── test_parsing.py
│   └── test_rule_decision.py
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

```bash
# 1. Inspect and audit
python3 -m scripts.inspect_dataset
python3 -m scripts.data_audit

# 2. Build golden labels (if needed)
python3 -m scripts.label_golden
# Optional: python3 -m scripts.yelp_verify --api-key YOUR_YELP_KEY --limit 2000

# 3. Run both rule-based and ML conflation, then evaluate
python3 -m scripts.run_conflation

# Or run individually:
python3 -m scripts.conflation.rule_based_selection
python3 -m scripts.conflation.ml_selection
python3 -m scripts.conflation.evaluate_methods
```

## Run Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Key Design Decisions

- **Rule-based baseline** — confidence + quality heuristics for a simple, interpretable baseline.
- **Random Forest ML** — single shared model trained on golden labels, suitable for an 8-week project scope.
- **Golden labels** — manual labeling plus proxy labels (confidence + quality) for weak supervision where manual labels are sparse.
