# Predicting 30-Day Hospital Readmission in Diabetic Inpatients

A ranked-worklist model for hospital discharge planning, built on ten years of clinical records from 130 US hospitals.

**Deliverable 2 — Data Preprocessing and Initial Model Development**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eYXfmnfr8qVV2nsrU0fsjE2HV467Lqp9)

---

## What this is

Hospital quality-operations teams are accountable under the CMS Hospital Readmissions Reduction Program, which withholds up to 3% of Medicare reimbursement for excess 30-day readmissions. Discharge-planning capacity is fixed: follow-up calls, home-health referrals and medication reconciliation reach roughly the top 20% of discharges, not all of them.

The team does not need a system that decides anything. It needs those limited slots allocated well.

So this is a **ranked worklist**, not an automated decision — and that framing drives every methodological choice below.

| Framing decision | Consequence |
| --- | --- |
| Output is a ranked list, not an action | Ranking quality (PR-AUC) is the headline metric, not accuracy |
| The positive class is a small minority | A model predicting "nobody is readmitted" scores high on accuracy, so accuracy is disqualified |
| A missed readmission costs more than an unnecessary call | The threshold is set by capacity, not left at 0.50 |
| Output informs clinician judgement | The model must be interpretable and auditable across groups |

---

## Quickstart

**Colab (recommended).** Click the badge above. The first cell downloads the dataset directly from UCI — no manual upload needed.

**Local.**

```bash
git clone https://github.com/<org>/<repo>.git
cd <repo>
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Deliverable2_Readmission.ipynb
```

Run all cells in order. Total runtime is a few minutes, dominated by the grid search and permutation importance.

---

## Repository layout

```
.
├── Deliverable2_Readmission.ipynb   # the analysis — single source of truth
├── deliverable2_pipeline.py         # same pipeline as a plain script
├── Deliverable2_Report.docx         # written report (submission length)
├── Deliverable2_Report_Full.docx    # written report (unabridged)
└── README.md
```

Intermediate CSVs are **regenerated, not committed**, so the pipeline reproduces from the raw UCI archive alone. `RANDOM_STATE = 42` is fixed throughout — any member re-running the notebook reproduces identical splits and metrics.

Suggested `.gitignore`:

```gitignore
*.zip
diabetic_data.csv
IDS_mapping.csv
readmission_analysis_ready.csv
.ipynb_checkpoints/
```

---

## Dataset

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999-2008* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5230J>

101,766 inpatient encounters · 130 hospitals · 50 features · 1999–2008 · licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Target: readmission within 30 days of discharge. The raw label has three levels; `>30` is treated as negative because a readmission at 45 days falls outside both the CMS penalty window and the team's intervention window.

---

## The decisions that matter

Most of the work here is preprocessing, and most of it is invisible if you skip it — **none of these steps raises an error when omitted.** They just quietly change what the model learns.

**Two parsing decisions precede any analysis.** Missing values are a literal `?`, so `na_values=['?', '']` is required. More consequentially, `keep_default_na=False` stops pandas coercing the string `'None'` to a null — in `A1Cresult` that does not mean missing, it means *the test was never ordered*, which is the exact variable the source study investigated.

**Cohort construction.** Three filters, each encoding a clinical assumption:

1. **Expired and hospice discharges removed.** These patients could not be readmitted; retaining them trains on an outcome that was never possible.
2. **Repeat encounters removed**, keeping each patient's first admission. Without this the same individual lands in train *and* test, so validation partly measures memorisation. Largest source of optimistic bias in published work on this dataset.
3. **`gender == 'Unknown/Invalid'` removed** — compared against the literal string, not a null. The dataset stores it as text, so `.notna()` removes nothing at all.

**`payer_code` is audited, not predicted.** It survives the 40% missingness rule, so using it was available. Insurance status is an ability-to-pay proxy, and encoding it into a clinical triage score is not defensible — but checking whether the model underserves self-pay patients is.

**Missing race gets an explicit `Unknown` level, not mode imputation.** Imputing a protected attribute assigns the majority race to patients whose race was never recorded, producing a fairness audit on a population that does not exist.

**Skew is measured, not assumed.** Count columns are scored with `.skew()` and only those above 1.0 are winsorized and log-compressed. Applying `log1p` to an already-symmetric column introduces left skew rather than removing right skew — `num_lab_procedures` is the trap here.

**ICD-9 grouping.** Three diagnosis columns hold ~700 codes each. One-hot encoding them raw adds ~2,100 near-empty columns. Collapsing to the nine clinical groups of Strack et al. yields about 30 informative indicators.

---

## Pipeline

```
load → parse → cohort filters → feature engineering → stratified split
     → ColumnTransformer (inside Pipeline)  → baselines → GridSearchCV
     → final_model → capacity threshold → interpretability → fairness audit
```

Every transform lives **inside** a scikit-learn `Pipeline`, so it refits independently within each cross-validation fold. The split happens before any transform is fitted — fitting a scaler on the full dataset and splitting afterwards leaks test statistics and inflates the reported score.

The grid-searched forest becomes `final_model`, and every downstream step uses that estimator. Tuning one model and evaluating a different one is a silent and common error; naming the final estimator once makes it impossible.

Tuning optimises `average_precision`, not accuracy. Against this class balance, accuracy rewards predicting the majority class everywhere.

---

## Results

> Fill these from your executed run. The notebook prints every figure below.

| Model | ROC-AUC | PR-AUC | Accuracy |
| --- | --- | --- | --- |
| Logistic Regression | — | — | — |
| Random Forest (untuned) | — | — | — |
| **Random Forest (tuned, final)** | — | — | — |

Reference values, both printed in the notebook:

- **Majority-class accuracy** (predict nobody readmitted) — compare any accuracy figure against this, not against zero.
- **Random-classifier PR-AUC** (equals the cohort positive rate) — compare PR-AUC against this.

**At the operating threshold** (top 20% of predicted risk): recall — · precision — · lift — ×

Discrimination on this dataset is modest in all published work, typically ROC-AUC 0.64–0.68. That is a property of the problem, not a defect of the pipeline: 30-day readmission is driven substantially by social and post-discharge factors — housing stability, caregiver support, medication affordability, outpatient access — that an inpatient encounter record does not observe.

---

## Fairness and ethics

The notebook carries a **quantitative audit, not a prose assertion**. Group-level recall, false-positive rate, precision and AUC are computed by race, gender, age band and payer code — at the deployment threshold, not at 0.50, because auditing a threshold the system will never use answers a question nobody asked.

The criterion is **equal opportunity**: among patients who genuinely were readmitted, does the model flag them at the same rate across groups? A gap in recall means one group is systematically denied the follow-up resource.

Recall is a ratio over actual readmissions, so a group's recall is only estimable if that group holds enough of them. The audit reports the spread twice — across all groups, and restricted to groups with at least 30 actual readmissions. **A headline spread above the 0.05 review trigger is evidence of bias only if it survives that restriction.**

Known structural biases: `race` is staff-recorded rather than self-reported; the target captures readmission only to the same system, so a patient readmitted elsewhere is labelled negative; and the data encodes 1999–2008 practice, including its inequities.

On transparency, the forest supports aggregate explanation but not per-patient explanation. A coordinator who cannot say why a patient was flagged cannot defend that decision — so the model is documented as a triage aid with a stated error profile. It produces a ranked worklist and never withholds care.

---

## Deviations from the project hints

The course guidance is dataset-agnostic. Five of its defaults are harmful here, so the substitutions are documented rather than made silently.

| Hints guidance | Problem here | Substitution |
| --- | --- | --- |
| `fillna(method='ffill')` | Propagates one patient's race or diagnosis into another's record; `method=` is removed in pandas 3.x | Column-specific handling; explicit `Unknown` level for categoricals |
| `drop_duplicates()` | Removes exact duplicate rows — there are none, `encounter_id` is unique. The real leak is duplicate *patients* | `drop_duplicates('patient_nbr')` |
| `scaler.fit_transform(df)` then split | Fits the scaler on test data — leakage | Split first; scaling inside a `Pipeline`, refit per fold |
| `GridSearchCV(scoring='accuracy')` | Accuracy is maximised by predicting nobody is readmitted | `scoring='average_precision'` |
| `get_dummies(df)` | Would one-hot ~700 ICD-9 codes per column | Clinical grouping, then `OneHotEncoder(min_frequency=20)` |

---

## Team

| Member | Sections | Contribution |
| --- | --- | --- |
| Chin | 1–2 | Problem framing; parsing decisions; cohort filters and audit trail; ICD-9 and discharge grouping; derived features; exploratory figures |
| Om | 3 | Stratified split; `ColumnTransformer` pipeline; baseline models; grid search; capacity threshold; permutation importance |
| San | 4–7 | Fairness audit; ethics and transparency; deviations table; summary; written report |

Each member reviews one other's work — Chin reviews Om's, Om reviews San's, San reviews Chin's — so nothing merges on its author's approval alone.

**Work runs sequentially, not in parallel.** Two people editing the same `.ipynb` produce cell-metadata conflicts, and the work is genuinely ordered: Chin's filters change the data Om's models consume, and Om's outputs feed San's audit. Open Colab from GitHub and save back with **File → Save a copy in GitHub** to keep one lineage. One commit per fix, so the history records who did what.

---

## Next steps (Deliverable 3)

1. Gradient boosting (XGBoost / LightGBM) with early stopping as a stronger tabular baseline.
2. Calibration assessment — a coordinator reading "38% risk" needs that to mean something. `CalibratedClassifierCV` plus a reliability curve.
3. Cost-sensitive threshold optimisation using the team's real cost ratio rather than a capacity proxy.
4. Grouped cross-validation by hospital, to estimate transfer to an unseen site.
5. SHAP values for per-patient explanations, closing the transparency gap above.

---

## References

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999-2008* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5230J>

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. *BioMed Research International*, 2014, Article 781670. <https://doi.org/10.1155/2014/781670>

---

<sub>Course project. The model is a triage aid, not a clinical decision system, and is not validated for patient care.</sub>
