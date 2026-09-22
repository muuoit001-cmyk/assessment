# 2-3 Minute Loom Presentation Script
## Spotter Machine Learning Engineer Assessment: Freight Rate Prediction

> **Target Duration:** 2 minutes 30 seconds (150 seconds)  
> **Speaker Goal:** Deliver a crisp, confident, senior ML engineer presentation covering all five assessment rubric topics.

---

### Timing Breakdown
| Time | Section | Screen to Share |
|---|---|---|
| **0:00 – 0:30** | Introduction & Key EDA Findings | `train-test.csv` summary / Slide / PDF Report |
| **0:30 – 1:00** | Data Quality Issues Identified & Fixes | Code in `src/data_loader.py` |
| **1:00 – 1:35** | Training & Validation Strategy (Data Split) | Table in `freight_rate_prediction_report.pdf` |
| **1:35 – 2:05** | Model Selection & Benchmarking Results | Benchmark results table / `src/train.py` |
| **2:05 – 2:30** | December Prediction Chart & Code Walkthrough | `scorer_results/candidate_december.png` & `src/predict.py` |

---

### Spoken Script

#### 1. Introduction & Key Exploratory Findings [0:00 – 0:30]
*(Screen: Show the Executive Summary or Dataset Overview from the PDF Report)*

> *"Hi everyone, today I'm walking through my end-to-end Machine Learning solution for the Spotter Freight Rate Prediction challenge.*
>
> *When exploring the dataset—which contains 48,000 historical spot loads from January through October 2025—a few key patterns immediately emerged:*
> 1. *First, freight rate determination is strongly distance-driven, but modulated by spot market capacity, payload weight, equipment type, and weekly shipper cycles.*
> 2. *Second, the `quote_signal` column approximates a baseline rate per mile. Across all loads, the product of distance and `quote_signal` represents the unadjusted quote benchmark, which we use as a strong baseline.*
> 3. *Third, all 64 unique pickup and delivery locations map 1-to-1 deterministically to exact geographic coordinates."*

---

#### 2. Data Quality Issues & How They Were Addressed [0:30 – 1:00]
*(Screen: Show `src/data_loader.py` highlighting lines 40–70)*

> *"During data audit, I identified three major data quality issues:*
> 1. ***Negative Weights:** 292 loads in the training set and 145 loads in the validation set had negative weights, reaching up to -47,500 lbs. Because the absolute values match standard trailer payloads, this was clearly a sign-inversion UI error. I sanitized this using `abs(weight)`.*
> 2. ***Missing Cargo Weights:** Around 0.6% of weights were missing. I imputed these using median payload values grouped by equipment type—Dry Van, Reefer, and Flatbed.*
> 3. ***Missing Market Index Values:** 374 missing values in train and 249 in validation. Because daily market index varies with very low variance (sigma of ~0.025), I imputed missing values using the cross-sectional daily median for that specific shipping date.*
> 4. ***Synthetic December Scenario Gap:** The December input file omits `quote_signal` and `market_index`. I resolved this cleanly by estimating lane-specific quote signals from historical Lexington-to-Fort Wayne loads and utilizing daily December market index medians from the validation set."*

---

#### 3. Training & Validation Approach: How the Data Was Split [1:00 – 1:35]
*(Screen: Show Section 3 of `freight_rate_prediction_report.pdf`)*

> *"For our validation strategy, **we explicitly avoided random K-Fold cross-validation**. Freight spot rates are non-stationary time-series data governed by seasonal market tightening and weekly demand cycles. Randomly shuffling loads would cause look-ahead data leakage—training on future loads to predict the past—leading to overly optimistic metrics.*
>
> *Instead, I implemented a **chronological out-of-time holdout split**:*
> - *The training set uses Months 1 through 8 (January through August, ~38,500 loads).*
> - *The holdout validation set uses Months 9 and 10 (September and October, ~9,500 loads).*
>
> *This 2-month validation horizon directly mirrors the 2-month unlabelled evaluation horizon (November through December), giving us an honest, leakage-free benchmark."*

---

#### 4. Model Selection & Reasoning [1:35 – 2:05]
*(Screen: Show Section 5 Benchmark Table in the PDF Report)*

> *"Looking at the benchmarks on our out-of-time holdout set:*
> - *The naive quote baseline had an MAE of **$246.02**.*
> - *A regularized Ridge Regression improved this to **$191.54**.*
> - *LightGBM and XGBoost reached **$173.37** and **$179.96**.*
> - ***CatBoost was our top-performing standalone model**, achieving an MAE of **$131.90** and an out-of-time MAPE of just **6.09%**, cutting error nearly in half compared to baseline.*
>
> *I chose gradient boosted trees—specifically CatBoost and LightGBM—because tabular freight data features intricate non-linear relationships, such as interactions between payload density, transit circuitousness, and day-of-week seasonality. CatBoost's ordered boosting and superior handling of categorical lane and equipment combinations prevented overfitting on sparse lanes."*

---

#### 5. Code Walkthrough & December Prediction Chart [2:05 – 2:30]
*(Screen: Switch to `scorer_results/candidate_december.png`, then quickly show `score.py` execution)*

> *"Finally, here is the December prediction chart generated by `score.py` for the fixed Lexington-to-Fort Wayne dry van lane.*
>
> *You can observe two vital real-world freight dynamics:*
> 1. *Clear intra-week cyclicality—rates peak midweek during high dispatch volume and dip on weekends.*
> 2. *A distinct upward trajectory from December 22nd through New Year's Eve, rising to $852, capturing end-of-year holiday capacity tightening and driver availability shortages.*
>
> *All 12,000 validation loads and 31 December predictions passed the official scorer with zero formatting errors. The codebase is organized modularly under `src/` with complete instructions in the `README.md`. Thank you!"*
