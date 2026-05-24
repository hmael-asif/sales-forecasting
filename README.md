# 📈 Sales Forecasting with CatBoost & Upgini

A machine learning project that forecasts retail sales using **CatBoost**, enhanced with automatic feature enrichment via **Upgini**. The project includes a clean side-by-side comparison between a baseline model and an enriched model to quantify the value added by external data.

---

## 🧠 Project Overview

This notebook-based project trains a gradient boosting regressor on historical retail sales data. It then leverages Upgini's feature enrichment to automatically discover and inject relevant external features (e.g., calendar signals, macroeconomic indicators) — and measures whether they improve forecast accuracy.

---

## 🗂️ Project Structure

```
SalesForecasting-MLProject.ipynb   # Main Colab notebook
README.md                          # Project documentation
```

---

## 📦 Dependencies

Install all required packages with:

```bash
pip install -Uq upgini catboost
```

| Package    | Purpose                                      |
|------------|----------------------------------------------|
| `catboost` | Gradient boosting regressor for sales forecasting |
| `upgini`   | Automatic feature enrichment from external data sources |
| `pandas`   | Data loading, manipulation, and preprocessing |

---

## 📊 Dataset

- **Source:** [`upgini/upgini` GitHub repository](https://github.com/upgini/upgini/raw/main/notebooks/train.csv.zip)
- **Size:** 10,000 rows sampled (random state = 0) from the full dataset
- **Features:**
  - `date` — transaction date (parsed as `datetime`)
  - `store` — store identifier (treated as categorical)
  - `item` — item identifier (treated as categorical)
  - `sales` — target variable (units sold)

**Train/Test Split:** Chronological split at `2017-01-01`
- **Train:** All records before January 1, 2017
- **Test:** All records from January 1, 2017 onward

---

## 🔄 Workflow

```
Raw Data
   └─► Sample & Preprocess
          └─► Train/Test Split (time-based)
                 ├─► Baseline Model (CatBoost on raw features)
                 └─► Upgini Enrichment
                        └─► Enriched Model (CatBoost on enriched features)
                               └─► SMAPE Comparison
```

### Steps

1. **Data Loading & Preprocessing**
   - Load CSV, sample 10,000 rows
   - Cast `store` and `item` as strings (categorical)
   - Parse `date` as datetime and sort chronologically

2. **Feature Enrichment with Upgini**
   - Initialize `FeaturesEnricher` with `date` as the search key
   - Use `CVType.time_series` to respect temporal ordering in cross-validation
   - Call `.fit()` with the train set and test set as `eval_set`
   - Call `.calculate_metrics()` to preview enrichment impact
   - Call `.transform()` on both train and test sets to obtain enriched features

3. **Baseline Model**
   - Train `CatBoostRegressor` on original features
   - Evaluate on test set using **SMAPE** (Symmetric Mean Absolute Percentage Error)

4. **Enriched Model**
   - Train the same `CatBoostRegressor` on Upgini-enriched features
   - Evaluate on test set using **SMAPE**
   - Verify feature column consistency between enriched train and test sets

---

## 📐 Evaluation Metric

**SMAPE** — Symmetric Mean Absolute Percentage Error

$$\text{SMAPE} = \frac{100\%}{n} \sum_{t=1}^{n} \frac{|F_t - A_t|}{(|A_t| + |F_t|)/2}$$

Lower SMAPE = better forecast accuracy. Results are compared between the baseline and enriched models to quantify the uplift from external features.

---

## ▶️ How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/drive/1x1Kj_J2jUYaPziRw55wgV_eWw7XLuMQt)
2. Run all cells in order (`Runtime > Run all`)
3. Review the metric output from `enricher.calculate_metrics()` and the final SMAPE scores

> **Note:** Upgini feature enrichment requires a free account. If you hit the unregistered-user limit, `enricher.transform()` may return `None`. The notebook handles this gracefully with a warning message.

---

## ⚠️ Notes & Caveats

- **Upgini limits:** Free/unregistered users may encounter enrichment limits. Register at [upgini.com](https://upgini.com) for full access.
- **Reproducibility:** `random_state=0` is set for both sampling and model training to ensure consistent results.
- **Column consistency check:** The notebook verifies that enriched train and test feature sets share the same columns before model training.

---

## 📚 References

- [CatBoost Documentation](https://catboost.ai/docs/)
- [Upgini Documentation](https://docs.upgini.com/)
- [Original Upgini Example Notebook](https://github.com/upgini/upgini/tree/main/notebooks)

---

## 📄 License

This project is for educational and demonstration purposes.
