# Walmart Stock and Sales Forecasting Challenge
Consumer Instock Values Stockout-Aware Demand + Value of Instock 

## Why this project
In supply chain settings, **observed sales are not the same as true demand** when items are out-of-stock (OOS) or otherwise unavailable. This “censoring” creates bias in both model training and evaluation, and it can mislead prioritization decisions (e.g., which SKUs to place closer to customers or invest in to improve availability).

This repo builds a small **Consumer Instock Value (CIV)-style** prototype:
1) forecast demand in a **stockout-aware** way, and  
2) translate forecasts into a **Value of Instock** ranking under a counterfactual “better instock” scenario.

---

## Problem setup (CIV framing)
**Unit:** item × store × day  
**Goal:** predict demand for a future horizon (e.g., next 7/28 days) and estimate the **incremental units/value** if instock improves.

### Censoring
Let true demand be `d_t` and observed sales be `y_t`.
- If in stock: `y_t = d_t`
- If OOS: `y_t <= d_t` (we observe a lower bound)

Most public datasets don’t include true inventory signals, so I demonstrate the method by **simulating stockouts** (controlled censoring) and showing how naive evaluation becomes biased.

---

## Data
- **M5 Forecasting** data: daily sales with price and calendar features.
- **Synthetic OOS simulation:** create an `is_oos_t` flag with:
  - a low base OOS rate, and
  - higher OOS probability during high-demand contexts (promo/holiday),
  then censor observed sales to produce `y_obs`.

Artifacts are stored in `data/processed/` (parquet/csv).

---
## Exploratory Analysis
---

## Methods
### Backtesting (time-aware evaluation)
- **Rolling-origin backtest** (no random split) to avoid leakage.
- Metrics:
  - **WAPE** (primary; robust for retail scale)
  - RMSE (secondary)
- Key evaluation split:
  - Evaluate on **all days** vs **in-stock-only** days to highlight censoring bias.

### Models
1) **Seasonal naive**: `y_hat[t] = y_obs[t-7]`  
2) **LightGBM** (tabular):
   - lag features (1/7/14/28), rolling stats (7/28), price, calendar
3) **Deep learning (DeepAR)**:
   - probabilistic forecasting across many series with item/store embeddings
   - outputs quantiles / samples for uncertainty

### Stockout-aware training (core idea)
During training, time steps with `is_oos_t=1` are treated as **censored**:
- **Implemented:** mask / down-weight censored points in the loss (or drop censored steps from training windows).
- **Next iteration (not implemented):** censored likelihood (Tobit-style), where OOS points contribute `P(demand ≥ observed)`.

---

## Results (3 key figures)
1) **Censoring bias demo:** the same model looks “better” when evaluated on all days (many OOS zeros) vs in-stock-only days.  
2) **Rolling backtest comparison:** seasonal naive vs LightGBM vs DeepAR (masked censoring).  
3) **Example SKU-store:** probabilistic forecast intervals + counterfactual instock improvement and resulting Δunits/Δvalue.

Figures are exported to `reports/figures/`.

---

## Decision layer: Value of Instock (counterfactual)
To estimate “Value of Instock,” I define a counterfactual scenario with improved availability:
- reduce future OOS probability (or set future `is_oos=0` for selected SKUs)

For each item-store:
- `Δunits = E[demand_cf] - E[demand_base]`
- `Δvalue = Δunits × margin` (assumed constant margin; documented in `src/config.py`)
Uncertainty is propagated from DeepAR samples (e.g., p10/p50/p90 for Δvalue).

Outputs:
- `reports/tables/top_value_of_instock.csv` (ranked opportunities)

---

## Limitations + next steps
- This demo uses **synthetic OOS** due to public data limitations; production systems would use true inventory/offer availability and promise speed.
- Demand substitution/cannibalization is not modeled; next steps include:
  - censored likelihood training,
  - substitution-aware demand models,
  - hierarchical reconciliation across item/category/region,
  - monitoring for drift (OOS rate shifts, calibration drift).

---

## How to run
1) Install deps: `pip install -r requirements.txt`  
2) Run notebooks in order:
   - `notebooks/01_data_build_and_stockout_simulation.ipynb`
   - `notebooks/02_baselines_and_backtest.ipynb`
   - `notebooks/03_deepar_masked_loss_and_value_of_instock.ipynb`

