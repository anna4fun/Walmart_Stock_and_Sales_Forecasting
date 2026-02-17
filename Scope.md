Here’s a **4-day execution plan** that will give you an interview-ready CIV-flavored project **with deep learning + statistics** without boiling the ocean.

## Target scope (what you’ll finish in 4 days)

* Dataset: **M5** (simulate stockouts)
* Baseline: **seasonal naive + LightGBM**
* Deep learning: **DeepAR** (probabilistic) *or* TFT if you already know it (DeepAR is faster)
* Stockout handling: **loss masking** (implemented) + **censored likelihood** (describe as “next iteration”)
* Output: “**Value of Instock**” ranking + uncertainty bands

---

# Day 1 — Data + stockout simulation + backtesting scaffold

**Deliverable:** clean panel + rolling split function + synthetic OOS.

1. **Load + build panel**

* item-store-day sales
* merge calendar + price

2. **Create time features**

* day-of-week, week-of-year, holiday flags
* lags: 1, 7, 14, 28
* rolling means: 7/28

3. **Simulate stockouts (important for CIV story)** (SKIP)
   Create an OOS indicator `oos_t`:

* Start with random OOS at low rate (e.g., 3–8%)
* Add “realism”: higher probability of OOS during promo/holiday or high-demand periods
  Then create observed sales:
* `y_obs = y_true` if in stock
* `y_obs = 0` (or capped at a small number) if OOS

Also store:

* `is_censored = oos_t` (so you can mask loss later)

4. **Rolling backtest split**
   Implement something like:

* Train: days 1..T, Validate: next 28 days
* Slide forward 28 days, repeat 3–5 folds

✅ End of Day 1: You can already run baselines on fold 1.

---

# Day 2 — Baselines + metrics + “censoring bias” demo

**Deliverable:** baseline results + a plot that proves censoring matters.

1. **Baseline 1: seasonal naive**

* predict `y_hat[t] = y_obs[t-7]`

2. **Baseline 2: LightGBM**

* target: next-day or next-7-day sum
* features: lags + rollings + price + calendar + embeddings via categorical codes (item/store ids)

3. **Evaluation**
   Use rolling backtest.
   Metrics:

* **WAPE** (good for retail)
* RMSE (secondary)
  Add a **calibration / interval** placeholder: even just quantiles from residuals is fine.

4. **One killer figure**
   Show two evaluation regimes:

* Evaluate on **all days** (includes OOS → model looks “better” because many zeros)
* Evaluate on **in-stock only** days (true demand periods)
  This demonstrates **naive evaluation is biased under censoring** — very CIV.

✅ End of Day 2: You have a solid “statistics story” even before DL.

---

# Day 3 — Deep learning (DeepAR) + masking censored loss

**Deliverable:** Deep learning model that beats/competes and shows probabilistic forecast.

### Choose DeepAR (fast path)

Use a library that makes this painless:

* **GluonTS** (classic DeepAR)
* or **PyTorch Forecasting** (if you already know it)

**Model setup**

* Inputs: past `y_obs`, known covariates (price/calendar), static categories (item/store)
* Output: distribution for next horizon (e.g., 7 or 28 days)

**Key CIV twist: mask censored points**
During training, for time steps where `is_censored=1`:

* don’t compute loss (or weight it ~0)
  So the model learns demand from reliable in-stock points.

If masking is hard in the library:

* simpler workaround: **drop censored steps** from training windows (still acceptable; just document it)

**Probabilistic outputs**

* produce p50/p90 for horizon
* show 1–2 example series plots with prediction intervals

✅ End of Day 3: You can say “I used probabilistic DL forecasting and handled censoring via masked loss.”

---

# Day 4 — “Value of Instock” layer + repo polish (interview package)

**Deliverable:** README + 1 notebook + a crisp story.

## 4A) Define “instock improvement” scenario

Pick one lever:

* “Reduce OOS probability by 20% for top SKUs” (or set `oos=0` for those periods)

For each item-store:

* Predict demand under status quo (`oos` as observed)
* Predict demand under improved instock (counterfactual `oos=0` for future)

Incremental units:

* `Δunits = demand_cf - demand_base`
  Value:
* `Δvalue = Δunits * margin` (use a fixed margin like 20% and state it)

## 4B) Rank opportunities

Create a table:

* top 50 item-store by Δvalue
* include uncertainty band (p10/p90) from DeepAR samples

## 4C) Write the README like an Amazon doc

Suggested sections:

1. **Problem (CIV framing)**: sales are censored under stockouts; want value of improving instock/speed
2. **Data**: M5 + synthetic OOS
3. **Method**

   * rolling backtest
   * baselines
   * DeepAR probabilistic
   * censoring-aware training (mask)
4. **Results**

   * baseline vs DL metrics
   * the “bias demo” figure
5. **Decision layer**

   * value ranking
   * uncertainty
6. **Next steps**

   * censored likelihood (Tobit-style)
   * hierarchical reconciliation
   * incorporate true inventory signals (Amazon internal)

✅ End of Day 4: You have a complete “mini CIV system”: measurement + model + decision.

---

## What you’ll say to the hiring manager (30-second pitch)

“I built a small CIV-style prototype: demand forecasting under stockout censoring. I simulated OOS to demonstrate how naive evaluation and training are biased, used rolling backtests, and trained a probabilistic DeepAR model with censored points masked in the loss. Then I turned predictions into a value-of-instock ranking with uncertainty bands to prioritize which items benefit most from improving availability.”

---

## If you want, I can also give you:

* a repo skeleton (folders + notebook names)
* the exact plots to include (3 figures max)
* and 5 interview Q&As that map directly to this project

Just tell me: **Python stack preference** (GluonTS vs PyTorch Forecasting) and whether you want **7-day** or **28-day** horizon.
