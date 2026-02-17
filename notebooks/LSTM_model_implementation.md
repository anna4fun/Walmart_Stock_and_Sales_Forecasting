Awesome — here’s a **minimal PyTorch Forecasting template** for a **7-day multi-horizon LSTM quantile forecaster** plus:

* rolling backtest scaffold (simple, practical)
* **in-stock-only evaluation**
* **Value of Instock** counterfactual ranking

It’s written to be dropped into `notebooks/03_deepar_masked_loss_and_value_of_instock.ipynb` (even though it’s LSTM, not DeepAR). No fancy extras.

---

## Core idea (how we handle OOS/censoring here)

PyTorch Forecasting doesn’t natively “mask individual time steps” inside a sequence loss in one line, so the simplest reliable approach in 4 days is:

1. **Train on in-stock points only** (drop censored targets from the training windows), and
2. still keep an `is_oos` feature for inference/scenario simulation, and
3. Evaluate on **in-stock-only** days (your “true demand proxy”).

This is defendable in interviews, and you can say: “next iteration is censored likelihood / masked loss.”

---

## Code template (single notebook cell blocks)

```python
# =========
# Setup
# =========
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import RecurrentNetwork
from pytorch_forecasting.metrics import QuantileLoss

# Repro
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)
```

```python
# =========================
# Load your processed panel
# =========================
# Expected columns (minimum):
#   date (datetime or int time_idx)
#   time_idx (int, increasing)
#   item_id (str/int)
#   store_id (str/int)
#   y_obs (float)         # observed sales (censored)
#   y_true (float)        # optional: if you simulated; otherwise omit
#   is_oos (0/1)          # OOS indicator (1=censored)
#   price (float)         # optional
#   promo (0/1)           # optional
#   dow (0-6)             # optional
#
# If you don't have y_true, just evaluate on "is_oos==0" using y_obs as demand proxy.
df = pd.read_parquet("data/processed/panel.parquet")

# Ensure types
df["time_idx"] = df["time_idx"].astype(int)
df["is_oos"] = df["is_oos"].astype(int)

# group id for multi-series
df["series_id"] = df["item_id"].astype(str) + "||" + df["store_id"].astype(str)

df = df.sort_values(["series_id", "time_idx"]).reset_index(drop=True)
df.head()
```

```python
# =========================
# Rolling backtest split
# =========================
HORIZON = 7
ENC_LEN = 56        # ~8 weeks history (tunable)
VAL_LEN = 28        # validate on 4 weeks (tunable)

max_t = df["time_idx"].max()
# pick a single fold cut for simplicity; you can loop for multiple folds
cutoff = max_t - VAL_LEN - HORIZON

print("max_t:", max_t, "cutoff:", cutoff)

train_df = df[df["time_idx"] <= cutoff].copy()
val_df   = df[(df["time_idx"] > cutoff) & (df["time_idx"] <= cutoff + VAL_LEN)].copy()

# Train on in-stock only targets to avoid learning censored zeros as demand
train_df = train_df[train_df["is_oos"] == 0].copy()

print("train rows:", len(train_df), "val rows:", len(val_df))
```

```python
# =====================================
# Build TimeSeriesDataSet (PyTorch Fcst)
# =====================================
# Known future features: calendar, planned promo, etc.
# Unknown features include the target itself (y_obs) and other observed signals.

target_col = "y_obs"  # if you have y_true in simulation, still keep target as observed sales and train on in-stock only

# Feature lists (edit based on what you have)
time_varying_known_reals = []
time_varying_known_categoricals = []
time_varying_unknown_reals = [target_col]
time_varying_unknown_categoricals = []

# Optional known/unknown covariates
for col in ["price"]:
    if col in df.columns:
        time_varying_known_reals.append(col)

for col in ["promo", "dow", "is_oos"]:
    if col in df.columns:
        # promo/dow are known in future; is_oos is unknown in reality but for this project we treat it as scenario-controlled
        time_varying_known_reals.append(col)

static_categoricals = []
# We'll just use series_id as group id, but you can also add item_id/store_id as static cats.
# (pytorch-forecasting uses embeddings for categoricals)
for col in ["item_id", "store_id"]:
    if col in df.columns:
        static_categoricals.append(col)

training = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target=target_col,
    group_ids=["series_id"],
    max_encoder_length=ENC_LEN,
    max_prediction_length=HORIZON,
    static_categoricals=static_categoricals,
    time_varying_known_reals=time_varying_known_reals,
    time_varying_known_categoricals=time_varying_known_categoricals,
    time_varying_unknown_reals=time_varying_unknown_reals,
    time_varying_unknown_categoricals=time_varying_unknown_categoricals,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)

BATCH_SIZE = 128
train_loader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
val_loader   = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

print(training)
```

```python
# =========================
# Train LSTM (RecurrentNetwork)
# =========================
# Quantile forecasting: p10/p50/p90 is great for CIV-style uncertainty
quantiles = [0.1, 0.5, 0.9]
loss = QuantileLoss(quantiles=quantiles)

model = RecurrentNetwork.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=64,
    rnn_layers=2,
    dropout=0.1,
    loss=loss,
    log_interval=50,
    optimizer="Adam",
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, mode="min"),
    LearningRateMonitor(logging_interval="epoch"),
]

trainer = Trainer(
    max_epochs=10,
    accelerator="gpu" if DEVICE == "cuda" else "cpu",
    devices=1,
    gradient_clip_val=0.1,
    callbacks=callbacks,
    enable_checkpointing=True,
    log_every_n_steps=50,
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

```python
# =========================
# Predict on validation
# =========================
# Returns quantile predictions aligned with each prediction window
pred = model.predict(val_loader, mode="quantiles")  # shape: [n, horizon, n_quantiles]
pred.shape
```

```python
# ===========================================
# Build a tidy prediction table for evaluation
# ===========================================
# We'll use the dataset's built-in conversion helper to align predictions.
raw_predictions, x = model.predict(val_loader, mode="raw", return_x=True)

# raw_predictions["prediction"] is [batch, horizon, n_quantiles] for quantile loss
qpred = raw_predictions["prediction"].detach().cpu().numpy()

# get identifiers + time index for each prediction
index = x["decoder_time_idx"].detach().cpu().numpy()              # [batch, horizon]
series = x["groups"][0].detach().cpu().numpy()                    # encoded group; not human-readable
# We’ll recover series_id using x_to_index helper:
idx_df = validation.x_to_index(x)

# idx_df contains (time_idx, series_id) for each row of prediction horizon
# But it's repeated per horizon step; easiest is flatten everything:
out = idx_df.copy()
out["q10"] = qpred[..., 0].reshape(-1)
out["q50"] = qpred[..., 1].reshape(-1)
out["q90"] = qpred[..., 2].reshape(-1)

# Merge actuals + is_oos (from val_df)
out = out.merge(
    df[["series_id", "time_idx", "y_obs", "is_oos"]], 
    on=["series_id", "time_idx"], how="left"
)

out.head(), out.shape
```

```python
# =========================
# Metrics: WAPE (in-stock only)
# =========================
def wape(y_true, y_pred):
    denom = np.sum(np.abs(y_true)) + 1e-9
    return np.sum(np.abs(y_true - y_pred)) / denom

# Evaluate on in-stock only (proxy for true demand)
eval_df = out[out["is_oos"] == 0].copy()

wape_q50 = wape(eval_df["y_obs"].values, eval_df["q50"].values)
print("WAPE (q50, in-stock only):", wape_q50)
```

---

## Value of Instock (7-day counterfactual) with uncertainty

We’ll define a simple counterfactual: **future is_oos = 0** for the next 7 days for each series, and see how predicted demand changes. Since `is_oos` is in `time_varying_known_reals` above, we can override it for prediction windows.

```python
# ==========================================
# Counterfactual: set future is_oos = 0
# ==========================================
# Create a copy of val_df and force is_oos=0 on the prediction window
val_cf = val_df.copy()
val_cf["is_oos"] = 0

validation_cf = TimeSeriesDataSet.from_dataset(training, val_cf, predict=True, stop_randomization=True)
val_loader_cf = validation_cf.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

raw_predictions_cf, x_cf = model.predict(val_loader_cf, mode="raw", return_x=True)
qpred_cf = raw_predictions_cf["prediction"].detach().cpu().numpy()

idx_df_cf = validation_cf.x_to_index(x_cf)
out_cf = idx_df_cf.copy()
out_cf["q10_cf"] = qpred_cf[..., 0].reshape(-1)
out_cf["q50_cf"] = qpred_cf[..., 1].reshape(-1)
out_cf["q90_cf"] = qpred_cf[..., 2].reshape(-1)

# Join baseline prediction table on (series_id, time_idx)
merged = out.merge(out_cf, on=["series_id", "time_idx"], how="inner")

# Incremental demand per day
merged["d_q10"] = merged["q10_cf"] - merged["q10"]
merged["d_q50"] = merged["q50_cf"] - merged["q50"]
merged["d_q90"] = merged["q90_cf"] - merged["q90"]

# Aggregate to 7-day value per series
MARGIN = 0.2  # assumed contribution margin
agg = (merged.groupby("series_id")[["d_q10", "d_q50", "d_q90"]]
       .sum()
       .reset_index())

agg["value_q10"] = agg["d_q10"] * MARGIN
agg["value_q50"] = agg["d_q50"] * MARGIN
agg["value_q90"] = agg["d_q90"] * MARGIN

top = agg.sort_values("value_q50", ascending=False).head(20)
top.head(10)
```

```python
# Save ranking table
os.makedirs("reports/tables", exist_ok=True)
top.to_csv("reports/tables/top_value_of_instock.csv", index=False)
print("Saved: reports/tables/top_value_of_instock.csv")
```

---

## The 3 plots (exactly as promised)

### Plot 1: Censoring bias demo (all days vs in-stock-only WAPE)

```python
# compute WAPE on all days vs in-stock-only
wape_all = wape(out["y_obs"].fillna(0).values, out["q50"].fillna(0).values)
wape_instock = wape(eval_df["y_obs"].values, eval_df["q50"].values)

plt.figure()
plt.bar(["All days", "In-stock only"], [wape_all, wape_instock])
plt.ylabel("WAPE (q50)")
plt.title("Censoring Bias: Evaluation Regimes")
plt.tight_layout()
os.makedirs("reports/figures", exist_ok=True)
plt.savefig("reports/figures/fig1_censoring_bias_wape.png", dpi=200)
plt.show()
```

### Plot 2: Forecast example with uncertainty (pick one series)

```python
example_series = top["series_id"].iloc[0]
ex = merged[merged["series_id"] == example_series].sort_values("time_idx")

plt.figure()
plt.plot(ex["time_idx"], ex["y_obs"], label="Observed sales (censored)")
plt.plot(ex["time_idx"], ex["q50"], label="Forecast q50 (baseline)")
plt.fill_between(ex["time_idx"], ex["q10"], ex["q90"], alpha=0.2, label="q10-q90 (baseline)")
plt.plot(ex["time_idx"], ex["q50_cf"], label="Forecast q50 (CF instock)")
plt.fill_between(ex["time_idx"], ex["q10_cf"], ex["q90_cf"], alpha=0.2, label="q10-q90 (CF instock)")
plt.title(f"Example series: {example_series} (7-day horizon points)")
plt.xlabel("time_idx")
plt.ylabel("units")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/fig2_example_uncertainty_counterfactual.png", dpi=200)
plt.show()
```

### Plot 3: Top-K Value of Instock with uncertainty bands

```python
plot_df = top.copy()
plot_df = plot_df.sort_values("value_q50", ascending=True)

plt.figure(figsize=(8, 6))
plt.barh(plot_df["series_id"], plot_df["value_q50"])
plt.xlabel("Δvalue (q50) over 7 days (margin-scaled)")
plt.title("Top Value of Instock Opportunities (with uncertainty available in table)")
plt.tight_layout()
plt.savefig("reports/figures/fig3_top_value_bar.png", dpi=200)
plt.show()
```

---

## What to say if asked “Where is the masked loss?”

Be honest + sharp:

> “In this fast prototype, I avoided teaching the model wrong labels by **training on in-stock targets only** and evaluating on in-stock-only as a proxy for true demand. The next step is implementing a **censored likelihood** (Tobit-style) or explicit time-step weighting inside the loss so OOS points contribute ‘demand ≥ observed’ rather than being dropped.”

That answer scores well because it shows you understand the *real* statistical issue and know the principled fix.

---

If you want, I can also paste a **ready-to-use `requirements.txt`** (pinned versions that won’t conflict) and a **README snippet** describing why `is_oos` is treated as a scenario-controlled covariate for the counterfactual.
