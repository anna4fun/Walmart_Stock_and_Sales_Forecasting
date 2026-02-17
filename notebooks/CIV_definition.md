In CIV (Consumer Instock Value), **censoring** means: you don’t get to observe the “true” outcome you care about, because the system **caps** what can happen.

### The concrete CIV version

What you *want* is **true demand**:

> “How many units customers would have bought if the item were available with a normal delivery promise?”

What you *observe* in logs is usually **sales / shipped units** (or conversions).
But **sales are limited by availability and promise**. So when an item is out of stock (or effectively unbuyable / too-slow promise), **sales are a lower bound** on demand.

That is censoring: **the observation is truncated/capped by a constraint**.

### A simple example

* True demand that day: **100** customers want it.
* Inventory available: **30** units.
* Observed sales: **30**.

You will never see “100” in the data for that day, because the system couldn’t sell more than 30. The outcome is **censored at 30**.

Same idea if the offer is suppressed or the delivery promise becomes terrible:

* demand might still be high,
* but observed purchases drop because the system makes buying impossible or unattractive.

### Why it creates “censoring bias”

If you train a model on observed sales directly:

* the model learns “when inventory is low, demand is low,”
* but that’s backwards: inventory is low **causing** low sales, not revealing low demand.

And if you evaluate forecasts on days with OOS:

* predicting 0 can look “accurate,”
* even though the true demand could have been high.

That’s what I mean by **“censoring bias”**: your data makes demand look smaller than it really is in exactly the cases CIV cares most about.

### How CIV teams typically handle it (conceptually)

They try to reconstruct / approximate demand using signals like:

* in-stock vs out-of-stock flags (buyable status)
* inventory snapshots
* delivery promise speed / availability
* substitution behavior (customers buy a similar item)
* backorders / delayed purchases (demand shifts in time)

And method-wise, common patterns are:

* train/evaluate on “in-stock” periods to learn baseline demand
* model demand with availability as a constraint (censored models)
* use causal methods to estimate “what if instock improved?”

### One sentence you can use in an interview

> “In CIV, sales are a censored observation of demand — when items aren’t buyable or have poor promise, observed purchases are capped, so naïvely modeling sales biases demand down exactly when availability is constrained.”

If you want, I can draw a tiny diagram (variables and arrows) showing *demand → sales* with *availability* as the censoring gate, because that picture tends to make it click instantly.
