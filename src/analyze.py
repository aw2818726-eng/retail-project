"""Clean the retail data, compute KPIs / RFM / cohorts / churn model, export dashboard data.

Run:  python src/analyze.py
Reads  data/*.csv
Writes outputs/*.csv, outputs/metrics.json, dashboard/data.js
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT, DASH = ROOT / "data", ROOT / "outputs", ROOT / "dashboard"
SEED = 42

CITY_MAP = {
    "القاهرة": "Cairo", "الجيزة": "Giza", "الإسكندرية": "Alexandria", "alex": "Alexandria",
}
BIG_CITIES = {"Cairo", "Giza", "Alexandria"}


# ---------------------------------------------------------------- cleaning
def load_and_clean():
    customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv(DATA / "products.csv")
    orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
    items = pd.read_csv(DATA / "order_items.csv")

    dq = {}
    dq["duplicate_orders_removed"] = int(orders.duplicated("order_id").sum())
    orders = orders.drop_duplicates("order_id").copy()

    dq["missing_payment_method"] = int(orders.payment_method.isna().sum())
    orders["payment_method"] = orders.payment_method.fillna("Unknown")

    raw_city = customers.city.copy()
    key = customers.city.str.strip().str.lower()
    customers["city"] = key.map(lambda c: CITY_MAP.get(c, c))
    customers["city"] = customers.city.map(lambda c: c.title() if isinstance(c, str) else c)
    customers["city"] = customers.city.replace({"Alex": "Alexandria"})
    dq["missing_city"] = int(customers.city.isna().sum())
    dq["city_values_before"] = int(raw_city.nunique())
    # customers with no city cannot be placed in any city view, so they (and their orders) are dropped
    customers = customers[customers.city.notna()].copy()
    dq["city_values_after"] = int(customers.city.nunique())
    dq["orders_removed_missing_city"] = int((~orders.customer_id.isin(customers.customer_id)).sum())
    orders = orders[orders.customer_id.isin(customers.customer_id)].copy()

    # items must belong to a known order
    dq["orphan_items_removed"] = int((~items.order_id.isin(orders.order_id)).sum())
    items = items[items.order_id.isin(orders.order_id)]

    lines = (items.merge(orders, on="order_id")
                  .merge(products[["product_id", "product_name", "category", "base_price", "cost_price"]],
                         on="product_id")
                  .merge(customers[["customer_id", "city", "acquisition_channel", "signup_date"]],
                         on="customer_id"))
    lines["revenue"] = lines.quantity * lines.unit_price * (1 - lines.discount_pct / 100)
    cost_ratio = lines.cost_price / lines.base_price  # cost scales with the price level
    lines["gross_profit"] = lines.revenue - lines.quantity * lines.unit_price * cost_ratio
    lines["month"] = lines.order_date.dt.to_period("M").astype(str)
    return customers, products, orders, lines, dq


# ---------------------------------------------------------------- KPIs
def kpis(orders, delivered):
    n_orders = delivered.order_id.nunique()
    per_cust = delivered.groupby("customer_id").order_id.nunique()
    return dict(
        net_revenue=round(float(delivered.revenue.sum())),
        delivered_orders=int(n_orders),
        active_customers=int(per_cust.size),
        avg_order_value=round(float(delivered.revenue.sum() / n_orders)),
        gross_margin_pct=round(float(delivered.gross_profit.sum() / delivered.revenue.sum() * 100), 1),
        repeat_customer_rate_pct=round(float((per_cust >= 2).mean() * 100), 1),
        cancel_rate_pct=round(float((orders.order_status == "Cancelled").mean() * 100), 1),
        return_rate_pct=round(float((orders.order_status == "Returned").mean() * 100), 1),
    )


def monthly_table(delivered):
    first_month = delivered.groupby("customer_id").month.min().rename("first_month")
    d = delivered.join(first_month, on="customer_id")
    d["is_new"] = d.month == d.first_month
    m = d.groupby("month").agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"),
                               profit=("gross_profit", "sum")).reset_index()
    m["aov"] = m.revenue / m.orders
    nv = d.groupby(["month", "is_new"]).revenue.sum().unstack(fill_value=0)
    m["new_customer_revenue"] = m.month.map(nv.get(True, 0))
    m["returning_customer_revenue"] = m.month.map(nv.get(False, 0))
    return m


# ---------------------------------------------------------------- RFM
def rfm(delivered, snapshot):
    g = delivered.groupby("customer_id")
    t = pd.DataFrame(dict(
        recency=(snapshot - g.order_date.max()).dt.days,
        frequency=g.order_id.nunique(),
        monetary=g.revenue.sum()))
    t["r"] = pd.qcut(t.recency.rank(method="first", ascending=False), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    t["f"] = pd.qcut(t.frequency.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    t["m"] = pd.qcut(t.monetary.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    def segment(r):
        if r.r >= 4 and r.f >= 4:
            return "Champions"
        if r.r >= 3 and r.f >= 3:
            return "Loyal"
        if r.r >= 4 and r.f <= 2:
            return "New / Promising"
        if r.r <= 2 and r.f >= 3:
            return "At Risk"
        if r.r <= 2 and r.f <= 2:
            return "Hibernating"
        return "Needs Attention"

    t["segment"] = t.apply(segment, axis=1)
    seg = t.groupby("segment").agg(customers=("recency", "size"), revenue=("monetary", "sum"),
                                   avg_recency_days=("recency", "mean"),
                                   avg_orders=("frequency", "mean")).reset_index()
    seg["customer_share_pct"] = seg.customers / seg.customers.sum() * 100
    seg["revenue_share_pct"] = seg.revenue / seg.revenue.sum() * 100
    return t, seg.sort_values("revenue", ascending=False)


# ---------------------------------------------------------------- cohorts
def cohorts(delivered, customers, start="2025-01-01"):
    """Retention by first-order month; only customers who signed up inside the window."""
    new_ids = customers.loc[customers.signup_date >= start, "customer_id"]
    d = delivered[delivered.customer_id.isin(new_ids)].copy()
    d["period"] = d.order_date.dt.to_period("M")
    first = d.groupby("customer_id").period.min().rename("cohort")
    d = d.join(first, on="customer_id")
    d["k"] = (d.period - d.cohort).apply(lambda x: x.n)
    active = d.groupby(["cohort", "k"]).customer_id.nunique().unstack(fill_value=0)
    size = active[0]
    last_period = delivered.order_date.max().to_period("M")
    rows = []
    for cohort, n in size.items():
        max_k = (last_period - cohort).n
        rows.append(dict(cohort=str(cohort), size=int(n),
                         retention=[round(float(active.loc[cohort].get(k, 0) / n * 100), 1)
                                    if k <= max_k else None for k in range(0, 12)]))
    return rows


# ---------------------------------------------------------------- channels
def channel_table(delivered, customers, orders):
    g = delivered.groupby("customer_id").agg(orders=("order_id", "nunique"), revenue=("revenue", "sum"))
    c = customers.set_index("customer_id").join(g).fillna({"orders": 0, "revenue": 0})
    c = c[c.signup_date >= "2025-01-01"]
    out = c.groupby("acquisition_channel").agg(
        customers=("orders", "size"),
        buyers=("orders", lambda s: int((s > 0).sum())),
        repeat_buyers=("orders", lambda s: int((s >= 2).sum())),
        revenue=("revenue", "sum")).reset_index()
    out["conversion_pct"] = out.buyers / out.customers * 100
    out["repeat_rate_pct"] = out.repeat_buyers / out.buyers * 100
    out["revenue_per_customer"] = out.revenue / out.customers
    return out


# ---------------------------------------------------------------- churn model
def churn_model(orders, delivered, customers, cutoff="2026-07-31", horizon_end="2026-10-31"):
    """Predict whether a customer who bought before `cutoff` buys again by `horizon_end`.

    Features use only information up to the cutoff (no leakage); the label is observed after it.
    """
    cutoff, horizon_end = pd.Timestamp(cutoff), pd.Timestamp(horizon_end)
    past = delivered[delivered.order_date <= cutoff]
    future = delivered[(delivered.order_date > cutoff) & (delivered.order_date <= horizon_end)]
    o_past = orders[(orders.order_date <= cutoff)]

    per_order = past.groupby(["customer_id", "order_id", "order_date"]).agg(
        value=("revenue", "sum"), disc=("discount_pct", "mean")).reset_index()
    g = per_order.groupby("customer_id")
    X = pd.DataFrame(dict(
        recency_days=(cutoff - g.order_date.max()).dt.days,
        frequency=g.order_id.nunique(),
        monetary=g.value.sum(),
        avg_order_value=g.value.mean(),
        tenure_days=(cutoff - g.order_date.min()).dt.days,
        avg_discount_pct=g.disc.mean()))
    X["orders_last_90d"] = per_order[per_order.order_date > cutoff - pd.Timedelta(days=90)] \
        .groupby("customer_id").order_id.nunique().reindex(X.index).fillna(0)
    gap = per_order.sort_values("order_date").groupby("customer_id").order_date.apply(
        lambda s: s.diff().dt.days.mean())
    X["avg_days_between_orders"] = gap.reindex(X.index).fillna(X.tenure_days.clip(lower=1) + 30)
    status = o_past.groupby("customer_id").order_status.agg(
        cancel_rate=lambda s: (s == "Cancelled").mean(), return_rate=lambda s: (s == "Returned").mean())
    X = X.join(status).fillna(0)
    cod = o_past.assign(cod=o_past.payment_method == "Cash on Delivery").groupby("customer_id").cod.mean()
    X["cod_share"] = cod.reindex(X.index).fillna(0)
    cu = customers.set_index("customer_id")
    X["big_city"] = cu.city.reindex(X.index).isin(BIG_CITIES).astype(int)
    X = X.join(pd.get_dummies(cu.acquisition_channel.reindex(X.index), prefix="channel").astype(int))

    y = (~X.index.isin(future.customer_id.unique())).astype(int)  # 1 = churned
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Gradient boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08,
                                                            max_iter=200, random_state=SEED),
    }
    results = []
    for name, model in models.items():
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        results.append(dict(model=name, roc_auc=round(roc_auc_score(yte, p), 3),
                            pr_auc=round(average_precision_score(yte, p), 3),
                            precision=round(precision_score(yte, pred), 3),
                            recall=round(recall_score(yte, pred), 3)))
    # naive baseline: the longer since last order, the likelier to churn
    base = Xte.recency_days
    results.insert(0, dict(model="Baseline (recency only)", roc_auc=round(roc_auc_score(yte, base), 3),
                           pr_auc=round(average_precision_score(yte, base), 3),
                           precision=None, recall=None))

    gb = models["Gradient boosting"]
    pi = permutation_importance(gb, Xte, yte, scoring="roc_auc", n_repeats=15, random_state=SEED)
    imp = (pd.DataFrame(dict(feature=X.columns, importance=pi.importances_mean))
           .sort_values("importance", ascending=False).head(8))
    return dict(cutoff=str(cutoff.date()), horizon_end=str(horizon_end.date()),
                customers_scored=int(len(X)), churn_rate_pct=round(float(y.mean() * 100), 1),
                test_size=int(len(Xte)), results=results,
                feature_importance=[dict(feature=r.feature, importance=round(float(r.importance), 4))
                                    for r in imp.itertuples()])


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    DASH.mkdir(exist_ok=True)
    customers, products, orders, lines, dq = load_and_clean()
    delivered = lines[lines.order_status == "Delivered"]
    snapshot = orders.order_date.max() + pd.Timedelta(days=1)

    overall = kpis(orders, delivered)
    monthly = monthly_table(delivered)
    rfm_table, rfm_seg = rfm(delivered, snapshot)
    cohort_rows = cohorts(delivered, customers)
    channels = channel_table(delivered, customers, orders)
    model = churn_model(orders, delivered, customers)

    cat = delivered.groupby("category").agg(revenue=("revenue", "sum"), profit=("gross_profit", "sum"),
                                            units=("quantity", "sum")).reset_index()
    cat["margin_pct"] = cat.profit / cat.revenue * 100
    cancel = orders.groupby("payment_method").order_status.agg(
        orders="size", cancel_rate_pct=lambda s: (s == "Cancelled").mean() * 100,
        return_rate_pct=lambda s: (s == "Returned").mean() * 100).reset_index()
    city = delivered.groupby("city").agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"),
                                         customers=("customer_id", "nunique")).reset_index()
    city["revenue_per_customer"] = city.revenue / city.customers

    # tables for the repo
    monthly.round(2).to_csv(OUT / "monthly_summary.csv", index=False)
    rfm_seg.round(2).to_csv(OUT / "rfm_segments.csv", index=False)
    rfm_table.round(2).to_csv(OUT / "rfm_customers.csv")
    channels.round(2).to_csv(OUT / "channel_performance.csv", index=False)
    cat.round(2).to_csv(OUT / "category_performance.csv", index=False)
    city.round(2).to_csv(OUT / "city_performance.csv", index=False)
    cancel.round(2).to_csv(OUT / "cancellation_by_payment.csv", index=False)

    metrics = dict(data_quality=dq, kpis=overall, model=model,
                   period=dict(start=str(orders.order_date.min().date()), end=str(orders.order_date.max().date())))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- dashboard payload (aggregated; no customer-level data)
    facts = (delivered.groupby(["month", "city", "category"])
             .agg(revenue=("revenue", "sum"), profit=("gross_profit", "sum"),
                  units=("quantity", "sum"), orders=("order_id", "nunique")).reset_index())
    orders_mc = (delivered.groupby(["month", "city"])
                 .agg(revenue=("revenue", "sum"), orders=("order_id", "nunique")).reset_index())
    r0 = lambda df: df.round(0).to_dict("records")
    payload = dict(
        meta=dict(synthetic=True, start=metrics["period"]["start"], end=metrics["period"]["end"],
                  customers=int(len(customers)), orders=int(len(orders)), currency="EGP"),
        kpis=overall,
        facts=json.loads(facts.round(0).to_json(orient="records")),
        orders_mc=json.loads(orders_mc.round(0).to_json(orient="records")),
        monthly=json.loads(monthly.round(0).to_json(orient="records")),
        rfm=json.loads(rfm_seg.round(1).to_json(orient="records")),
        cohorts=cohort_rows,
        channels=json.loads(channels.round(1).to_json(orient="records")),
        cancellation=json.loads(cancel.round(1).to_json(orient="records")),
        model=model, data_quality=dq)
    (DASH / "data.js").write_text("window.DASHBOARD_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n")

    # ---- console summary
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("DATA QUALITY", dq)
    print("KPIS", overall)
    print("\nMONTHLY\n", monthly.round(0).to_string(index=False))
    print("\nCATEGORY\n", cat.round(1).sort_values("revenue", ascending=False).to_string(index=False))
    print("\nCITY\n", city.round(0).sort_values("revenue", ascending=False).to_string(index=False))
    print("\nRFM SEGMENTS\n", rfm_seg.round(1).to_string(index=False))
    print("\nCHANNELS\n", channels.round(1).to_string(index=False))
    print("\nCANCELLATION\n", cancel.round(1).to_string(index=False))
    print("\nMODEL", json.dumps(model, indent=1))
    print("\nCOHORTS (first 4)", cohort_rows[:4])


if __name__ == "__main__":
    main()
