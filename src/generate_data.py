"""Generate a synthetic Egyptian e-commerce dataset.

Why synthetic: real retail transaction data is private. This generator produces
four relational tables (customers, products, orders, order_items) with realistic
structure, Egyptian seasonality (Ramadan, Eid, White Friday), customer churn,
and deliberately messy fields so the cleaning step has real work to do.

Run:  python src/generate_data.py
Output: data/customers.csv, data/products.csv, data/orders.csv, data/order_items.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2026-10-31")
N_CUSTOMERS = 6000

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CITIES = {
    "Cairo": 0.34, "Giza": 0.17, "Alexandria": 0.16, "Mansoura": 0.05,
    "Tanta": 0.04, "Assiut": 0.04, "Port Said": 0.03, "Suez": 0.03,
    "Luxor": 0.03, "Zagazig": 0.03, "Ismailia": 0.03, "Aswan": 0.05,
}
SHIPPING_FEE = {"Cairo": 40, "Giza": 40, "Alexandria": 55}
DEFAULT_SHIPPING_FEE = 75
FREE_SHIPPING_THRESHOLD = 1500

CHANNELS = {  # share, avg lifetime (months), order-rate multiplier
    "Organic": (0.35, 9, 1.0),
    "Social": (0.30, 6, 0.9),
    "Paid Ads": (0.20, 5, 0.85),
    "Referral": (0.15, 12, 1.25),
}

PAYMENT_METHODS = {
    "Cash on Delivery": 0.55, "Credit Card": 0.25, "Mobile Wallet": 0.15, "InstaPay": 0.05,
}

CATEGORY_SPEC = {
    "Electronics": dict(price=(800, 25000), margin=(0.08, 0.18), names=[
        "Wireless Earbuds", "Bluetooth Speaker", "Smart Watch", "Power Bank", "Phone Case",
        "USB-C Charger", "LED Monitor", "Laptop Stand", "Mechanical Keyboard", "Webcam",
        "Wi-Fi Router", "Tablet", "Gaming Mouse", "Smart Bulb"]),
    "Fashion": dict(price=(250, 2500), margin=(0.35, 0.55), names=[
        "Cotton T-Shirt", "Denim Jeans", "Hooded Sweatshirt", "Summer Dress", "Abaya",
        "Leather Belt", "Sneakers", "Sandals", "Backpack", "Handbag", "Scarf", "Sunglasses",
        "Polo Shirt", "Jacket", "Socks 5-Pack", "Classic Watch", "Wallet", "Cap"]),
    "Home & Kitchen": dict(price=(150, 5000), margin=(0.22, 0.38), names=[
        "Non-stick Pan Set", "Blender", "Air Fryer", "Bed Sheet Set", "Towel Set",
        "Food Container Set", "Electric Kettle", "Dinner Set", "Curtains", "Floor Lamp",
        "Storage Boxes", "Pillow Pair", "Cutlery Set", "Coffee Maker", "Steam Iron", "Wall Clock"]),
    "Beauty": dict(price=(80, 1500), margin=(0.40, 0.60), names=[
        "Face Wash", "Moisturizer", "Sunscreen SPF50", "Hair Oil", "Shampoo", "Perfume 100ml",
        "Lipstick", "Foundation", "Eyeliner", "Body Lotion", "Hair Dryer", "Beard Trimmer"]),
    "Groceries": dict(price=(30, 600), margin=(0.10, 0.20), names=[
        "Olive Oil 1L", "Basmati Rice 5kg", "Pasta Pack", "Tea Box", "Coffee Beans 500g",
        "Honey Jar", "Dates 1kg", "Mixed Nuts", "Cooking Oil 2L", "Sugar 5kg",
        "Canned Tuna 6-Pack", "Cheese Pack"]),
    "Sports": dict(price=(150, 4000), margin=(0.25, 0.40), names=[
        "Yoga Mat", "Dumbbell Set", "Football", "Running Shoes", "Resistance Bands",
        "Water Bottle", "Gym Bag", "Jump Rope"]),
}
CATS = list(CATEGORY_SPEC)
CAT_BASE_ALPHA = np.array([1.2, 1.5, 1.2, 1.0, 1.3, 0.6])
MONTHLY_PRICE_INFLATION = 0.012


def build_products(rng):
    rows, pid = [], 1
    for cat, spec in CATEGORY_SPEC.items():
        lo, hi = spec["price"]
        for name in spec["names"]:
            price = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            price = max(5, round(price / 5) * 5)
            margin = rng.uniform(*spec["margin"])
            rows.append(dict(product_id=f"P{pid:03d}", product_name=name, category=cat,
                             base_price=price, cost_price=round(price * (1 - margin), 2)))
            pid += 1
    return pd.DataFrame(rows)


def build_calendar(days):
    """Per-day demand multiplier and per-day category multipliers."""
    n = len(days)
    overall = np.ones(n)
    cat_mult = np.ones((n, len(CATS)))
    ix = {c: i for i, c in enumerate(CATS)}

    def span(a, b):
        return (days >= pd.Timestamp(a)) & (days <= pd.Timestamp(b))

    ramadan = span("2025-03-01", "2025-03-29") | span("2026-02-18", "2026-03-19")
    overall[ramadan] *= 1.15
    cat_mult[ramadan, ix["Groceries"]] *= 2.5
    cat_mult[ramadan, ix["Home & Kitchen"]] *= 1.3

    # approximate pre-Eid shopping windows
    eid = (span("2025-03-20", "2025-03-29") | span("2025-05-27", "2025-06-05")
           | span("2026-03-10", "2026-03-19") | span("2026-05-17", "2026-05-26"))
    overall[eid] *= 1.2
    cat_mult[eid, ix["Fashion"]] *= 1.8
    cat_mult[eid, ix["Beauty"]] *= 1.4

    sep = days.month == 9  # back to school
    overall[sep] *= 1.15
    cat_mult[sep, ix["Fashion"]] *= 1.3

    white_friday = span("2025-11-21", "2025-11-30")
    overall[white_friday] *= 2.4
    cat_mult[white_friday, ix["Electronics"]] *= 1.8
    cat_mult[white_friday, ix["Home & Kitchen"]] *= 1.3
    overall[span("2025-11-01", "2025-11-20")] *= 1.1
    overall[days.month == 12] *= 1.1

    overall[days.dayofweek == 4] *= 1.2  # Friday
    overall[days.dayofweek == 5] *= 1.1  # Saturday
    return overall, cat_mult, white_friday


def build_customers(rng):
    cities = rng.choice(list(CITIES), N_CUSTOMERS, p=np.array(list(CITIES.values())) / sum(CITIES.values()))
    ch_names = list(CHANNELS)
    ch_p = np.array([CHANNELS[c][0] for c in ch_names])
    channels = rng.choice(ch_names, N_CUSTOMERS, p=ch_p / ch_p.sum())

    span_days = (pd.Timestamp("2026-09-30") - START).days
    n_old = 1000
    old = pd.Timestamp("2024-07-01") + pd.to_timedelta(rng.integers(0, 184, n_old), unit="D")
    new_offsets = (rng.beta(1.3, 1.0, N_CUSTOMERS - n_old) * span_days).astype(int)
    new = START + pd.to_timedelta(new_offsets, unit="D")
    signup = pd.Series(list(old) + list(new)).sample(frac=1, random_state=SEED).reset_index(drop=True)

    return pd.DataFrame(dict(
        customer_id=[f"C{i:05d}" for i in range(1, N_CUSTOMERS + 1)],
        signup_date=signup.dt.date, city=cities, acquisition_channel=channels))


def simulate_orders(rng, customers, products, overall, cat_mult, white_friday):
    days = pd.date_range(START, END, freq="D")
    n_days = len(days)
    max_season = overall.max()

    prod_by_cat = {}
    for ci, cat in enumerate(CATS):
        sub = products[products.category == cat].reset_index(drop=True)
        w = 1.0 / np.arange(1, len(sub) + 1) ** 0.8
        w = w[rng.permutation(len(sub))]
        prod_by_cat[ci] = (sub, w / w.sum())

    raw_orders, raw_items = [], []
    for row in customers.itertuples(index=False):
        signup = pd.Timestamp(row.signup_date)
        _, life_months, rate_mult = CHANNELS[row.acquisition_channel]
        lam = rng.gamma(1.5, 0.20) * rate_mult  # orders per month at baseline demand
        lifetime_days = rng.exponential(life_months * 30)
        start_ix = max((signup - START).days, 0)
        end_ix = min(n_days - 1, int((signup + pd.Timedelta(days=lifetime_days) - START).days))
        if end_ix < start_ix:
            continue

        order_ix = []
        if signup >= START and rng.random() < 0.93:  # first order shortly after signup
            first = start_ix + int(rng.geometric(0.35)) - 1
            if first <= n_days - 1:
                order_ix.append(first)
        n_cand = rng.poisson(lam * (end_ix - start_ix + 1) / 30 * max_season)
        if n_cand:
            cand = rng.integers(start_ix, end_ix + 1, n_cand)
            order_ix += list(cand[rng.random(n_cand) < overall[cand] / max_season])

        cust_w = rng.dirichlet(CAT_BASE_ALPHA)
        for d in order_ix:
            n_lines = min(5, 1 + rng.poisson(0.6))
            w = cust_w * cat_mult[d]
            w = w / w.sum()
            lines, has_fashion = [], False
            for _ in range(n_lines):
                ci = rng.choice(len(CATS), p=w)
                sub, pw = prod_by_cat[ci]
                prod = sub.iloc[rng.choice(len(sub), p=pw)]
                months_elapsed = (days[d].year - START.year) * 12 + days[d].month - START.month
                price = round(prod.base_price * (1 + MONTHLY_PRICE_INFLATION) ** months_elapsed
                              * rng.normal(1, 0.02))
                if CATS[ci] == "Groceries":
                    qty = 1 + rng.poisson(1.0)
                elif CATS[ci] == "Electronics":
                    qty = 1
                else:
                    qty = 1 + int(rng.random() < 0.12)
                if white_friday[d] and rng.random() < 0.85:
                    disc = rng.choice([20, 30, 40, 50])
                else:
                    disc = rng.choice([0, 5, 10, 15, 20], p=[0.72, 0.10, 0.10, 0.05, 0.03])
                lines.append((prod.product_id, int(qty), int(price), int(disc)))
                has_fashion |= CATS[ci] == "Fashion"

            subtotal = sum(q * p * (1 - dc / 100) for _, q, p, dc in lines)
            fee = 0 if subtotal >= FREE_SHIPPING_THRESHOLD else SHIPPING_FEE.get(row.city, DEFAULT_SHIPPING_FEE)
            pay = rng.choice(list(PAYMENT_METHODS), p=list(PAYMENT_METHODS.values()))
            p_cancel = 0.09 if pay == "Cash on Delivery" else 0.04
            u = rng.random()
            if u < p_cancel:
                status = "Cancelled"
            elif u < p_cancel + (0.075 if has_fashion else 0.035):
                status = "Returned"
            else:
                status = "Delivered"
            raw_orders.append(dict(day=d, customer_id=row.customer_id, payment_method=pay,
                                   order_status=status, shipping_fee=fee, lines=lines))

    raw_orders.sort(key=lambda o: (o["day"], o["customer_id"]))
    orders, items = [], []
    for i, o in enumerate(raw_orders, start=1):
        oid = f"ORD-{i:06d}"
        orders.append(dict(order_id=oid, customer_id=o["customer_id"],
                           order_date=days[o["day"]].date(), payment_method=o["payment_method"],
                           order_status=o["order_status"], shipping_fee=o["shipping_fee"]))
        for pid, qty, price, disc in o["lines"]:
            items.append(dict(order_id=oid, product_id=pid, quantity=qty,
                              unit_price=price, discount_pct=disc))
    return pd.DataFrame(orders), pd.DataFrame(items)


def add_mess(rng, customers, orders):
    """Inject realistic data-quality problems for the cleaning step to fix."""
    customers = customers.copy()
    orders = orders.copy()

    variants = {
        "Cairo": ["cairo", "CAIRO", "Cairo ", "القاهرة"],
        "Giza": ["giza", "Giza ", "الجيزة"],
        "Alexandria": ["alexandria", "Alex", "الإسكندرية", "ALEXANDRIA"],
    }
    mask = rng.random(len(customers)) < 0.08
    for i in customers.index[mask]:
        city = customers.at[i, "city"]
        if city in variants:
            customers.at[i, "city"] = rng.choice(variants[city])
        else:
            customers.at[i, "city"] = city.lower() + " "
    customers.loc[rng.random(len(customers)) < 0.015, "city"] = np.nan

    orders.loc[rng.random(len(orders)) < 0.02, "payment_method"] = np.nan
    dupes = orders.sample(frac=0.004, random_state=SEED)
    orders = pd.concat([orders, dupes], ignore_index=True).sort_values("order_id", kind="stable")
    return customers, orders.reset_index(drop=True)


def main():
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(exist_ok=True)
    days = pd.date_range(START, END, freq="D")
    overall, cat_mult, white_friday = build_calendar(days)

    products = build_products(rng)
    customers = build_customers(rng)
    orders, items = simulate_orders(rng, customers, products, overall, cat_mult, white_friday)
    customers, orders = add_mess(rng, customers, orders)

    customers.to_csv(DATA_DIR / "customers.csv", index=False)
    products.to_csv(DATA_DIR / "products.csv", index=False)
    orders.to_csv(DATA_DIR / "orders.csv", index=False)
    items.to_csv(DATA_DIR / "order_items.csv", index=False)
    print(f"customers={len(customers):,} products={len(products)} "
          f"orders={len(orders):,} order_items={len(items):,}")


if __name__ == "__main__":
    main()
