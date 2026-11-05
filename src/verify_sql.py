"""Check that the BigQuery SQL in sql/ returns the same numbers as the pandas pipeline.

The BigQuery statements are transpiled to DuckDB with sqlglot and run on the same CSV files,
then compared with the tables written by src/analyze.py. No cloud access needed.

Run:  python src/analyze.py && python src/verify_sql.py
"""
import json
import re
from pathlib import Path

import duckdb
import pandas as pd
import sqlglot

ROOT = Path(__file__).resolve().parent.parent
PROJECT, DATASET = "my-first-pro-504117", "egypt_retail"


def connect():
    con = duckdb.connect()
    con.execute(f'ATTACH \':memory:\' AS "{PROJECT}"')
    con.execute(f'CREATE SCHEMA "{PROJECT}".{DATASET}')
    for t in ["customers", "products", "orders", "order_items"]:
        con.execute(f"""CREATE TABLE "{PROJECT}".{DATASET}.{t} AS
                        SELECT * FROM read_csv('{ROOT / "data" / (t + ".csv")}', header=true, auto_detect=true)""")
    return con


def run_file(con, name):
    """Run every statement in a BigQuery file via DuckDB; return the DataFrame of the last SELECT."""
    sql = (ROOT / "sql" / name).read_text()
    result = None
    for stmt in sqlglot.parse(sql, read="bigquery"):
        duck_sql = stmt.sql(dialect="duckdb")
        cur = con.execute(duck_sql)
        if stmt.key == "select" or stmt.key == "with":
            result = cur.fetchdf()
    return result


def close(a, b, tol=1.0):
    return bool(abs(float(a) - float(b)) <= tol)


def main():
    con = connect()
    checks = []

    dq = run_file(con, "01_clean_views.sql").iloc[0]
    metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text())["data_quality"]
    checks.append(("01 duplicate orders removed",
                   int(dq.raw_orders - dq.clean_orders) == metrics["duplicate_orders_removed"]))
    checks.append(("01 city spellings after cleaning",
                   int(dq.city_values_after) == metrics["city_values_after"]))

    sql_m = run_file(con, "02_monthly_kpis.sql")
    py_m = pd.read_csv(ROOT / "outputs" / "monthly_summary.csv")
    checks.append(("02 months match", len(sql_m) == len(py_m)))
    checks.append(("02 revenue matches", all(close(a, b) for a, b in zip(sql_m.revenue, py_m.revenue))))
    checks.append(("02 orders match", list(sql_m.orders) == list(py_m.orders)))
    checks.append(("02 gross profit matches", all(close(a, b) for a, b in zip(sql_m.gross_profit, py_m.profit))))
    checks.append(("02 new-customer revenue matches",
                   all(close(a, b) for a, b in zip(sql_m.new_customer_revenue, py_m.new_customer_revenue))))

    sql_r = run_file(con, "03_rfm_segmentation.sql").set_index("segment")
    py_r = pd.read_csv(ROOT / "outputs" / "rfm_segments.csv").set_index("segment")
    diffs = {s: (int(sql_r.customers[s]), int(py_r.customers[s])) for s in py_r.index}
    worst = max(abs(a - b) for a, b in diffs.values())
    checks.append((f"03 RFM segment sizes (max diff {worst} customers; NTILE vs qcut boundary rounding)", worst <= 5))
    checks.append(("03 RFM total customers", int(sql_r.customers.sum()) == int(py_r.customers.sum())))

    sql_c = run_file(con, "04_cohort_retention.sql")
    py_c = {c["cohort"]: c for c in json.loads(
        re.sub(r"^window\.DASHBOARD_DATA = |;\s*$", "", (ROOT / "dashboard" / "data.js").read_text()))["cohorts"]}
    ok_c = True
    for r in sql_c.itertuples():
        expected = py_c[str(r.cohort_month)[:7]]["retention"][int(r.months_since_first_order)]
        if expected is not None and abs(expected - r.retention_pct) > 0.11:
            ok_c = False
    checks.append(("04 cohort retention matches", ok_c))

    # file 05 has two SELECTs; run them separately
    text = (ROOT / "sql" / "05_channel_and_profitability.sql").read_text()
    parts = [p for p in text.split(";") if "SELECT" in p]
    sql_ch = con.execute(sqlglot.transpile(parts[0], read="bigquery", write="duckdb")[0]).fetchdf().set_index("acquisition_channel")
    py_ch = pd.read_csv(ROOT / "outputs" / "channel_performance.csv").set_index("acquisition_channel")
    checks.append(("05A channel repeat rates match",
                   all(close(sql_ch.repeat_rate_pct[c], py_ch.repeat_rate_pct[c], 0.11) for c in py_ch.index)))
    sql_d = con.execute(sqlglot.transpile(parts[1], read="bigquery", write="duckdb")[0]).fetchdf()
    checks.append(("05B discount-band query runs", len(sql_d) > 10))

    width = max(len(n) for n, _ in checks)
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{sum(ok for _, ok in checks)}/{len(checks)} checks passed")
    sql_d.to_csv(ROOT / "outputs" / "discount_margin_by_category.csv", index=False)
    raise SystemExit(0 if all(ok for _, ok in checks) else 1)


if __name__ == "__main__":
    main()
