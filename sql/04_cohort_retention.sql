-- 04_cohort_retention.sql
-- Monthly retention by first-order cohort (customers who signed up on/after 2025-01-01).
-- retention_pct = share of the cohort that placed a delivered order N months after their first one.
WITH delivered AS (
  SELECT * FROM `my-first-pro-504117.egypt_retail.order_lines`
  WHERE order_status = 'Delivered'
),
cohort AS (
  SELECT customer_id, MIN(order_month) AS cohort_month
  FROM delivered
  WHERE signup_date >= DATE '2025-01-01'
  GROUP BY customer_id
),
activity AS (
  SELECT DISTINCT
    d.customer_id,
    c.cohort_month,
    DATE_DIFF(d.order_month, c.cohort_month, MONTH) AS months_since_first_order
  FROM delivered AS d
  JOIN cohort AS c USING (customer_id)
),
counts AS (
  SELECT cohort_month, months_since_first_order, COUNT(DISTINCT customer_id) AS active_customers
  FROM activity
  GROUP BY cohort_month, months_since_first_order
)
SELECT
  cohort_month,
  months_since_first_order,
  active_customers,
  ROUND(100 * SAFE_DIVIDE(
    active_customers,
    FIRST_VALUE(active_customers) OVER (PARTITION BY cohort_month ORDER BY months_since_first_order)
  ), 1) AS retention_pct
FROM counts
WHERE months_since_first_order BETWEEN 0 AND 11
ORDER BY cohort_month, months_since_first_order;
