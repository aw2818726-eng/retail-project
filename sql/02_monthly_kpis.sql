-- 02_monthly_kpis.sql
-- Monthly revenue, orders, AOV, margin, month-over-month growth, and new-customer revenue.
WITH delivered AS (
  SELECT * FROM `my-first-pro-504117.egypt_retail.order_lines`
  WHERE order_status = 'Delivered'
),
first_order AS (
  SELECT customer_id, MIN(order_month) AS first_month
  FROM delivered
  GROUP BY customer_id
),
monthly AS (
  SELECT
    d.order_month,
    ROUND(SUM(d.revenue)) AS revenue,
    COUNT(DISTINCT d.order_id) AS orders,
    ROUND(SUM(d.gross_profit)) AS gross_profit,
    ROUND(SUM(IF(d.order_month = f.first_month, d.revenue, 0))) AS new_customer_revenue
  FROM delivered AS d
  JOIN first_order AS f USING (customer_id)
  GROUP BY d.order_month
)
SELECT
  order_month,
  revenue,
  orders,
  ROUND(SAFE_DIVIDE(revenue, orders)) AS avg_order_value,
  gross_profit,
  ROUND(100 * SAFE_DIVIDE(gross_profit, revenue), 1) AS margin_pct,
  new_customer_revenue,
  ROUND(100 * SAFE_DIVIDE(revenue - LAG(revenue) OVER (ORDER BY order_month),
                          LAG(revenue) OVER (ORDER BY order_month)), 1) AS mom_growth_pct
FROM monthly
ORDER BY order_month;
