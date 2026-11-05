-- 05_channel_and_profitability.sql
-- Two answers in one file. Run each statement separately in the BigQuery console.

-- A) Which acquisition channel brings customers who actually come back?
WITH per_customer AS (
  SELECT
    c.customer_id,
    c.acquisition_channel,
    COUNT(DISTINCT d.order_id) AS orders,
    COALESCE(SUM(d.revenue), 0) AS revenue
  FROM `my-first-pro-504117.egypt_retail.clean_customers` AS c
  LEFT JOIN `my-first-pro-504117.egypt_retail.order_lines` AS d
    ON d.customer_id = c.customer_id AND d.order_status = 'Delivered'
  WHERE c.signup_date >= DATE '2025-01-01'
  GROUP BY c.customer_id, c.acquisition_channel
)
SELECT
  acquisition_channel,
  COUNT(*) AS customers,
  COUNTIF(orders >= 1) AS buyers,
  COUNTIF(orders >= 2) AS repeat_buyers,
  ROUND(100 * SAFE_DIVIDE(COUNTIF(orders >= 1), COUNT(*)), 1) AS conversion_pct,
  ROUND(100 * SAFE_DIVIDE(COUNTIF(orders >= 2), COUNTIF(orders >= 1)), 1) AS repeat_rate_pct,
  ROUND(SAFE_DIVIDE(SUM(revenue), COUNT(*))) AS revenue_per_customer
FROM per_customer
GROUP BY acquisition_channel
ORDER BY revenue_per_customer DESC;

-- B) Do deep discounts still make money? Margin by category and discount depth.
SELECT
  category,
  CASE
    WHEN discount_pct = 0 THEN '0%'
    WHEN discount_pct <= 15 THEN '5-15%'
    WHEN discount_pct <= 30 THEN '20-30%'
    ELSE '40-50%'
  END AS discount_band,
  COUNT(*) AS order_lines,
  ROUND(SUM(revenue)) AS revenue,
  ROUND(SUM(gross_profit)) AS gross_profit,
  ROUND(100 * SAFE_DIVIDE(SUM(gross_profit), SUM(revenue)), 1) AS margin_pct
FROM `my-first-pro-504117.egypt_retail.order_lines`
WHERE order_status = 'Delivered'
GROUP BY category, discount_band
ORDER BY category, MIN(discount_pct);
