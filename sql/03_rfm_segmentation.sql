-- 03_rfm_segmentation.sql
-- RFM scoring (1-5 quintiles) and named customer segments, summarised per segment.
WITH delivered AS (
  SELECT * FROM `my-first-pro-504117.egypt_retail.order_lines`
  WHERE order_status = 'Delivered'
),
snapshot AS (
  SELECT DATE_ADD(MAX(order_date), INTERVAL 1 DAY) AS snapshot_date
  FROM `my-first-pro-504117.egypt_retail.clean_orders`
),
customer_metrics AS (
  SELECT
    d.customer_id,
    DATE_DIFF(ANY_VALUE(s.snapshot_date), MAX(d.order_date), DAY) AS recency_days,
    COUNT(DISTINCT d.order_id) AS frequency,
    SUM(d.revenue) AS monetary
  FROM delivered AS d
  CROSS JOIN snapshot AS s
  GROUP BY d.customer_id
),
scored AS (
  SELECT
    *,
    NTILE(5) OVER (ORDER BY recency_days DESC, customer_id) AS r,  -- 5 = most recent
    NTILE(5) OVER (ORDER BY frequency, customer_id) AS f,
    NTILE(5) OVER (ORDER BY monetary, customer_id) AS m
  FROM customer_metrics
),
segmented AS (
  SELECT
    *,
    CASE
      WHEN r >= 4 AND f >= 4 THEN 'Champions'
      WHEN r >= 3 AND f >= 3 THEN 'Loyal'
      WHEN r >= 4 AND f <= 2 THEN 'New / Promising'
      WHEN r <= 2 AND f >= 3 THEN 'At Risk'
      WHEN r <= 2 AND f <= 2 THEN 'Hibernating'
      ELSE 'Needs Attention'
    END AS segment
  FROM scored
)
SELECT
  segment,
  COUNT(*) AS customers,
  ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS customer_share_pct,
  ROUND(SUM(monetary)) AS revenue,
  ROUND(100 * SUM(monetary) / SUM(SUM(monetary)) OVER (), 1) AS revenue_share_pct,
  ROUND(AVG(recency_days)) AS avg_recency_days,
  ROUND(AVG(frequency), 1) AS avg_orders
FROM segmented
GROUP BY segment
ORDER BY revenue DESC;
