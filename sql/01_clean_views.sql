-- 01_clean_views.sql
-- Cleans the four raw tables and builds one line-level fact view used by every other query.
-- Raw tables expected in `my-first-pro-504117.egypt_retail`: customers, products, orders, order_items
-- (see sql/load_to_bigquery.sh).

-- 1) Customers: standardise city spellings (case, whitespace, Arabic names) and drop customers with no city.
CREATE OR REPLACE VIEW `my-first-pro-504117.egypt_retail.clean_customers` AS
SELECT
  customer_id,
  signup_date,
  acquisition_channel,
  CASE
    WHEN TRIM(city) = 'القاهرة' THEN 'Cairo'
    WHEN TRIM(city) = 'الجيزة' THEN 'Giza'
    WHEN TRIM(city) = 'الإسكندرية' THEN 'Alexandria'
    WHEN LOWER(TRIM(city)) = 'alex' THEN 'Alexandria'
    ELSE INITCAP(TRIM(city))
  END AS city
FROM `my-first-pro-504117.egypt_retail.customers`
WHERE city IS NOT NULL AND TRIM(city) != '';

-- 2) Orders: drop exact duplicate order_ids, orders of customers removed above, and label missing payment methods.
CREATE OR REPLACE VIEW `my-first-pro-504117.egypt_retail.clean_orders` AS
SELECT
  order_id,
  customer_id,
  order_date,
  DATE_TRUNC(order_date, MONTH) AS order_month,
  COALESCE(payment_method, 'Unknown') AS payment_method,
  order_status,
  shipping_fee
FROM `my-first-pro-504117.egypt_retail.orders`
WHERE customer_id IN (SELECT customer_id FROM `my-first-pro-504117.egypt_retail.clean_customers`)
QUALIFY ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY order_date) = 1;

-- 3) Line-level fact view: revenue and gross profit per order line.
--    Revenue is net of discount. Cost is scaled with the price level (cost_price / base_price),
--    so a deep discount reduces profit but not cost.
CREATE OR REPLACE VIEW `my-first-pro-504117.egypt_retail.order_lines` AS
SELECT
  oi.order_id,
  o.customer_id,
  o.order_date,
  o.order_month,
  o.order_status,
  o.payment_method,
  c.city,
  c.acquisition_channel,
  c.signup_date,
  p.category,
  p.product_name,
  oi.quantity,
  oi.unit_price,
  oi.discount_pct,
  oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100) AS revenue,
  oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)
    - oi.quantity * oi.unit_price * SAFE_DIVIDE(p.cost_price, p.base_price) AS gross_profit
FROM `my-first-pro-504117.egypt_retail.order_items` AS oi
JOIN `my-first-pro-504117.egypt_retail.clean_orders` AS o USING (order_id)
JOIN `my-first-pro-504117.egypt_retail.clean_customers` AS c USING (customer_id)
JOIN `my-first-pro-504117.egypt_retail.products` AS p USING (product_id);

-- Data-quality check: how many problems did the views remove?
SELECT
  (SELECT COUNT(*) FROM `my-first-pro-504117.egypt_retail.orders`)       AS raw_orders,
  (SELECT COUNT(*) FROM `my-first-pro-504117.egypt_retail.clean_orders`) AS clean_orders,
  (SELECT COUNT(DISTINCT city) FROM `my-first-pro-504117.egypt_retail.customers`)       AS city_values_before,
  (SELECT COUNT(DISTINCT city) FROM `my-first-pro-504117.egypt_retail.clean_customers`) AS city_values_after;
