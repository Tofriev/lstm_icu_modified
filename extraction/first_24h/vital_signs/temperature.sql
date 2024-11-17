WITH t_f AS (
  SELECT
    ce.stay_id,
    ce.charttime,
    (ce.valuenum - 32) / 1.8 AS temperature_c
  FROM
    chartevents ce
  INNER JOIN icustays ic ON ce.stay_id = ic.stay_id
  WHERE
    ce.charttime BETWEEN ic.intime AND datetime(ic.intime, '+24 hours')
    AND ce.valuenum IS NOT NULL
    AND ce.itemid = 223761  -- Fahrenheit
    AND ce.valuenum BETWEEN 68 AND 113
),
t_c AS (
  SELECT
    ce.stay_id,
    ce.charttime,
    ce.valuenum AS temperature_c
  FROM
    chartevents ce
  INNER JOIN icustays ic ON ce.stay_id = ic.stay_id
  WHERE
    ce.charttime BETWEEN ic.intime AND datetime(ic.intime, '+24 hours')
    AND ce.valuenum IS NOT NULL
    AND ce.itemid = 223762  -- Celsius
    AND ce.valuenum BETWEEN 20 AND 45
),
temperatures AS (
  SELECT * FROM t_f
  UNION ALL
  SELECT * FROM t_c
)
SELECT
  stay_id,
  charttime,
  temperature_c AS temperature_value
FROM
  temperatures
ORDER BY
  stay_id, charttime;
