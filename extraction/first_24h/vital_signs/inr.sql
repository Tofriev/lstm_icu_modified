-- Prothrombin Time
WITH inr AS (
  SELECT
    ce.stay_id,
    ce.charttime,
    ce.valuenum AS inr_value
  FROM
    chartevents ce
  INNER JOIN icustays ic ON ce.stay_id = ic.stay_id
  WHERE
    ce.charttime BETWEEN ic.intime AND datetime(ic.intime, '+24 hours') AND
    ce.valuenum IS NOT NULL AND
    ce.itemid = 227467
    AND ce.valuenum <= 6
    AND ce.valuenum >= 0.2
)
SELECT
  ic.stay_id,
  inr.charttime,
  inr.inr_value
FROM
  icustays ic
INNER JOIN inr ON ic.stay_id = inr.stay_id
ORDER BY
  ic.stay_id, inr.charttime
