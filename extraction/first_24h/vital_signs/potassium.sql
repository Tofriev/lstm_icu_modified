WITH potassium AS (
  SELECT
    ce.stay_id,
    ce.charttime,
    ce.valuenum AS potassium_value
  FROM
    chartevents ce
  INNER JOIN icustays ic ON ce.stay_id = ic.stay_id
  WHERE
    ce.charttime BETWEEN ic.intime AND datetime(ic.intime, '+24 hours') AND
    ce.valuenum IS NOT NULL AND
    ce.itemid = 227442
    AND ce.valuenum <= 7
    AND ce.valuenum >= 2.5
)
SELECT
  ic.stay_id,
  potassium.charttime,
  potassium.potassium_value
FROM
  icustays ic
INNER JOIN potassium ON ic.stay_id = potassium.stay_id
ORDER BY
  ic.stay_id, potassium.charttime
