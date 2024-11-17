WITH platelets AS (
  SELECT
    ce.stay_id,
    ce.charttime,
    ce.valuenum AS platelets_value
  FROM
    chartevents ce
  INNER JOIN icustays ic ON ce.stay_id = ic.stay_id
  WHERE
    ce.charttime BETWEEN ic.intime AND datetime(ic.intime, '+24 hours') AND
    ce.valuenum IS NOT NULL AND
    ce.itemid = 227457
    AND ce.valuenum <= 1000
    AND ce.valuenum >= 10
)
SELECT
  ic.stay_id,
  platelets.charttime,
  platelets.platelets_value
FROM
  icustays ic
INNER JOIN platelets ON ic.stay_id = platelets.stay_id
ORDER BY
  ic.stay_id, platelets.charttime
  