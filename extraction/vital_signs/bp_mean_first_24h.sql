WITH mean_bp AS (
  SELECT
    ce.stay_id,
    ce.charttime,
    ce.valuenum AS mbp_value
  FROM
    chartevents ce
  INNER JOIN icustays ic ON ce.stay_id = ic.stay_id
  WHERE
    ce.charttime BETWEEN ic.intime AND datetime(ic.intime, '+24 hours') AND
    ce.valuenum IS NOT NULL AND
    ce.itemid IN (220052, 220181)
    AND ce.valuenum <= 400
    AND ce.valuenum >= 20
)

SELECT
  ic.stay_id,
  b.charttime,
  b.mbp_value
FROM
  icustays ic
INNER JOIN mean_bp b ON ic.stay_id = b.stay_id
ORDER BY
  ic.stay_id, b.charttime;
