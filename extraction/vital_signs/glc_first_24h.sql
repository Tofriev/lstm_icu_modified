WITH glucose AS (
  SELECT
    ce.stay_id,
    ce.charttime,
    ce.valuenum AS glc_value
  FROM
    chartevents ce
  INNER JOIN icustays ic ON ce.stay_id = ic.stay_id
  WHERE
    ce.charttime BETWEEN ic.intime AND datetime(ic.intime, '+24 hours') AND
    ce.valuenum IS NOT NULL AND
    ce.itemid = 220621
    AND ce.valuenum <= 2000
    AND ce.valuenum >= 5
)

SELECT
  ic.stay_id,
  g.charttime,
  g.glc_value
FROM
  icustays ic
INNER JOIN glucose g ON ic.stay_id = g.stay_id
ORDER BY
  ic.stay_id, g.charttime;
