WITH heartrate AS (
  SELECT
    ce.stay_id,
    ce.charttime,
    ce.valuenum AS hr_value
  FROM
    chartevents ce
  INNER JOIN icustays ic ON ce.stay_id = ic.stay_id
  WHERE
    ce.charttime BETWEEN ic.intime AND datetime(ic.intime, '+24 hours') AND
    ce.valuenum IS NOT NULL AND
    ce.itemid = 220045
    AND ce.valuenum <= 300
    AND ce.valuenum >= 10
)

SELECT
  ic.stay_id,
  h.charttime,
  h.hr_value
FROM
  icustays ic
INNER JOIN heartrate h ON ic.stay_id = h.stay_id
ORDER BY
  ic.stay_id, h.charttime;
