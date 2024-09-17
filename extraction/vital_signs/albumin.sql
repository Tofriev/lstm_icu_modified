WITH albumin AS (
  SELECT
    ce.stay_id,
    ce.charttime,
    ce.valuenum AS albumin_value
  FROM
    chartevents ce
  INNER JOIN icustays ic ON ce.stay_id = ic.stay_id
  WHERE
    ce.charttime BETWEEN ic.intime AND datetime(ic.intime, '+24 hours') AND
    ce.valuenum IS NOT NULL AND
    ce.itemid = 227456
    AND ce.valuenum <= 60
    AND ce.valuenum >= 2
)
SELECT
  ic.stay_id,
  albumin.charttime,
  albumin.albumin_value
FROM
  icustays ic
INNER JOIN albumin ON ic.stay_id = albumin.stay_id
ORDER BY
  ic.stay_id, albumin.charttime
