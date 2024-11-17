WITH urea_nitrogen AS (
  SELECT
    icu.stay_id,
    l.charttime,
    l.valuenum AS urea_value
  FROM
    labevents l
  INNER JOIN icustays icu
    ON l.hadm_id = icu.hadm_id
    AND l.charttime BETWEEN icu.intime AND icu.outtime
  WHERE
    l.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours')
    AND l.valuenum IS NOT NULL
    AND l.itemid IN (51006, 51300)
    AND l.valuenum <= 100
    AND l.valuenum >= 1
)
SELECT
  stay_id,
  charttime,
  urea_value
FROM
  urea_nitrogen
ORDER BY
  stay_id, charttime;
