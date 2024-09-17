WITH lactate AS (
  SELECT
    icu.stay_id,
    l.charttime,
    l.valuenum AS lactate_value
  FROM
    labevents l
  INNER JOIN icustays icu
    ON l.hadm_id = icu.hadm_id
    AND l.charttime BETWEEN icu.intime AND icu.outtime
  WHERE
    l.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours')
    AND l.valuenum IS NOT NULL
    AND l.itemid IN (50813, 52442, 53154)
    AND l.valuenum BETWEEN 0.1 AND 200
)
SELECT
  stay_id,
  charttime,
  lactate_value
FROM
  lactate
ORDER BY
  stay_id, charttime
