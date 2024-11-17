WITH anion_gap AS (
  SELECT
    ce.stay_id,
    ce.charttime,
    ce.valuenum AS anion_gap_value
  FROM
    chartevents ce
  INNER JOIN icustays ic ON ce.stay_id = ic.stay_id
  WHERE
    ce.charttime BETWEEN ic.intime AND datetime(ic.intime, '+24 hours') AND
    ce.valuenum IS NOT NULL AND
    ce.itemid = 227073
    AND ce.valuenum <= 25
    AND ce.valuenum >= 1
)
SELECT
  ic.stay_id,
  anion_gap.charttime,
  anion_gap.anion_gap_value
FROM
  icustays ic
INNER JOIN anion_gap ON ic.stay_id = anion_gap.stay_id
ORDER BY
  ic.stay_id, anion_gap.charttime
