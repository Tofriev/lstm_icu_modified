WITH ph AS (
  SELECT
    ce.stay_id,
    ce.charttime,
    ce.valuenum AS ph_value
  FROM
    chartevents ce
  INNER JOIN icustays ic ON ce.stay_id = ic.stay_id
  WHERE
    ce.charttime BETWEEN ic.intime AND datetime(ic.intime, '+24 hours') AND
    ce.valuenum IS NOT NULL AND
    ce.itemid IN (220274, 223830) --venous, arterial 
        AND ce.valuenum <= 9
        AND ce.valuenum >= 5
)
SELECT
  ic.stay_id,
  ph.charttime,
  ph.ph_value
FROM
  icustays ic
INNER JOIN ph ON ic.stay_id = ph.stay_id
ORDER BY
  ic.stay_id, ph.charttime
