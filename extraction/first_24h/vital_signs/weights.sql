WITH weight_during AS (
  SELECT
    icu.stay_id,
    ce.charttime,
    ce.valuenum AS weight_value
  FROM
    chartevents ce
  INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
  WHERE
    ce.valuenum IS NOT NULL
    AND ce.itemid IN (224639, 226512)  -- Daily Weight and Admission Weight
    AND ce.valuenum BETWEEN 20 AND 500
    AND ce.charttime BETWEEN icu.intime AND icu.outtime
),
weight_before AS (
  SELECT
    icu.stay_id,
    icu.intime AS charttime,  -- charttime to icu.intime
    ce.valuenum AS weight_value,
    ROW_NUMBER() OVER (PARTITION BY icu.stay_id ORDER BY ce.charttime DESC) AS rn
  FROM
    chartevents ce
  INNER JOIN icustays icu ON ce.hadm_id = icu.hadm_id
  WHERE
    ce.valuenum IS NOT NULL
    AND ce.itemid IN (224639, 226512)
    AND ce.valuenum BETWEEN 20 AND 500
    AND ce.charttime < icu.intime
    AND icu.stay_id NOT IN (SELECT stay_id FROM weight_during)
),
combined_weights AS (
  SELECT * FROM weight_during
  UNION ALL
  SELECT stay_id, charttime, weight_value
  FROM weight_before
  WHERE rn = 1
)
SELECT
  stay_id,
  charttime,
  weight_value
FROM
  combined_weights
ORDER BY
  stay_id, charttime;
