WITH resprate AS (
  SELECT
    ce.stay_id,
    ce.charttime,
    ce.valuenum AS rr_value
  FROM
    chartevents ce
  INNER JOIN icustays ic ON ce.stay_id = ic.stay_id
  WHERE
    ce.charttime BETWEEN ic.intime AND datetime(ic.intime, '+24 hours') AND
    ce.valuenum IS NOT NULL AND
    ce.itemid = 220210
    AND ce.valuenum <= 50
    AND ce.valuenum >= 5
), 
mort AS ( 
  SELECT
    ic.subject_id,
    ic.stay_id,
    ic.hadm_id,
    CASE
      WHEN adm.deathtime BETWEEN datetime(ic.intime, '+24 hours') AND datetime(ic.intime, '+48 hours') THEN 1
      WHEN adm.dischtime BETWEEN datetime(ic.intime, '+24 hours') AND datetime(ic.intime, '+48 hours') AND adm.discharge_location = 'DIED' THEN 1 
      ELSE 0
    END AS Mortality_next24h    
  FROM icustays ic
  INNER JOIN admissions adm ON ic.hadm_id = adm.hadm_id
  WHERE
    -- patients have >= 48h of data or died in the second 24h (not the first 24h)
    (
      (strftime('%s', ic.outtime) - strftime('%s', ic.intime)) >= 172800
      OR adm.deathtime BETWEEN datetime(ic.intime, '+24 hours') AND datetime(ic.intime, '+48 hours')
    )
    AND NOT EXISTS (
      SELECT 1
      FROM admissions adm2
      WHERE adm2.hadm_id = ic.hadm_id
      AND adm2.deathtime BETWEEN ic.intime AND datetime(ic.intime, '+24 hours')
    )

)

SELECT
  m.Mortality_next24h AS mortality,
  ic.stay_id,
  r.charttime,
  r.rr_value
FROM
  icustays ic
INNER JOIN resprate r ON ic.stay_id = r.stay_id
INNER JOIN mort m ON ic.stay_id = m.stay_id
ORDER BY
  ic.stay_id, r.charttime;
