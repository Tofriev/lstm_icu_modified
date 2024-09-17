WITH mort AS (
    SELECT
        ic.subject_id,
        ic.stay_id,
        ic.hadm_id,
        ic.intime,
       	datetime(ic.intime, '+24 hours') AS first_day_end,
        CASE
          WHEN 
            adm.deathtime BETWEEN ic.intime AND ic.outtime AND adm.deathtime >= datetime(ic.intime, '+25 hours') THEN 1
            ELSE 0
        END AS mortality   
    FROM icustays ic
    INNER JOIN admissions adm ON ic.hadm_id = adm.hadm_id
),
height_data AS ( -- same logic for height 
    SELECT
        icu.stay_id,
        icu.intime,
        ce.charttime,
        ce.valuenum AS height,
        CASE
            WHEN ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') THEN 'after'
            ELSE 'before'
        END AS time_category
    FROM
        icustays icu
        INNER JOIN chartevents ce ON icu.stay_id = ce.stay_id
    WHERE
        ce.valuenum IS NOT NULL AND
        ce.valuenum != 0 AND
        ce.itemid IN (226730) 
        AND ce.valuenum <= 260
),
height_after AS (
    SELECT
        stay_id,
        AVG(height) AS height_mean
    FROM height_data
    WHERE time_category = 'after'
    GROUP BY stay_id
),
height_before AS (
    SELECT
        stay_id,
        height AS height_mean,
        ROW_NUMBER() OVER(PARTITION BY stay_id ORDER BY ABS(JULIANDAY(charttime) - JULIANDAY(intime))) AS rn
    FROM height_data
    WHERE time_category = 'before'
),
height_final AS (
    SELECT
        stay_id,
        height_mean
    FROM height_after
    UNION ALL
    SELECT
        stay_id,
        height_mean
    FROM height_before
    WHERE rn = 1 AND stay_id NOT IN (SELECT stay_id FROM height_after)
)
SELECT
  m.mortality,
  p.anchor_age,
  p.gender,
  hf.height_mean AS height,
  m.intime,
  m.first_day_end,
  ic.stay_id
FROM
  icustays ic
INNER JOIN mort m ON ic.stay_id = m.stay_id
INNER JOIN patients p ON ic.subject_id = p.subject_id
LEFT JOIN height_final AS hf ON ic.stay_id = hf.stay_id
ORDER BY
  ic.stay_id
