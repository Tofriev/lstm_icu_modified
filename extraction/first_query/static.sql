WITH long_stays AS (
    SELECT *
    FROM icustays
    WHERE outtime >= datetime(intime,'+24 hours')
),
mort AS (
    SELECT
        ls.subject_id,
        ls.stay_id,
        ls.hadm_id,
        ls.intime,
        ls.outtime,
        datetime(ls.intime,'+24 hours')    AS first_day_end,
        CASE
            WHEN admissions.deathtime BETWEEN ls.intime AND ls.outtime
            THEN 1 ELSE 0 END                                        AS mortality
    FROM long_stays ls
    JOIN admissions ON admissions.hadm_id = ls.hadm_id
),
hw_raw AS (
    SELECT
        ls.stay_id,
        ls.intime,
        ce.charttime,
        ce.itemid,
        ce.valuenum                                                  AS value,
        CASE
            WHEN ce.charttime BETWEEN ls.intime
                                 AND datetime(ls.intime,'+24 hours')
            THEN 'after' ELSE 'before' END                           AS time_flag
    FROM long_stays ls
    JOIN chartevents ce ON ce.stay_id = ls.stay_id
    WHERE ce.itemid IN (226730, 226512)
      AND ce.valuenum IS NOT NULL
      AND ( (ce.itemid = 226730 AND ce.valuenum BETWEEN 20  AND 260)  /* height  cm */
         OR (ce.itemid = 226512 AND ce.valuenum BETWEEN 20  AND 500) )/* weight  kg */
),
hw_after AS (
    SELECT
        stay_id,
        MAX(CASE WHEN itemid = 226730 THEN 1 END)               AS has_h,
        MAX(CASE WHEN itemid = 226512 THEN 1 END)               AS has_w,
        AVG(CASE WHEN itemid = 226730 THEN value END)           AS height_mean,
        AVG(CASE WHEN itemid = 226512 THEN value END)           AS weight_mean
    FROM hw_raw
    WHERE time_flag = 'after'
    GROUP BY stay_id
),
hw_before AS (
    SELECT
        stay_id,
        itemid,
        value                                                     AS vw,
        ROW_NUMBER() OVER (
            PARTITION BY stay_id, itemid
            ORDER BY ABS(julianday(charttime) - julianday(intime))
        )                                                         AS rn
    FROM hw_raw
    WHERE time_flag = 'before'
),
height_fallback AS (
    SELECT stay_id, vw AS height_mean
    FROM hw_before
    WHERE itemid = 226730 AND rn = 1
),
weight_fallback AS (
    SELECT stay_id, vw AS weight_mean
    FROM hw_before
    WHERE itemid = 226512 AND rn = 1
),
hw_final AS (
    SELECT
        ls.stay_id,
        COALESCE(ha.height_mean, hf.height_mean) AS height_mean,
        COALESCE(ha.weight_mean, wf.weight_mean) AS weight_mean
    FROM long_stays ls
    LEFT JOIN hw_after       ha ON ha.stay_id = ls.stay_id
    LEFT JOIN height_fallback hf ON hf.stay_id = ls.stay_id
    LEFT JOIN weight_fallback wf ON wf.stay_id = ls.stay_id
)
SELECT
    m.stay_id,
    m.intime,
    m.first_day_end,
    m.mortality AS mortality_value,
    p.anchor_age AS age_value,
    p.gender AS gender_value,
    hw.height_mean AS height_value,
    hw.weight_mean AS weight_value
FROM mort          m
JOIN patients      p  ON p.subject_id = m.subject_id
LEFT JOIN hw_final hw ON hw.stay_id   = m.stay_id
ORDER BY m.stay_id;
