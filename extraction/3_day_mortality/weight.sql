WITH mort AS (
    SELECT
        ic.subject_id,
        ic.stay_id,
        ic.hadm_id,
        ic.intime,
        ic.outtime,
        CASE
            WHEN adm.deathtime BETWEEN ic.intime AND ic.outtime THEN 1
            ELSE 0
        END AS mortality,
        adm.deathtime
    FROM icustays ic
    INNER JOIN admissions adm ON ic.hadm_id = adm.hadm_id
),
data_intervals AS (
    SELECT
        m.subject_id,
        m.stay_id,
        m.hadm_id,
        m.intime,
        m.outtime,
        m.mortality,
        m.deathtime,
        CASE
            WHEN m.mortality = 1 THEN datetime(m.deathtime, '-3 days')
            ELSE m.intime
        END AS data_start,
        CASE
            WHEN m.mortality = 1 THEN datetime(m.deathtime, '-3 days', '+24 hours')
            ELSE datetime(m.intime, '+24 hours')
        END AS data_end
    FROM mort m
),
valid_intervals AS (
    SELECT
        di.subject_id,
        di.stay_id,
        di.hadm_id,
        di.data_start,
        di.data_end,
        CASE
            WHEN di.data_end <= di.outtime THEN 1
            ELSE 0
        END AS valid_observation
    FROM data_intervals di
    WHERE di.data_end <= di.outtime
),
weight_during AS (
    SELECT
        vi.stay_id,
        ce.charttime,
        ce.valuenum AS weight_value
    FROM chartevents ce
    INNER JOIN valid_intervals vi ON ce.stay_id = vi.stay_id
    WHERE vi.valid_observation = 1
        AND ce.valuenum IS NOT NULL
        AND ce.itemid IN (224639, 226512) -- Daily Weight and Admission Weight
        AND ce.valuenum BETWEEN 20 AND 500
        AND ce.charttime BETWEEN vi.data_start AND vi.data_end
),
weight_before AS (
    SELECT
        vi.stay_id,
        vi.data_start AS charttime,
        ce.valuenum AS weight_value,
        ROW_NUMBER() OVER (PARTITION BY vi.stay_id ORDER BY ce.charttime DESC) AS rn
    FROM chartevents ce
    INNER JOIN valid_intervals vi ON ce.hadm_id = vi.hadm_id
    WHERE ce.valuenum IS NOT NULL
        AND ce.itemid IN (224639, 226512)
        AND ce.valuenum BETWEEN 20 AND 500
        AND ce.charttime < vi.data_start
        AND vi.stay_id NOT IN (SELECT stay_id FROM weight_during)
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
FROM combined_weights
ORDER BY stay_id, charttime;
