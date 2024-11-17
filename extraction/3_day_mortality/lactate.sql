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
lactate_data AS (
    SELECT
        vi.stay_id,
        l.charttime,
        l.valuenum AS lactate_value
    FROM labevents l
    INNER JOIN valid_intervals vi
        ON l.hadm_id = vi.hadm_id
    WHERE vi.valid_observation = 1
        AND l.charttime BETWEEN vi.data_start AND vi.data_end
        AND l.valuenum IS NOT NULL
        AND l.itemid IN (50813, 52442, 53154)
        AND l.valuenum BETWEEN 0.1 AND 200
)
SELECT
    stay_id,
    charttime,
    lactate_value
FROM lactate_data
ORDER BY stay_id, charttime;
