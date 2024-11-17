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
urea_nitrogen_data AS (
    SELECT
        vi.stay_id,
        l.charttime,
        l.valuenum AS urea_value
    FROM labevents l
    INNER JOIN valid_intervals vi
        ON l.hadm_id = vi.hadm_id
    WHERE vi.valid_observation = 1
        AND l.charttime BETWEEN vi.data_start AND vi.data_end
        AND l.valuenum IS NOT NULL
        AND l.itemid IN (51006, 51300) -- Urea Nitrogen Item IDs
        AND l.valuenum <= 100
        AND l.valuenum >= 1
)
SELECT
    stay_id,
    charttime,
    urea_value
FROM urea_nitrogen_data
ORDER BY stay_id, charttime;
