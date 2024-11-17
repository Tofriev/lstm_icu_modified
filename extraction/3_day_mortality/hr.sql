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
        di.data_start,
        di.data_end,
        CASE
            WHEN di.data_end <= di.outtime THEN 1
            ELSE 0
        END AS valid_observation
    FROM data_intervals di
    WHERE di.data_end <= di.outtime
),
heartrate_data AS (
    SELECT
        ce.stay_id,
        ce.charttime,
        ce.valuenum AS hr_value
    FROM chartevents ce
    INNER JOIN valid_intervals vi ON ce.stay_id = vi.stay_id
    WHERE vi.valid_observation = 1
        AND ce.charttime BETWEEN vi.data_start AND vi.data_end
        AND ce.valuenum IS NOT NULL
        AND ce.itemid = 220045
        AND ce.valuenum <= 300
        AND ce.valuenum >= 10
)
SELECT
    vi.stay_id,
    hr.charttime,
    hr.hr_value
FROM valid_intervals vi
INNER JOIN heartrate_data hr ON vi.stay_id = hr.stay_id
ORDER BY vi.stay_id, hr.charttime;
