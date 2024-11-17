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
        -- Define data_start based on mortality status
        CASE
            WHEN m.mortality = 1 THEN datetime(m.deathtime, '-3 days') -- Deceased patients: 3 days before death
            ELSE m.intime -- Survivors: Start from admission
        END AS data_start,
        -- Define data_end as 24 hours after data_start
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
anion_gap_data AS (
    SELECT
        ce.stay_id,
        ce.charttime,
        ce.valuenum AS anion_gap_value
    FROM
        chartevents ce
    INNER JOIN valid_intervals vi ON ce.stay_id = vi.stay_id
    WHERE
        vi.valid_observation = 1 
        AND ce.charttime BETWEEN vi.data_start AND vi.data_end 
        AND ce.valuenum IS NOT NULL
        AND ce.itemid = 227073 
        AND ce.valuenum <= 25 
        AND ce.valuenum >= 1
)
SELECT
    vi.stay_id,
    ag.charttime,
    ag.anion_gap_value
FROM
    valid_intervals vi
INNER JOIN anion_gap_data ag ON vi.stay_id = ag.stay_id
ORDER BY
    vi.stay_id, ag.charttime;
