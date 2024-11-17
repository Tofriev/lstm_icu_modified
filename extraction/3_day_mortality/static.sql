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
        -- when deceased start 3 days before death, othewrise start at admission
        CASE
            WHEN m.mortality = 1 THEN datetime(m.deathtime, '-3 days')
            ELSE m.intime
        END AS data_start,
        --  end 24 hours after the start time
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
        di.intime,
        di.outtime,
        di.mortality,
        di.data_start,
        di.data_end,
        -- is interval valid ?
        CASE
            WHEN datetime(di.data_start, '+24 hours') <= di.outtime THEN 1
            ELSE 0
        END AS valid_observation
    FROM data_intervals di
    WHERE datetime(di.data_start, '+24 hours') <= di.outtime -- include if 24 hours of data are available
),
height_data AS (
    SELECT
        icu.stay_id,
        ce.charttime,
        ce.valuenum AS height,
        CASE
            WHEN ce.charttime BETWEEN vi.data_start AND vi.data_end THEN 'selected'
            ELSE 'excluded'
        END AS time_category
    FROM
        icustays icu
    INNER JOIN valid_intervals vi ON icu.stay_id = vi.stay_id
    INNER JOIN chartevents ce ON icu.stay_id = ce.stay_id
    WHERE
        vi.valid_observation = 1 -- only valid observations
        AND ce.valuenum IS NOT NULL
        AND ce.valuenum != 0
        AND ce.itemid IN (226730)
        AND ce.valuenum <= 260
),
height_final AS (
    SELECT
        stay_id,
        AVG(height) AS height_mean
    FROM height_data
    WHERE time_category = 'selected'
    GROUP BY stay_id
)
SELECT
    vi.mortality,
    p.anchor_age,
    p.gender,
    hf.height_mean AS height,
    vi.intime,
    vi.data_start,
    vi.data_end,
    ic.stay_id
FROM
    icustays ic
INNER JOIN valid_intervals vi ON ic.stay_id = vi.stay_id
INNER JOIN patients p ON ic.subject_id = p.subject_id
LEFT JOIN height_final hf ON ic.stay_id = hf.stay_id
WHERE vi.valid_observation = 1 -- only valid intervals are included
ORDER BY
    ic.stay_id;
