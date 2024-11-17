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
gcs_eyes AS (
    SELECT
        ce.stay_id,
        ce.charttime,
        ce.valuenum AS eyes_value
    FROM chartevents ce
    INNER JOIN valid_intervals vi ON ce.stay_id = vi.stay_id
    WHERE vi.valid_observation = 1
        AND ce.charttime BETWEEN vi.data_start AND vi.data_end
        AND ce.valuenum IS NOT NULL
        AND ce.itemid = 220739
        AND ce.valuenum <= 4
        AND ce.valuenum >= 1
),
gcs_verbal AS (
    SELECT
        ce.stay_id,
        ce.charttime,
        ce.valuenum AS verbal_value
    FROM chartevents ce
    INNER JOIN valid_intervals vi ON ce.stay_id = vi.stay_id
    WHERE vi.valid_observation = 1
        AND ce.charttime BETWEEN vi.data_start AND vi.data_end
        AND ce.valuenum IS NOT NULL
        AND ce.itemid = 223900
        AND ce.valuenum <= 5
        AND ce.valuenum >= 1
),
gcs_motor AS (
    SELECT
        ce.stay_id,
        ce.charttime,
        ce.valuenum AS motor_value
    FROM chartevents ce
    INNER JOIN valid_intervals vi ON ce.stay_id = vi.stay_id
    WHERE vi.valid_observation = 1
        AND ce.charttime BETWEEN vi.data_start AND vi.data_end
        AND ce.valuenum IS NOT NULL
        AND ce.itemid = 223901
        AND ce.valuenum <= 6
        AND ce.valuenum >= 1
)
SELECT
    eyes.stay_id,
    eyes.charttime,
    eyes.eyes_value,
    verbal.verbal_value,
    motor.motor_value,
    (eyes.eyes_value + verbal.verbal_value + motor.motor_value) AS gcs_total_value
FROM gcs_eyes eyes
INNER JOIN gcs_verbal verbal
    ON eyes.stay_id = verbal.stay_id AND eyes.charttime = verbal.charttime
INNER JOIN gcs_motor motor
    ON eyes.stay_id = motor.stay_id AND eyes.charttime = motor.charttime
ORDER BY eyes.stay_id, eyes.charttime;
