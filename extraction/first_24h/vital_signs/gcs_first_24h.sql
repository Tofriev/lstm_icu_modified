WITH gcs_eyes AS (
  SELECT
    ce.stay_id,
    ce.charttime,
    ce.valuenum AS eyes_value
  FROM
    chartevents ce
  INNER JOIN icustays ic ON ce.stay_id = ic.stay_id
  WHERE
    ce.charttime BETWEEN ic.intime AND datetime(ic.intime, '+24 hours') AND
    ce.valuenum IS NOT NULL AND
    ce.itemid = 220739
    AND ce.valuenum <= 4
    AND ce.valuenum >= 1
),
gcs_verbal AS (SELECT
    ce.stay_id,
    ce.charttime,
    ce.valuenum AS verbal_value
  FROM
    chartevents ce
  INNER JOIN icustays ic ON ce.stay_id = ic.stay_id
  WHERE
    ce.charttime BETWEEN ic.intime AND datetime(ic.intime, '+24 hours') AND
    ce.valuenum IS NOT NULL AND
    ce.itemid = 223900
    AND ce.valuenum <= 5
    AND ce.valuenum >= 1
),
gcs_motor AS (SELECT
    ce.stay_id,
    ce.charttime,
    ce.valuenum AS motor_value
  FROM
    chartevents ce
  INNER JOIN icustays ic ON ce.stay_id = ic.stay_id
  WHERE
    ce.charttime BETWEEN ic.intime AND datetime(ic.intime, '+24 hours') AND
    ce.valuenum IS NOT NULL AND
    ce.itemid = 223901
    AND ce.valuenum <= 6
    AND ce.valuenum >= 1
)

SELECT
  ic.stay_id,
  eyes.charttime AS eyes_charttime,
  verbal.charttime AS verbal_charttime,
  motor.charttime AS motor_charttime,
  eyes.eyes_value,
  verbal.verbal_value,
  motor.motor_value
FROM
  icustays ic
INNER JOIN gcs_eyes eyes ON ic.stay_id = eyes.stay_id
INNER JOIN gcs_verbal verbal ON ic.stay_id = verbal.stay_id
INNER JOIN gcs_motor motor ON ic.stay_id = motor.stay_id
ORDER BY
  ic.stay_id, eyes_charttime, verbal_charttime, motor_charttime;
