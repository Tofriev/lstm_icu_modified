variables = {
    "mbp": {"type": "numerical", "training": True},  # Mean Blood Pressure
    "gcs_total": {"type": "numerical", "training": True},  # Glasgow Coma Scale Total
    "glc": {"type": "numerical", "training": True},  # Glucose levels
    # "resprate_mortality": {"type": "numerical", "training": True},  # Respiratory Rate TODO: extract resprate. we dont need mortality here anymore
    "creatinine": {"type": "numerical", "training": True},  # Creatinine levels
    "potassium": {"type": "numerical", "training": True},  # Potassium levels
    "hr": {"type": "numerical", "training": True},  # Heart Rate
    # "sodium": {"type": "numerical", "training": True},  # Sodium levels
    "wbc": {"type": "numerical", "training": True},  # White Blood Cells (leukocytes)
    "platelets": {"type": "numerical", "training": True},  # Platelets (thrombocytes)
    "inr": {
        "type": "numerical",
        "training": True,
    },  # International Normalized Ratio (Prothrombin Time)
    "anion_gap": {"type": "numerical", "training": True},  # Anion Gap
    "lactate": {"type": "numerical", "training": True},  # Lactate levels
    # "urea": {"type": "numerical", "training": True},  # Urea levels
    "temperature": {"type": "numerical", "training": True},  # Body Temperature
    "weight": {"type": "numerical", "training": True},  # Weight over time (time series)
    "static_data": {
        "mortality": {"type": "target", "training": False},  # Mortality outcome
        "age": {"type": "numerical", "training": True},  # Age
        "gender": {"type": "categorical", "training": True},  # Gender
        #'height': {'type': 'numerical', 'training': True},     # Height
        "intime": {"type": "datetime", "training": False},
        "first_day_end": {"type": "datetime", "training": False},
        "stay_id": {"type": "id", "training": False},
    },
}


def count_features(variables):
    exclude_keys = {"mortality", "intime", "first_day_end", "stay_id", "static_data"}

    top_level_keys = [key for key in variables if key not in exclude_keys]
    count = len(top_level_keys)
    count += len([key for key in variables["static_data"] if key not in exclude_keys])

    return count

def count_sequential_and_static_features(variables):
    # Sequential features: top-level keys except "static_data"
    sequential_keys = [key for key in variables if key != "static_data"]
    
    # Static features: keys in "static_data" excluding target and meta keys.
    exclude_static = {"mortality", "intime", "first_day_end", "stay_id"}
    static_keys = [key for key in variables.get("static_data", {}) if key not in exclude_static]
    
    return len(sequential_keys), len(static_keys)
