# %%
import pandas as pd
import os
import sys
import re

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)


def validate_exitus_data(measurements_file, stays_file, exitus_col):
    """
    Validates the measurements data against stays data.
    The stays file should contain a column with the exitus status.
    For the old stays file, use exitus_col="exitus";
    for the new stays file, use exitus_col="discharged_to_exitus".
    """
    # Load data with pipe delimiter
    measurements = pd.read_csv(measurements_file, sep="|")
    stays = pd.read_csv(stays_file, sep="|")

    # If the specified exitus column is not found, raise an error
    if exitus_col not in stays.columns:
        raise ValueError(f"Column '{exitus_col}' not found in {stays_file}")

    # Convert the exitus column to numeric (non-numeric become NaN)
    stays[exitus_col] = pd.to_numeric(stays[exitus_col], errors="coerce")

    # Filter stays for patients with exitus==1 (patient died)
    exitus_stays = stays[stays[exitus_col] == 1]
    exitus_caseids = exitus_stays["caseid"].unique()

    # Get unique case IDs from the measurements data
    measured_caseids = measurements["caseid"].unique()

    # Identify exitus case IDs missing in the measurements data
    missing_caseids = [cid for cid in exitus_caseids if cid not in measured_caseids]
    total_exitus = len(exitus_caseids)
    missing_percentage = (
        (len(missing_caseids) / total_exitus * 100) if total_exitus else 0
    )

    print(f"--- Validating {measurements_file} using {stays_file} ---")
    print(f"Total exitus (using '{exitus_col}') case IDs in stays: {total_exitus}")
    print(f"Missing case IDs in measurements: {len(missing_caseids)}")
    print(f"Percentage of missing case IDs: {missing_percentage:.2f}%")
    if missing_caseids:
        print("Missing case IDs:", missing_caseids)
    else:
        print("All exitus case IDs have corresponding measurements.")
    print("=" * 50)

    return missing_caseids, missing_percentage


def compare_stays(old_file, new_file):
    """
    Compares the old and new stays files thoroughly.
    The old file uses column 'exitus' while the new uses 'discharged_to_exitus'.
    For the purpose of row-by-row comparisons, the new file's column is renamed to 'exitus'.
    """
    # Load stays files (pipe-delimited)
    old_df = pd.read_csv(old_file, sep="|")
    new_df = pd.read_csv(new_file, sep="|")

    print("=== Comparing Stays Files ===\n")

    # 1. Compare File Shapes
    print("File Shapes:")
    print(f"Old stays file shape: {old_df.shape}")
    print(f"New stays file shape: {new_df.shape}\n")

    # 2. Compare Column Names
    old_columns = set(old_df.columns)
    new_columns = set(new_df.columns)
    print("Column Comparison:")
    print("Columns in old file only:", old_columns - new_columns)
    print("Columns in new file only:", new_columns - old_columns)

    # For exitus-related comparison, align the naming:
    if "discharged_to_exitus" in new_df.columns:
        # Create a copy to avoid modifying original new_df permanently
        new_df = new_df.copy()
        new_df.rename(columns={"discharged_to_exitus": "exitus"}, inplace=True)
        print(
            "Renamed 'discharged_to_exitus' to 'exitus' in new stays file for comparison."
        )
    # Recompute common columns after potential renaming
    common_columns = set(old_df.columns).intersection(new_df.columns)
    print("Common columns:", common_columns, "\n")

    # 3. Compare Unique Case IDs
    print("Case ID Comparison:")
    old_caseids = set(old_df["caseid"].unique())
    new_caseids = set(new_df["caseid"].unique())
    missing_in_new = old_caseids - new_caseids
    extra_in_new = new_caseids - old_caseids
    print("Total unique caseids in old stays file:", len(old_caseids))
    print("Total unique caseids in new stays file:", len(new_caseids))
    print("Caseids missing in new stays file:", missing_in_new)
    print("Extra caseids in new stays file:", extra_in_new, "\n")

    # 4. Row-by-Row Comparison for Common Case IDs
    print("Row-by-Row Differences (based on common caseids):")
    merged = pd.merge(old_df, new_df, on="caseid", suffixes=("_old", "_new"))
    differences = {}
    for col in common_columns:
        if col == "caseid":
            continue
        col_old = col + "_old"
        col_new = col + "_new"
        if col_old in merged.columns and col_new in merged.columns:
            # Replace NaN with a placeholder string to allow proper comparison
            diff = merged[
                merged[col_old].fillna("NaN") != merged[col_new].fillna("NaN")
            ]
            if not diff.empty:
                differences[col] = diff[["caseid", col_old, col_new]]
    if differences:
        for col, diff_df in differences.items():
            print(f"Column: {col}")
            print(diff_df.to_string(index=False))
            print("")
    else:
        print("No differences found in common columns for common caseids.\n")

    # 5. Compare Summary Statistics for Numeric Columns
    print("Summary Statistics for Numeric Columns:")
    numeric_cols = old_df.select_dtypes(include="number").columns.intersection(
        common_columns
    )
    for col in numeric_cols:
        print(f"Column: {col}")
        print("Old stays file stats:")
        print(old_df[col].describe())
        print("New stays file stats:")
        print(new_df[col].describe())
        print("")

    # 6. Compare Missing Values Count per Column
    print("Missing Values Comparison:")
    missing_old = old_df.isnull().sum()
    missing_new = new_df.isnull().sum()
    missing_df = pd.DataFrame({"Old": missing_old, "New": missing_new})
    print(missing_df, "\n")

    # 7. Exitus-Specific Comparison
    print("Exitus-Specific Rows Comparison:")
    # In old file, use column 'exitus'; in new file, after renaming, also use 'exitus'
    old_df["exitus"] = pd.to_numeric(old_df["exitus"], errors="coerce")
    new_df["exitus"] = pd.to_numeric(new_df["exitus"], errors="coerce")
    old_exitus = old_df[old_df["exitus"] == 1]
    new_exitus = new_df[new_df["exitus"] == 1]
    print("Old stays file (exitus==1) rows:", old_exitus.shape[0])
    print("New stays file (exitus==1) rows:", new_exitus.shape[0])
    old_exitus_ids = set(old_exitus["caseid"].unique())
    new_exitus_ids = set(new_exitus["caseid"].unique())
    missing_exitus = old_exitus_ids - new_exitus_ids
    extra_exitus = new_exitus_ids - old_exitus_ids
    print("Exitus caseids missing in new stays file:", missing_exitus)
    print("Extra exitus caseids in new stays file:", extra_exitus)
    print("\n" + "=" * 50)


def validate_caseid_letters_in_new_data(new_stays_file, new_measurements_file):
    """
    Validates the 'caseid' column in the new data files (stays and measurements)
    to check if any caseid contains letters instead of being numeric-only.
    For each file, if a caseid contains letters, the full row is printed.
    """
    pattern = re.compile(r"[A-Za-z]")
    files = {"Stays file": new_stays_file, "Measurements file": new_measurements_file}

    for file_label, file_path in files.items():
        df = pd.read_csv(file_path, sep="|", index_col=False)
        if "caseid" not in df.columns:
            print(f"{file_label} ({file_path}) does not contain a 'caseid' column.")
            continue

        # Convert all caseid values to string and check for alphabetic characters
        invalid_rows = df[
            df["caseid"].astype(str).apply(lambda x: bool(pattern.search(x)))
        ]

        if not invalid_rows.empty:
            print(
                f"In {file_label} ({file_path}), the following rows have a 'caseid' containing letters:"
            )
            print(invalid_rows)
        else:
            print(f"In {file_label} ({file_path}), all caseid values are numeric-only.")


def count_exitus_cases(old_df, new_df):
    """
    Counts and prints the number of exitus cases in the provided old and new datasets.

    Each DataFrame is assumed to have:
      - A 'stay_id' column identifying unique stays.
      - An 'exitus' column, where a value of 1 indicates an exitus case (e.g., death).

    The function groups the data by 'stay_id' (taking the first exitus value per stay)
    and prints the count of exitus cases along with the total number of stays.
    """
    old_df = pd.read_csv(old_df, delimiter="|", index_col=False)
    new_df = pd.read_csv(new_df, delimiter="|", index_col=False)
    print(old_df.head())
    # For each DataFrame, group by stay_id and take the first exitus value
    old_stays = old_df.groupby("caseid").first()
    new_stays = new_df.groupby("caseid").first()

    # Count exitus cases (assuming exitus is binary: 1 for exitus, 0 for non-exitus)
    old_exitus_count = old_stays["exitus"].sum()
    new_exitus_count = new_stays["exitus"].sum()

    # Get total number of unique stays in each dataset
    total_old_stays = len(old_stays)
    total_new_stays = len(new_stays)

    print(f"Old Data: {old_exitus_count} exitus cases out of {total_old_stays} stays.")
    print(f"New Data: {new_exitus_count} exitus cases out of {total_new_stays} stays.")


def compare_unique_caseids(stays_file, measurements_file):
    """
    Compares the number of unique case IDs in the stays file with the number of unique case IDs in the measurements file.
    """
    # Load data
    stays = pd.read_csv(stays_file, sep="|", index_col=False)
    measurements = pd.read_csv(measurements_file, sep="|", index_col=False)

    # Get unique case IDs
    unique_caseids_stays = stays["caseid"].nunique()
    unique_caseids_measurements = measurements["caseid"].nunique()

    print(f"Unique case IDs in stays: {unique_caseids_stays}")
    print(f"Unique case IDs in measurements: {unique_caseids_measurements}")


import pandas as pd


def find_missing_caseids(stays_file, measurements_file):
    """
    Finds case IDs that are in the stays file but not in the measurements file.
    """
    # Load data
    stays = pd.read_csv(stays_file, sep="|", index_col=False)
    measurements = pd.read_csv(measurements_file, sep="|", index_col=False)

    # Get unique case IDs
    stays_caseids = set(stays["caseid"].unique())
    measurements_caseids = set(measurements["caseid"].unique())

    # Find missing case IDs
    missing_caseids = stays_caseids - measurements_caseids

    print(
        f"Number of case IDs in stays but not in measurements: {len(missing_caseids)}"
    )
    print("Missing case IDs:", missing_caseids)

    return missing_caseids


if __name__ == "__main__":
    # File paths: adjust as needed
    print(os.getcwd())
    tudd_datapath = os.path.join(project_root, "data/raw/tudd/")
    old_measurements_file = os.path.join(tudd_datapath, "tudd_incomplete.csv")
    new_measurements_file = os.path.join(tudd_datapath, "measurement.csv")
    old_stays_file = os.path.join(tudd_datapath, "stays_others2_ane.csv")
    new_stays_file = os.path.join(tudd_datapath, "stays.csv")

    compare_unique_caseids(new_stays_file, new_measurements_file)
    find_missing_caseids(new_stays_file, new_measurements_file)

    # count_exitus_cases(old_stays_file, new_stays_file)
    # validate_caseid_letters_in_new_data(new_stays_file, new_measurements_file)

    # print("=== Validating Measurements Data ===\n")
    # print("Validating OLD measurements data:")
    # old_missing, old_percentage = validate_exitus_data(
    #     old_measurements_file, old_stays_file, "exitus"
    # )

    # print("Validating NEW measurements data:")
    # new_missing, new_percentage = validate_exitus_data(
    #     new_measurements_file, new_stays_file, "exitus"
    # )

    # if new_percentage == 0:
    #     print("The missing case IDs issue is fixed in the new measurements data.\n")
    # else:
    #     print(
    #         "The missing case IDs issue still persists in the new measurements data.\n"
    #     )

    # print("=== Comparing Old and New Stays Files ===\n")
    # compare_stays(old_stays_file, new_stays_file)
# %%
