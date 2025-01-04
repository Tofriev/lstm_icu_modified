# script to make functioning tudd data: small dataset without missings for exitus == 1
# stays_others2_ane has no non-missing cases so we only use stays_ane
# strategy: drop the one missing case in stays_ane and undersample for a positive target ratio of 4 %

# %%

import pandas as pd
import sys
import os


# load data
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)
tudd_datapath = os.path.join(project_root, "data/raw/tudd/")
mortality_info_path = os.path.join(tudd_datapath, "stays_ane.csv")
measurements_path = os.path.join(tudd_datapath, "tudd_incomplete.csv")
mortality_info = pd.read_csv(mortality_info_path, sep="|")
measurements = pd.read_csv(measurements_path, sep="|")


# filter
exitus_cases = mortality_info[mortality_info["exitus"] == 1]
non_missing_casids = exitus_cases[exitus_cases["caseid"].isin(measurements["caseid"])]
missing_caseids = exitus_cases[~exitus_cases["caseid"].isin(measurements["caseid"])]
mortality_info = mortality_info[
    ~mortality_info["caseid"].isin(missing_caseids["caseid"])
]
exitus_cases_new = mortality_info[mortality_info["exitus"] == 1]

mortality_info.to_csv(
    os.path.join(tudd_datapath, "stays_ane_new.csv"), sep="|", index=False
)
print(f"non-missing: {len(non_missing_casids)}")
print(f"missing: {len(missing_caseids)}")
print(f"total obervations: {len(mortality_info)}")

print(f"exitus ratio: {len(exitus_cases_new) / len(mortality_info)}")


# %%
