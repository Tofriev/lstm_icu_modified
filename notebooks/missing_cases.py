# %%
# script to extract the missing measurements
import pandas as pd
import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

tudd_datapath = os.path.join(project_root, "data/raw/tudd/")
mortality_info_x_path = os.path.join(tudd_datapath, "stays_ane.csv")
mortality_info_y_path = os.path.join(tudd_datapath, "stays_others2_ane.csv")
measurements_path = os.path.join(project_root, "data/old/tudd_incomplete.csv")


mortality_info_list = []
for path in [mortality_info_x_path, mortality_info_y_path]:
    if os.path.exists(path):
        mortality_info_list.append(pd.read_csv(path, sep="|"))
m_info_complete = pd.concat(mortality_info_list, ignore_index=True)

measurements = pd.read_csv(measurements_path, sep="|")

exitus_cases = m_info_complete[m_info_complete["exitus"] == 1]

non_missing_casids = exitus_cases[exitus_cases["caseid"].isin(measurements["caseid"])]
missing_caseids = exitus_cases[~exitus_cases["caseid"].isin(measurements["caseid"])]
print(len(non_missing_casids))
print(len(missing_caseids))
missing_caseids["caseid"].to_csv("missing_casesids.csv", index=False)
non_missing_casids["caseid"].to_csv("non_missing_caseids.csv", index=False)

# %%
