#%%
import os
import sys
import pandas as pd

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

tudd = pd.read_csv(os.path.join(project_root, 'data/raw/tudd/tudd_complete.csv'), sep='|')
# %%
print(tudd.head())  
# %%
