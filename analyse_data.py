import pickle
import numpy as np
import pandas as pd
import matplotlib
import missingno as msno

# list out lab test features for imputation
labs=['BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate',
      'Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']

# list out vital signal features for imputation
vitals = ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2']

# list out demographic features for imputation
demogs = ['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS']

# labels
labels = ['SepsisLabel']

with open('raw_data.pickle', 'rb') as f:
    data = pickle.load(f)

sepsis_ratio = data.groupby('SepsisLabel').count()['patient_id']
print(sepsis_ratio)
sepsis_ratio = sepsis_ratio[1] / sepsis_ratio.sum()
print(sepsis_ratio)

labs_df = data[labs]
vitals_df = data[vitals]
demogs_df = data[demogs]

np.mean((labs_df.isnull().sum() / labs_df.shape[0]))

np.mean((vitals_df.isnull().sum() / vitals_df.shape[0]))

np.mean((demogs_df.isnull().sum() / demogs_df.shape[0]))

msno.matrix(labs_df)

msno.matrix(vitals_df)

msno.matrix(demogs_df)

missingdata_df = data.columns[data.isnull().any()].tolist()
msno.bar(data[missingdata_df], color="blue", log=False, figsize=(30,18))

msno.heatmap(data[missingdata_df], figsize=(20,20))
