import pickle
import pandas as pd
with open('data_baseline.pickle', 'rb') as f:
    train_df, valid_df, test_df = pickle.load(f)

data = train_df

patients = data['patient_id'].unique()
ICULOS = data['ICULOS'].unique()

index = pd.MultiIndex.from_product([patients, ICULOS], names = ['patient_id', 'ICULOS'])
all_patients_and_ICULOS = pd.DataFrame(index = index).reset_index()

merged = pd.merge(all_patients_and_ICULOS, data,  how='left', left_on=['patient_id', 'ICULOS'], right_on = ['patient_id', 'ICULOS'])

#merged.loc[(merged['date'] == '1986-04-26') & (pd.isna(merged['feature_1'])), 'feature_1'] = 0
#merged.loc[(merged['date'] == '1986-04-26') & (pd.isna(merged['feature_2'])), 'feature_2'] = 0
#merged.loc[(merged['date'] == '1986-04-26') & (pd.isna(merged['feature_3'])), 'feature_3'] = 0
#merged.loc[(merged['date'] == '1986-04-26') & (pd.isna(merged['feature_4'])), 'feature_4'] = 0
#merged = merged.fillna(axis = 0, method = 'ffill')

features = list(set(data.columns)-set(['patient_id', 'ICULOS', 'HospAdmTime']))
reshaped_features = []
for f in features:
    reshaped_features.append(feature = merged['feature_1'].values.reshape(len(patients), len(features)))



feature_1 = merged['feature_1'].values.reshape(95420, 5)
feature_2 = merged['feature_2'].values.reshape(95420, 5)
feature_3 = merged['feature_3'].values.reshape(95420, 5)
feature_4 = merged['feature_4'].values.reshape(95420, 5)