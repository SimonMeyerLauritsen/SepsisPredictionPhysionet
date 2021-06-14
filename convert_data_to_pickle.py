import pickle
import modin.pandas as pd
import numpy as np
import os

def concat_patients(train_dir, patient_list):
    """
    concatenate individual patient dataframe for
    the training, valid, or other dataframes to define
    lower and upper bound of each feature.

    @params: train_dir: String, data folder
             patient_list: list of Strings, individual patient files

    @return: dataframe, concatenated training or valid data
    """

    # read in first patient in patient list
    p = pd.read_csv(train_dir + '/' + patient_list[0], sep="|")
    p['patient_id'] = patient_list[0][1:7]

    for i in range(1, len(patient_list)):
        p_n = pd.read_csv(train_dir + '/' + patient_list[i], sep="|")
        p_n['patient_id'] = patient_list[i][1:7]
        p = p.append(p_n)
        print(p['patient_id'])
    return p

#create a list of patient file names
train_dir = './data/raw/training/'
valid_dir = './data/raw/validation/'
test_dir = './data/raw/test/'

tr_patients = [p for p in sorted(os.listdir(train_dir))]
vld_patients = [p for p in sorted(os.listdir(valid_dir))]
ts_patients = [p for p in sorted(os.listdir(test_dir))]

print('num training patients:', len(tr_patients))
print('num valid patients:', len(vld_patients))
print('num test patients:', len(ts_patients))

# concate patients
train_df = concat_patients(train_dir, tr_patients)
valid_df = concat_patients(valid_dir, vld_patients)
test_df = concat_patients(test_dir, ts_patients)

# save dataframes to pickle files
with open('data_raw.pickle', 'wb') as f:
    pickle.dump([train_df, valid_df, test_df], f)