import numpy as np
import pandas as pd
import os
from random import seed

seed(1121)

data_path = '/Users/simonlauritsen/PycharmProjects/TF2/data/training/'

patient_id = sorted(os.listdir(data_path))

len_train = round(0.7*len(patient_id))+1
len_val = round(0.15*len(patient_id))
len_test = round(0.15*len(patient_id))
len_train + len_val + len_test == len(patient_id)
len_train

import random
train_id = random.sample(patient_id, len_train)
val_id = random.sample(set(patient_id) - set(train_id), len_val)
test_id = set(patient_id) - set(train_id) - set(val_id)

data_train = '/Users/simonlauritsen/PycharmProjects/TF2/data/raw/training/'
data_val = '/Users/simonlauritsen/PycharmProjects/TF2/data/raw/validation/'
data_test = '/Users/simonlauritsen/PycharmProjects/TF2/data/raw/test/'

for p in train_id:
    df = pd.read_csv(data_path + '/' + p, sep = "|")
    df.to_csv(data_train  + p, sep='|', index = False)

for p in val_id:
    df = pd.read_csv(data_path + '/' + p, sep = "|")
    df.to_csv(data_val + p, sep='|', index = False)

for p in test_id:
    df = pd.read_csv(data_path + '/' + p, sep = "|")
    df.to_csv(data_test + p, sep='|', index = False)

# function to fill missing values
def impute_missing_vals(df, attributes):
    """
    function that imputes missing values.

    @param df: dataframe that has missing values to be
               imputed
           attributes: list of String, attributes of dataframe
    @return df_clean: dataframe without missing values

    """

    """
    fill missing values by the closest values first
    ffill to fill missing values in the tail
    bfill to fill missing values in the head
    """
    # copy df
    df_clean = df.copy()
    for att in attributes:
        if df_clean[att].isnull().sum() == len(df_clean):
            df_clean[att] = df_clean[att].fillna(0)
        elif df_clean[att].isnull().sum() == len(df_clean) - 1:
            df_clean[att] = df_clean[att].ffill().bfill()
        else:
            df_clean[att] = df_clean[att].interpolate(method='nearest', limit_direction='both')
            df_clean[att] = df_clean[att].ffill().bfill()

    return df_clean

# impute missing values and create clean dfs for all patients
for p in patient_id:

    # read in patient data
    df = pd.read_csv(data_path + '/' + p, sep="|")
    attributes = df.columns[:-1]

    # impute missing values
    df_clean = impute_missing_vals(df, attributes)

    # drop unit1 and unit2 with half missing values
    # because these two features have few information
    # drop EtCO2 with all missing values
    df_clean = df_clean.drop(['Unit1', 'Unit2', 'EtCO2'], axis=1)

    # save new patient data
    if p in train_id:
        save_path = '/Users/simonlauritsen/PycharmProjects/TF2/data/processed/train_baseline/'
        df_clean.to_csv(save_path + p, sep='|', index=False)

    elif p in val_id:
        save_path = '/Users/simonlauritsen/PycharmProjects/TF2/data/processed/val_baseline/'
        df_clean.to_csv(save_path + p, sep='|', index=False)

    else:
        save_path = '/Users/simonlauritsen/PycharmProjects/TF2/data/processed/test_baseline/'
        df_clean.to_csv(save_path + p, sep='|', index=False)

    print(p)

import pickle
filename = '/Users/simonlauritsen/PycharmProjects/TF2/data/raw_data.pickle'
with open(filename, 'rb') as f:
    train = pickle.load(f)

td_num = []
for i in range(len(train_id)):
    td_num.append(train_id[i][1:7])

train_concate = train[train.patient_id.isin(td_num)]
train_concate.to_csv('/Users/Chili/Desktop/timeseries/new_project/train_concate.csv', sep = '|', index = False)
pd.read_csv('/Users/Chili/Desktop/timeseries/new_project/train_concate.csv', sep = '|')

train_concate.shape

val_num = []
for i in range(len(val_id)):
    val_num.append(val_id[i][1:7])

val_concate = train[train.patient_id.isin(val_num)]

val_concate.to_csv('/Users/Chili/Desktop/timeseries/new_project/val_concate.csv', sep = '|', index = False)

pd.read_csv('/Users/Chili/Desktop/timeseries/new_project/val_concate.csv', sep = '|')

test_num = []
for i in range(len(test_id)):
    test_num.append(list(test_id)[i][1:7])

test_concate = train[train.patient_id.isin(test_num)]

test_concate.to_csv('/Users/Chili/Desktop/timeseries/new_project/test_concate.csv', sep = '|', index = False)

pd.read_csv('/Users/Chili/Desktop/timeseries/new_project/test_concate.csv', sep = '|')



td=pd.read_csv('/Users/simonlauritsen/PycharmProjects/TF2/data/processed/train_baseline/p000001.psv', sep = '|')

