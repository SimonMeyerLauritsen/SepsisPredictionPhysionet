def predictors_labels_allocator(df):
    """
    function that allocate predictors and labels

    @param: df: Dataframe, complete dataframe
    @return: X: Numpy Array, predictors
             y: Numpy Array, labels
    """
    col_names = df.columns
    X = np.array(df[col_names[:-2]].values)
    y = df[col_names[-2]].values

    return X, y


def F(beta, precision, recall):
    """
    Function that calculate f1, f2, and f0.5 scores.

    @params: beta, Float, type of f score
             precision: Float, average precision
             recall: Float, average recall

    @return: Float, f scores
    """

    return (beta * beta + 1) * precision * recall / (beta * beta * precision + recall)


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
    return p

# import packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import random
import dill    # save or restore notebook session

from sklearn import linear_model, preprocessing
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from plot_metric.functions import BinaryClassification

#create a list of patient file names
train_dir = './data/baseline/train_baseline/'
valid_dir = './data/baseline/val_baseline/'
test_dir = './data/baseline/test_baseline/'

tr_patients = [p for p in sorted(os.listdir(train_dir))]
vld_patients = [p for p in sorted(os.listdir(valid_dir))]
ts_patients = [p for p in sorted(os.listdir(test_dir))]

print('num training patients:', len(tr_patients))
print('num valid patients:', len(vld_patients))
print('num test patients:', len(ts_patients))

import pickle
# concate patients
train_df = concat_patients(train_dir, tr_patients)
valid_df = concat_patients(valid_dir, vld_patients)
test_df = concat_patients(test_dir, ts_patients)

# # save dataframes to pickle files
# with open('data_baseline.pickle', 'wb') as f:
#     pickle.dump([train_df, valid_df, test_df], f)

with open('data_baseline.pickle', 'rb') as f:
    train_df, valid_df, test_df = pickle.load(f)