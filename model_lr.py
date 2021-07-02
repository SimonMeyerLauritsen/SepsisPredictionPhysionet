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
#train_df = concat_patients(train_dir, tr_patients)
#valid_df = concat_patients(valid_dir, vld_patients)
#test_df = concat_patients(test_dir, ts_patients)

# save dataframes to pickle files
#with open('data_baseline.pickle', 'wb') as f:
#    pickle.dump([train_df, valid_df, test_df], f)

with open('data_baseline.pickle', 'rb') as f:
    train_df, valid_df, test_df = pickle.load(f)

# check data missingness.
print('train data has missing values:', train_df.isnull().sum().sum() != 0)
print('valid data has missing values:', valid_df.isnull().sum().sum() != 0)
print('test data has missing values:', test_df.isnull().sum().sum() != 0)

Xtr, ytr = predictors_labels_allocator(train_df)
Xvld, yvld = predictors_labels_allocator(valid_df)
Xts, yts = predictors_labels_allocator(test_df)

# define scaler
scaler = preprocessing.StandardScaler()

# fit and transform data
scaler.fit(Xtr)
Xtr = scaler.transform(Xtr)
Xvld = scaler.transform(Xvld)
Xts = scaler.transform(Xts)

logreg = linear_model.LogisticRegression(C=10, solver='liblinear', max_iter=1000)
logreg.fit(Xtr, ytr)

yhat = logreg.predict(Xvld)

W = logreg.coef_.ravel()
plt.stem(W, use_line_collection=True)

ind = np.argsort(np.abs(W))
for k in range(1, 5):
    i = ind[-k]
    name = train_df.columns[:-2][i]
    print('The {0:d} most significant feature is {1:s}'.format(k, name))

# create a logistic regression model
logreg = linear_model.LogisticRegression(solver='saga', max_iter=1000)

# create hyperparameter search space
# create regularization penalty space
penalty = ['l1', 'l2']

# create regularization hyperparameter space
C = np.logspace(-1, 4, 10)

# create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

# create grid search using 5-fold cross validation
clf = GridSearchCV(logreg, hyperparameters, cv=5, verbose=0)

# best model
best_model = clf.fit(Xtr, ytr)

best_model.best_params_

# predict labels for test data
yhat_ts = best_model.predict(Xts)
# obtain the accuracy on the result
acc_ts = np.mean(yhat_ts == yts)
print('Accuracy on the test data is {0:f}'.format(acc_ts))

# prediction probability
yhat_probas = best_model.predict_proba(Xts)[:,1]

# visualisation with plot_metric
bc = BinaryClassification(yts, yhat_probas, labels=["nonSepsis", "Sepsis"])

# plots
plt.figure(figsize=(15,10))
plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
bc.plot_roc_curve()
plt.subplot2grid((2,6), (0,3), colspan=2)
bc.plot_precision_recall_curve()
plt.show()

# precision, recall, and f1 f2 scores
precision, recall, _ = precision_recall_curve(yts, yhat_probas)
fpr, tpr, _ = roc_curve(yts, yhat_probas)

print('f1 score {0:.4f}:'.format(F(1, np.mean(precision), np.mean(recall))))
print('f2 score {0:.4f}:'.format(F(2, np.mean(precision), np.mean(recall))))
print('precision {0:.4f}:'.format(precision_score(yts, yhat_ts)))
print('recall {0:.4f}:'.format(recall_score(yts, yhat_ts)))
print('AUPRC {0:.4f}:'.format(auc(recall, precision)))
print('AUROC {0:.4f}:'.format(auc(fpr, tpr)))
print('Acc {0:.4f}:'.format(accuracy_score(yts, yhat_ts)))

# report
bc.print_report()