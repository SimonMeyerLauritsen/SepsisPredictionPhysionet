import pickle
from sklearn import linear_model, preprocessing

from pycaret.classification import *
import numpy as np
import os
os.chdir("/Users/simonlauritsen/PycharmProjects/TF2/")
import pandas as pd
data_n = pd.read_csv('set_n.csv', index_col=0).iloc[1: , :]
data_t = pd.read_csv('set_t.csv', index_col=0).iloc[1: , :]
data_d = pd.read_csv('set_d.csv', index_col=0).iloc[1: , :]
data_r = pd.read_csv('set_r.csv', index_col=0).iloc[1: , :]

# https://pycaret.org/classification/
#bin_numeric_features: list, default = None

features = ['Temperature', 'SpO2', 'Heart rate', 'Diastolic BP',
            'Respiratory Frequency', 'Systolic BP', 'B-Leukocytes', 'B-Neutrophils',
            'B-Platelets', 'eGFR', 'P-Albumin', 'P-Bilirubine',
            'P-C-reactive protein', 'P-Glucose', 'P-Potassium', 'P-Creatinine',
            'P-Sodium', 'P(aB)-Hydrogen carbonate', 'P(aB)-Potassium', 'P(aB)-Chloride',
            'P(aB)-Lactate', 'P(aB)-Sodium', 'P(aB)-pCO2', 'P(aB)-pH', 'P(aB)-pO2']


os.chdir("/Users/simonlauritsen/PycharmProjects/TF2/model_n_noint")
exp_n = setup(data=data_n, target = 'label', ignore_features = [], session_id=1,
              log_experiment=False, log_profile=False, log_plots=False, )
model_n = create_model('xgboost', cross_validation=False)
plot_model(model_n, 'pr', save=True)
plot_model(model_n, 'auc', save=True)
plot_model(model_n, 'calibration', save=True)
#plot_model(model_n, 'threshold', save=True)
plot_model(model_n, 'feature', save=True)
save_model(model_n, 'model_n')

os.chdir("/Users/simonlauritsen/PycharmProjects/TF2/model_t_noint")
exp_t = setup(data=data_t, target = 'label', ignore_features = [], session_id=2)
              #log_experiment=True, log_profile=True, log_plots=True, )
model_t = create_model('xgboost', cross_validation=False)
plot_model(model_t, 'pr', save=True)
plot_model(model_t, 'auc', save=True)
plot_model(model_t, 'calibration', save=True)
#plot_model(model_t, 'threshold', save=True)
plot_model(model_t, 'feature', save=True)
save_model(model_t, 'model_t')

os.chdir("/Users/simonlauritsen/PycharmProjects/TF2/model_d_noint")
exp_d = setup(data=data_d, target = 'label', ignore_features = [], session_id=3)
              # log_experiment=True, log_profile=True, log_plots=True, )
model_d = create_model('xgboost', cross_validation=False)
plot_model(model_d, 'pr', save=True)
plot_model(model_d, 'auc', save=True)
plot_model(model_d, 'calibration', save=True)
#plot_model(model_d, 'threshold', save=True)
plot_model(model_d, 'feature', save=True)
save_model(model_d, 'model_d')

os.chdir("/Users/simonlauritsen/PycharmProjects/TF2/model_r_noint")
exp_r = setup(data=data_r, target = 'label', ignore_features = [], session_id=4)
              #log_experiment=True, log_profile=True, log_plots=True)
model_r = create_model('xgboost', cross_validation=False)
plot_model(model_r, 'pr', save=True)
plot_model(model_r, 'auc', save=True)
plot_model(model_r, 'calibration', save=True)
#plot_model(model_r, 'threshold', save=True)
plot_model(model_r, 'feature', save=True)
save_model(model_r, 'model_r')

calibrated_model_n = calibrate_model(model_n, method='isotonic')
plot_model(calibrated_model_n, 'calibration', save=True)

calibrated_model_t = calibrate_model(model_t, method='isotonic')
plot_model(model_t, 'calibration', save=True)

calibrated_model_d = calibrate_model(model_d, method='isotonic')
plot_model(calibrated_model_d, 'calibration', save=True)

calibrated_model_r = calibrate_model(model_r, method='isotonic')
plot_model(model_r, 'calibration', save=True)

# stacking
os.chdir("/Users/simonlauritsen/PycharmProjects/TF2/model_n_stacking")
exp_n = setup(data=data_n, target = 'label', ignore_features = [], session_id=1,
              log_experiment=False, log_profile=False, log_plots=False, )
top5 = compare_models(n_select = 5)
stacker = stack_models(estimator_list = top5[1:], meta_model = top5[0])
plot_model(stacker, 'pr', save=True)
plot_model(stacker, 'auc', save=True)
plot_model(stacker, 'calibration', save=True)
save_model(stacker, 'model_n')
os.chdir("/Users/simonlauritsen/PycharmProjects/TF2/model_n_blending")
blender = blend_models(estimator_list=top5)
plot_model(blender, 'pr', save=True)
plot_model(blender, 'auc', save=True)
plot_model(blender, 'calibration', save=True)
save_model(blender, 'model_n')

os.chdir("/Users/simonlauritsen/PycharmProjects/TF2/model_t_stacking")
exp_t = setup(data=data_t, target = 'label', ignore_features = [], session_id=1,
              log_experiment=False, log_profile=False, log_plots=False, )
top5 = compare_models(n_select = 5)
stacker = stack_models(estimator_list = top5[1:], meta_model = top5[0])
plot_model(stacker, 'pr', save=True)
plot_model(stacker, 'auc', save=True)
plot_model(stacker, 'calibration', save=True)
save_model(stacker, 'model_t')
os.chdir("/Users/simonlauritsen/PycharmProjects/TF2/model_t_blending")
blender = blend_models(estimator_list=top5)
plot_model(blender, 'pr', save=True)
plot_model(blender, 'auc', save=True)
plot_model(blender, 'calibration', save=True)
save_model(blender, 'model_t')

os.chdir("/Users/simonlauritsen/PycharmProjects/TF2/model_d_stacking")
exp_r = setup(data=data_d, target = 'label', ignore_features = [], session_id=1,
              log_experiment=False, log_profile=False, log_plots=False, )
top5 = compare_models(n_select = 5)
stacker = stack_models(estimator_list = top5[1:], meta_model = top5[0])
plot_model(stacker, 'pr', save=True)
plot_model(stacker, 'auc', save=True)
plot_model(stacker, 'calibration', save=True)
save_model(stacker, 'model_d')
os.chdir("/Users/simonlauritsen/PycharmProjects/TF2/model_d_blending")
blender = blend_models(estimator_list=top5)
plot_model(blender, 'pr', save=True)
plot_model(blender, 'auc', save=True)
plot_model(blender, 'calibration', save=True)
save_model(blender, 'model_d')

os.chdir("/Users/simonlauritsen/PycharmProjects/TF2/model_r_stacking")
exp_r = setup(data=data_r, target = 'label', ignore_features = [], session_id=1,
              log_experiment=False, log_profile=False, log_plots=False, )
top5 = compare_models(n_select = 5)
stacker = stack_models(estimator_list = top5[1:], meta_model = top5[0])
plot_model(stacker, 'pr', save=True)
plot_model(stacker, 'auc', save=True)
plot_model(stacker, 'calibration', save=True)
save_model(stacker, 'model_r')
os.chdir("/Users/simonlauritsen/PycharmProjects/TF2/model_r_blending")
blender = blend_models(estimator_list=top5)
plot_model(blender, 'pr', save=True)
plot_model(blender, 'auc', save=True)
plot_model(blender, 'calibration', save=True)
save_model(blender, 'model_r')


os.chdir("/Users/simonlauritsen/PycharmProjects/TF2/")
blender = load_model('model_n')
plot_model(blender, 'pr', save=True)
plot_model(blender, 'auc', save=True)
plot_model(blender, 'calibration', save=True)