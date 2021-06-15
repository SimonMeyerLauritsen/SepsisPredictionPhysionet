#!/usr/bin/env python

import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras

import pickle
import pandas as pd
import numpy as np
from keras.layers import LSTM
from keras.layers import Dense
from keras.regularizers import l2
from keras.layers import TimeDistributed, TimeDistributed, Bidirectional, BatchNormalization, Dropout, Input, Add, Masking
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

with open('data_baseline.pickle', 'rb') as f:
    train_df, valid_df, test_df = pickle.load(f)

with open('combined_raw_data.pickle', 'rb') as f:
    train_df, valid_df, test_df = pickle.load(f)


# todo: masking
# todo: imputation
# todo: try new model
# todo: sequence evaluation
# todo: split input into sequential and static
# todo: add GPU support

# imputation must be performed before masking

def df_to_tensor(data, unique_id, seq_id, label, exclude, nan_to_num=False):
    patients = data[unique_id].unique()
    ICULOS = data[seq_id].unique()

    index = pd.MultiIndex.from_product([patients, ICULOS], names = [unique_id, seq_id])
    all_patients_and_ICULOS = pd.DataFrame(index = index).reset_index()

    merged = pd.merge(all_patients_and_ICULOS, data,  how='left', left_on=[unique_id, seq_id],
                      right_on = [unique_id, seq_id])

    features = list(set(data.columns)-set([unique_id, seq_id]+exclude+[label]))
    print(set([unique_id, seq_id]+exclude+[label]))
    reshaped_features = []

    label_reshaped = merged[label].values.reshape(len(patients), np.max(merged[seq_id]))
    label_mask = np.where(np.isnan(label_reshaped), True, False)
    label_reshaped = np.expand_dims(label_reshaped, axis=2)


    for f in features:
        print('reshapeing feature: {}'.format(f))
        reshaped_features.append(merged[f].values.reshape(len(patients), np.max(merged[seq_id])))

    data_reshaped = np.hstack(reshaped_features).reshape(len(patients), len(features),
                                                         np.max(merged[seq_id])).transpose(0, 2, 1)

    # convert nans from unequal sequences to number for masking
    if nan_to_num:
        print('Converting nans to {}'.format(nan_to_num))
        label_reshaped = np.nan_to_num(label_reshaped, nan=0)
        data_reshaped = np.nan_to_num(data_reshaped, nan=nan_to_num)
    else:
        print('Keeping nans in files')

    return {'data': data_reshaped, 'labels': label_reshaped, 'n_samples': len(patients), 'n_features': len(features),
            'n_steps': np.max(merged[seq_id]), 'label_mask': label_mask}

data_train = df_to_tensor(data=train_df, unique_id='patient_id', seq_id='ICULOS', label='SepsisLabel', exclude=['HospAdmTime'], nan_to_num=9999)
data_val = df_to_tensor(data=valid_df, unique_id='patient_id', seq_id='ICULOS', label='SepsisLabel', exclude=['HospAdmTime'], nan_to_num=9999)
data_test = df_to_tensor(data=test_df, unique_id='patient_id', seq_id='ICULOS', label='SepsisLabel', exclude=['HospAdmTime'], nan_to_num=9999)


BATCH_SIZE = 32

# define model
inputs1 = Input(shape=(data_train['n_steps'], data_train['n_features']))
model1 =Masking(mask_value=9999,)(inputs1)
model1, state_h, state_c = LSTM(25, return_sequences=True, return_state=True)(inputs1)
model1, state_h, state_c = LSTM(10, return_sequences=True, return_state=True)(model1)
model1 = BatchNormalization()(model1)
model1 = Dense(15, kernel_regularizer=l2(0.001), activation='relu')(model1)
output = Dense(2, kernel_regularizer=l2(0.001), activation='softmax')(model1)

model = Model(inputs=inputs1, outputs=[output])
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='Adam')

checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
history = model.fit([data_train['data']],
                    to_categorical(data_train['labels']),
                    batch_size=BATCH_SIZE,
                    epochs=50,
                    validation_data=([np.nan_to_num(data_val['data'])], to_categorical(data_val['labels'])),
                    callbacks=[earlystop, checkpoint],
                    verbose=1)


import matplotlib.pyplot as plt


plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], '*-')
plt.plot(history.history['val_loss'], '*-')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.show()
plt.savefig('train.png')
plt.close()
plt.clf()

pred = model.predict([np.nan_to_num(data_test['data'])])

y_test = pred[:,:,1].flatten()
y_true = data_test['labels'].flatten().astype(int)
y_mask = data_test['label_mask'].flatten()

y_test = y_test[y_mask==False]
y_true = y_true[y_mask==False]

def roc(y_test, y_true):
    import sklearn.metrics as metrics
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_test)
    roc_auc = metrics.auc(fpr, tpr)

    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.savefig('roc.png')
    plt.clf()
    plt.close()


roc(y_test, y_true)
