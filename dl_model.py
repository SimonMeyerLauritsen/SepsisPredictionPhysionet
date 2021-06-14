import pickle
import pandas as pd
import numpy as np
with open('data_baseline.pickle', 'rb') as f:
    train_df, valid_df, test_df = pickle.load(f)

data = train_df

def df_to_tensor(data, unique_id, seq_id, label, exclude):
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

    for f in features:
        print('reshapeing feature: {}'.format(f))
        reshaped_features.append(merged[f].values.reshape(len(patients), np.max(merged[seq_id])))

    data_reshaped = np.hstack(reshaped_features).reshape(len(patients), len(features),
                                                         np.max(merged[seq_id])).transpose(0, 2, 1)

    return {'data': data_reshaped, 'labels': label_reshaped, 'n_samples': len(patients), 'n_features': len(features),
            'n_steps': np.max(merged[seq_id])}

data_train = df_to_tensor(data=data, unique_id='patient_id', seq_id='ICULOS', label='SepsisLabel', exclude=['HospAdmTime'])
data_val = df_to_tensor(data=valid_df, unique_id='patient_id', seq_id='ICULOS', label='SepsisLabel', exclude=['HospAdmTime'])
data_test = df_to_tensor(data=test_df, unique_id='patient_id', seq_id='ICULOS', label='SepsisLabel', exclude=['HospAdmTime'])

data_res = data_train['data']
labels_res = data_train['labels']
n_samples = data_train['n_samples']
n_features = data_train['n_features']
n_steps = data_train['n_steps']

labels_val = data_val['labels']
labels_test = data_test['labels']

labels_res = np.nan_to_num(labels_res)
labels_val = np.nan_to_num(labels_val)

labels_res = np.expand_dims(labels_res, axis=2)
labels_val = np.expand_dims(labels_val, axis=2)

data_res = np.nan_to_num(data_res)
data_val = np.nan_to_num(data_val)

np.asarray(labels_val).shape

#rod slut

import numpy as np
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import TimeDistributed, TimeDistributed, Bidirectional, BatchNormalization, Dropout, Input, Add, Masking
from tensorflow.keras import Model
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

BATCH_SIZE = 64

# define model
inputs1 = Input(shape=(n_steps, n_features))
model1, state_h, state_c = LSTM(10, return_sequences=True, return_state=True)(inputs1)
output = Dense(2, kernel_regularizer=l2(0.001), activation='softmax')(model1)

model = Model(inputs=inputs1, outputs=[output])
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam')

from tensorflow.keras.utils import to_categorical

checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
history = model.fit([data_res],
                    to_categorical(labels_res),
                    batch_size=BATCH_SIZE,
                    epochs=1,
                    validation_data=([np.nan_to_num(data_val['data'])], to_categorical(labels_val)),
                    callbacks=[earlystop, checkpoint],
                    verbose=1)
#save the history
pickle.dump(history, open('history.pkl', 'wb'))


