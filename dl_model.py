
# todo: split dynamic and static features
# todo: try new model
# todo: sequence evaluation
# todo: split input into sequential and static
# todo: new imputation method

import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Masking, Add
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

raw_df = pd.read_pickle('combined_raw_data.pkl')

def scale_data(data, scaler=False, exc_cols=False, fill_na=False):
    data_exc_cols = data[exc_cols]
    data = data.drop(columns=exc_cols)
    columns = data.columns
    if not scaler:
        scaler = preprocessing.StandardScaler()
        scaler.fit(data)
        data = pd.DataFrame(scaler.transform(data), columns=columns)
        if fill_na:
            data = data.fillna(0)
        return pd.concat([data, data_exc_cols], axis=1, join='inner'), scaler
    else:
        data = pd.DataFrame(scaler.transform(data), columns=columns)
        if fill_na:
            data = data.fillna(0)
        return pd.concat([data, data_exc_cols], axis=1, join='inner')
raw_df_scaled, scaler = scale_data(raw_df, False, ['patient', 'patient_id', 'ICULOS', 'SepsisLabel'], True)

def convert_raw_to_folds(df):
    """
    Function to split det complete dataset into folds.
    :param df: Complete dataset (Pandas Dataframe)
    :return: train, test and val folds (Pandas Dataframe)
    """
    u, indices = np.unique(df['patient_id'], return_index=True)
    len_train = round(0.7*len(u))+1
    len_val = round(0.15*len(u))
    len_test = round(0.15*len(u))
    len_train + len_val + len_test == len(u)
    import random
    train_id = random.sample(set(u), len_train)
    val_id = random.sample(set(u) - set(train_id), len_val)
    test_id = set(u) - set(train_id) - set(val_id)
    data_train = df.loc[df['patient_id'].isin(train_id)]
    data_val = df.loc[df['patient_id'].isin(val_id)]
    data_test = df.loc[df['patient_id'].isin(test_id)]
    return data_train, data_val, data_test
data_train, data_val, data_test = convert_raw_to_folds(raw_df_scaled)

def df_to_tensor(data, unique_id, seq_id, label, exclude, nan_to_num=False, static=False):
    """
       Text
       :param df: Complete dataset (Pandas Dataframe)
       :return: train, test and val folds (Pandas Dataframe)
    """
    patients = data[unique_id].unique()
    seq = data[seq_id].unique()

    index = pd.MultiIndex.from_product([patients, seq], names = [unique_id, seq_id])
    all_patients_and_seq = pd.DataFrame(index = index).reset_index()

    merged = pd.merge(all_patients_and_seq, data,  how='left', left_on=[unique_id, seq_id],
                      right_on = [unique_id, seq_id])

    features = list(set(data.columns)-set([unique_id, seq_id]+exclude+[label]))
    print(set([unique_id, seq_id]+exclude+[label]))
    reshaped_features = []

    label_reshaped = merged[label].values.reshape(len(patients), np.max(merged[seq_id]))
    label_mask = np.where(np.isnan(label_reshaped), True, False)
    label_reshaped = np.expand_dims(label_reshaped, axis=2)

    static_features = []
    for f in features:
        print('reshapeing feature: {}'.format(f))
        if static:
            if f in static:
                static_features.append(merged[f].values.reshape(len(patients), np.max(merged[seq_id])))
        reshaped_features.append(merged[f].values.reshape(len(patients), np.max(merged[seq_id])))

    data_reshaped = np.hstack(reshaped_features).reshape(len(patients), len(features),
                                                         np.max(merged[seq_id])).transpose(0, 2, 1)
    # todo: features should be replicated along time steps
    static_features = np.reshape(np.nanmax(np.vstack(static_features), axis=1), (len(patients), len(static)))
    static_features = np.moveaxis(np.repeat(static_features[:, :, np.newaxis], 336, axis=2), 1, 2)

    # convert nans from unequal sequences to number for masking
    if nan_to_num:
        print('Converting nans to {}'.format(nan_to_num))
        label_reshaped = np.nan_to_num(label_reshaped, nan=0)
        data_reshaped = np.nan_to_num(data_reshaped, nan=nan_to_num)
    else:
        print('Keeping nans in files')

    return {'data': data_reshaped, 'data_static': static_features, 'labels': label_reshaped, 'n_samples': len(patients), 'n_features': len(features),
            'n_steps': np.max(merged[seq_id]), 'label_mask': label_mask, 'feratures': features}
data_train = df_to_tensor(data=data_train, unique_id='patient_id', seq_id='ICULOS', label='SepsisLabel', exclude=['HospAdmTime', 'patient', 'Unit1', 'Unit2'], nan_to_num=9999, static=['Gender', 'Age'])
data_val = df_to_tensor(data=data_val, unique_id='patient_id', seq_id='ICULOS', label='SepsisLabel', exclude=['HospAdmTime', 'patient', 'Unit1', 'Unit2'], nan_to_num=9999,static=['Gender', 'Age'])
data_test = df_to_tensor(data=data_test, unique_id='patient_id', seq_id='ICULOS', label='SepsisLabel', exclude=['HospAdmTime', 'patient', 'Unit1', 'Unit2'], nan_to_num=9999, static=['Gender', 'Age'])

# define model
BATCH_SIZE = 128
# dynamic layers
inputs1 = Input(shape=(data_train['n_steps'], data_train['n_features']))
model1 = Masking(mask_value=9999,)(inputs1)
model1, state_h = GRU(40, return_sequences=True, return_state=True)(inputs1)
model1, state_h = GRU(40, return_sequences=True, return_state=True)(model1)
model1 = Dense(25, kernel_regularizer=l2(0.001), activation='relu')(model1)
# static layers
input2 = Input(shape=(data_train['n_steps'], 2,)) # todo: skal laves dynamisk
model2 = Dense(25, kernel_regularizer=l2(0.001), activation='relu')(input2)
# combine dynamic and static layers
model_add = Add()([model1, model2])
output = Dense(2, kernel_regularizer=l2(0.001), activation='softmax')(model_add)

model = Model(inputs=[inputs1,input2], outputs=[output])
print(model.summary())
opt = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt)

checkpoint = ModelCheckpoint('model_GRU.h5', monitor='val_loss', verbose=2, save_best_only=True, mode='auto', save_freq=1000)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
history = model.fit([data_train['data'], data_train['data_static']],
                    to_categorical(data_train['labels']),
                    batch_size=BATCH_SIZE,
                    epochs=25,
                    validation_data=([data_val['data'], data_val['data_static']], to_categorical(data_val['labels'])),
                    callbacks=[earlystop, checkpoint],
                    verbose=1)

def plot_loss(hist):
    plt.figure(figsize=(10,6))
    plt.plot(hist.history['loss'], '*-')
    plt.plot(hist.history['val_loss'], '*-')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.savefig('loss.png')
    plt.clf()
    plt.close()
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
    #plt.clf()
    #plt.close()
def flatten_preds(pred, data):
    y_test = pred[:,:,1].flatten()
    y_true = data['labels'].flatten().astype(int)
    y_mask = data['label_mask'].flatten()
    return y_test, y_true, y_mask

plot_loss(history)
pred = model.predict([data_test['data'], data_test['data_static']])
y_test, y_true, y_mask = flatten_preds(pred, data_test)

y_test = y_test[y_mask==False]
y_true = y_true[y_mask==False]
roc(y_test, y_true)

