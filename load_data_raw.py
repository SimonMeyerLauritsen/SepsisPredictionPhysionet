import pandas as pd
import pdb
import numpy as np
import glob

# get a list of all the files
files1 = glob.glob('data/training/*.psv')
files2 = glob.glob('data/training_setB/*.psv')
files = np.concatenate((files1, files2))

df_list = []
for ind, f in enumerate(files):
    patient_id = f.split('/')[1].split('.')[0]
    df = pd.read_csv(f, sep='|')
    df = df.assign(patient=patient_id)

    # redefine the labels to be 1 for t >= t_sepsis
    # in other words, a label of 1 now means that sepsis has occurred in this window
    # in practice, what this means is set the first six 1 labels to 0
    df.loc[df[df['SepsisLabel'] == 1].head(6).index.values, 'SepsisLabel'] = 1
    df['patient_id'] = f.split('/')[2].split('.')[0][1:]
    # print a status update
    if ind % 200 == 0:
        print(ind)

    # append the current parsed file to the list
    df_list.append(df)

# save all the loaded files into a pickle file
df = pd.concat(df_list)
df = df.reset_index(drop=True)
df.to_pickle('combined_raw_data.pkl')