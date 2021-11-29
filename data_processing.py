import zipfile
import shutil
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import wandb
from PIL import Image

path_reef = 'E:/kaggle_great_barrier_reef/Great_Barrier_Reef/'
zip_file = 'tensorflow-great-barrier-reef.zip'
data_path = os.path.join(path_reef, 'data_reef')

# with zipfile.ZipFile(path_reef + zip_file, 'r') as zip_ref:
#     zip_ref.extractall(data_path)


# Training and Testing dataframes
train_csv = pd.read_csv(data_path + '/train.csv')
test_csv = pd.read_csv(data_path + '/test.csv')

# Length of the training and testing data
print(len(train_csv))
print(len(test_csv))

# Dataframe head
print(train_csv.head(200))

# Frames with starfish
train_real = train_csv.loc[train_csv["annotations"] != "[]"]

# Dataframe head
print(train_real.head(100))


# Feature Summary - copied from Diego Gomez
def resumetable(df):
    '''function to create feature summary'''
    print(f'Shape: {df.shape}')
    summary = pd.DataFrame(df.dtypes, columns=['Data Type'])
    summary = summary.reset_index()
    summary = summary.rename(columns={'index': 'Features'})
    summary['Num of Null Value'] = df.isnull().sum().values
    summary['Num of Unique Value'] = df.nunique().values
    summary['1st Value'] = df.loc[0].values
    summary['2nd Value'] = df.loc[1].values
    summary['3rd Value'] = df.loc[2].values
    return summary

resumetable(train_csv)