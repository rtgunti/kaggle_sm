from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
import os

api = KaggleApi()
api.authenticate()

train_master_csv_file_name = 'train_master.csv'
train_csv_file_name = 'train.csv'
test_csv_file_name = 'test.csv'

if not os.path.isfile(train_master_csv_file_name):
    api.competition_download_file('siim-isic-melanoma-classification',
                               train_master_csv_file_name, path='./')
                
if not os.path.isfile(test_csv_file_name):
    api.competition_download_file('siim-isic-melanoma-classification',
                               test_csv_file_name, path='./')
                        
train_df_master = pd.read_csv(train_master_csv_file_name, sep = ',', header=0)
test_df = pd.read_csv(test_csv_file_name, sep = ',', header=0)

print(train_df_master.target.value_counts())

train_df = train_df_master[train_df_master.target == 0][:500]
train_df = train_df.append(train_df_master[train_df_master.target == 1][:500], ignore_index = True)

train_df.to_csv('train.csv', index = False)