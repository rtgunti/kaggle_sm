from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
import os
import zipfile

api = KaggleApi()
api.authenticate()

train_master_csv_file_name = 'train_master.csv'
train_csv_file_name = 'train.csv'
test_csv_file_name = 'test.csv'
n_samles = 30

train_images_path = "/content/train_images/"

if not os.path.isfile(train_master_csv_file_name):
    api.competition_download_file('siim-isic-melanoma-classification',
                               train_master_csv_file_name, path='./')
                
if not os.path.isfile(test_csv_file_name):
    api.competition_download_file('siim-isic-melanoma-classification',
                               test_csv_file_name, path='./')
                        
train_df_master = pd.read_csv(train_master_csv_file_name, sep = ',', header=0)
test_df = pd.read_csv(test_csv_file_name, sep = ',', header=0)

print(train_df_master.target.value_counts())

train_df = train_df_master[train_df_master.target == 0][:n_samles]
train_df = train_df.append(train_df_master[train_df_master.target == 1][:n_samles], ignore_index = True)


train_df.to_csv('train.csv', index = False)

for ind, file in enumerate(train_df.image_name.values):
    print(ind, file)
    if not os.path.exists(train_images_path + file + ".jpg"):
        api.competition_download_file('siim-isic-melanoma-classification', "jpeg/train/" + file + ".jpg", train_images_path)
    zip_file_name = train_images_path + file + ".jpg.zip"
    if os.path.exists(zip_file_name):
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(train_images_path)
            os.remove(zip_file_name)