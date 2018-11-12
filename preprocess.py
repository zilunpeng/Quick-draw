import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def get_num_data_per_category(dir_name, is_printing):
    tr_sizes = []
    for filename in os.listdir(dir_name):
        data = pd.read_csv(os.path.join(dir_name,filename))
        tr_sizes.append(data.shape[0])
    if is_printing: print(tr_sizes)

def train_test_split(dir_name,num_folds):
    kf = KFold(n_splits=num_folds,shuffle=True)
    for filename in os.listdir(dir_name):
        path = os.path.join(dir_name,filename)
        data = pd.read_csv(path)
        if data.shape[0] > 0:
            cat_name = os.path.splitext(filename)[0]
            if not os.path.exists(os.path.join('cv_simplified/',cat_name)):
                os.makedirs(os.path.join('cv_simplified/',cat_name))
            i = 1
            for tr_inds, val_inds in kf.split(data):
                tr_data = data.iloc[tr_inds, :]
                val_data = data.iloc[val_inds, :]
                cv_dir_name = 'cv_simplified/'+cat_name+'/'
                tr_data.to_csv(path_or_buf=cv_dir_name+"tr"+str(i)+".csv",columns=['drawing','recognized','word'],index_label='ind')
                val_data.to_csv(path_or_buf=cv_dir_name+"val"+str(i)+".csv",columns=['drawing','recognized','word'],index_label='ind')
                i += 1

#temp = pd.read_csv('cv_simplified/airplane/val1.csv',index_col='ind')
train_test_split('train_simplified',10)
#get_num_data_per_category('train_simplified',True)