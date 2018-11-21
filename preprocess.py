import ast
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

def get_most_cv_tr_size(dir_name):
    max_size = 0
    for cat_name in os.listdir(dir_name):
        data = pd.read_pickle(dir_name + '/' + cat_name + '/tr1.pkl')
        data = data.shape[0]
        if data > max_size: max_size=data
    print('maximum training data size in cv_simplified is ' + str(max_size))

def train_test_split(dir_name,num_folds):
    kf = KFold(n_splits=num_folds,shuffle=True)
    for filename in os.listdir(dir_name):
        cat_name = os.path.splitext(filename)[0]
        print('creating cv files for ' + cat_name)
        if not os.path.exists(os.path.join('cv_simplified/', cat_name)):
            path = os.path.join(dir_name,filename)
            data = pd.read_csv(path)
            drawings = list(map(ast.literal_eval, data['drawing']))
            data['drawing'] = drawings
            data = data.drop(columns=['countrycode','key_id','timestamp',])
            if data.shape[0] > 0:
                os.makedirs(os.path.join('cv_simplified/',cat_name))
                i = 1
                for tr_inds, val_inds in kf.split(data):
                    tr_data = data.iloc[tr_inds, :]
                    tr_data = tr_data.reset_index()
                    val_data = data.iloc[val_inds, :]
                    val_data = val_data.reset_index()
                    cv_dir_name = 'cv_simplified/'+cat_name+'/'
                    tr_data.to_pickle(path=cv_dir_name+"tr"+str(i)+".pkl")
                    val_data.to_pickle(path=cv_dir_name+"val"+str(i)+".pkl")
                    i += 1
        else:
            print(cat_name + ' already exists')

get_most_cv_tr_size('cv_simplified')
#train_test_split('train_simplified',10)
