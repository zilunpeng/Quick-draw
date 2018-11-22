import ast
import os
import pandas as pd
import numpy as np
import build_feature
from sklearn.model_selection import KFold
from multiprocessing import Pool

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

def train_test_split(dir_name,num_folds,num_workers):
    kf = KFold(n_splits=num_folds,shuffle=True)
    for filename in os.listdir(dir_name):
        cat_name = os.path.splitext(filename)[0]
        print('creating cv files for ' + cat_name)
        if not os.path.exists(os.path.join('cv_simplified',cat_name,'tr1.npy')):
            data = pd.read_csv(os.path.join(dir_name,filename))
            drawings = list(map(ast.literal_eval, data['drawing']))
            with Pool(num_workers) as p:
                data['drawing'] = p.map(build_feature.set_feature_mat, drawings)
            data = data.drop(columns=['countrycode','key_id','timestamp',])
            if data.shape[0] > 0:
                i = 1
                for tr_inds, val_inds in kf.split(data):
                    cv_dir_name = 'cv_simplified/' + cat_name + '/'
                    tr_data = data.iloc[tr_inds, :]
                    np.save(cv_dir_name+"tr"+str(i)+".npy", tr_data['drawing'].values)

                    val_data = data.iloc[val_inds, :]
                    val_data = val_data.reset_index()
                    np.save(cv_dir_name+"val"+str(i)+".npy", val_data[['drawing','index']].values)
                    i += 1
            print('finished creating '+cat_name)
            break
        else:
            print(cat_name + ' already exists')

#get_most_cv_tr_size('cv_simplified')
train_test_split('train_simplified',10,12)
