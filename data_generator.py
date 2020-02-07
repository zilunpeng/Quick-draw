import os
import pandas as pd
import numpy as np
import h5py
from build_feature import set_feature_mat
import ast
import argparse
import cv2

def sample_and_compose_data(dir_loc, num_tr_data_to_sample, num_val_data_to_sample, save_loc):
    tr_data = pd.DataFrame()
    val_data = pd.DataFrame()
    cat_ind = 0
    for cat_name in os.listdir(dir_loc):
        if cat_name != '.DS_Store':
            data = pd.read_pickle(os.path.join(dir_loc, cat_name+'/tr1.pkl'))
            data = data.sample(n=num_tr_data_to_sample, replace=False)
            data['cat_ind'] = np.ones(num_tr_data_to_sample, dtype=int) * cat_ind
            tr_data = tr_data.append(data)
            data = pd.read_pickle(os.path.join(dir_loc, cat_name + '/val1.pkl'))
            data = data.sample(n=num_val_data_to_sample, replace=False)
            data['cat_ind'] = np.ones(num_val_data_to_sample, dtype=int)*cat_ind
            val_data = val_data.append(data)
            cat_ind += 1

    print('tr data shape', tr_data.shape)
    print('val data shape', val_data.shape)

    tr_data = tr_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    tr_data = tr_data.drop(columns=['index'])
    val_data = val_data.drop(columns=['index'])
    tr_data.to_pickle(os.path.join(save_loc, 'tr.pkl'))
    val_data.to_pickle(os.path.join(save_loc, 'val.pkl'))

# For all 340 categories, sample and process, then store in h5py
def sample_from_all_data(data_path, saving_path, num_train_data_to_sample_in_each_cat, num_val_data_to_sample_in_each_cat, part):
    file_handler = h5py.File(os.path.join(saving_path,'quick_draw_data_part_'+str(part)+'.hdf5'), 'w')
    num_cats = 113
    num_feature = 851968
    tr_data_pointer = file_handler.create_dataset('tr_data', (num_cats * num_train_data_to_sample_in_each_cat, num_feature), dtype='bool',
                                                         chunks=(1, num_feature), compression="gzip", compression_opts=4)
    val_data_pointer = file_handler.create_dataset('val_data', (num_cats * num_val_data_to_sample_in_each_cat, num_feature), dtype='bool',
                                                      chunks=(1, num_feature), compression="gzip", compression_opts=4)

    tr_label_pointer = file_handler.create_dataset('tr_label', (num_cats * num_train_data_to_sample_in_each_cat, 1), dtype='int64',
                                                        chunks=(1, 1), compression="gzip", compression_opts=4)
    val_label_pointer = file_handler.create_dataset('val_label', (num_cats * num_val_data_to_sample_in_each_cat, 1), dtype='int64',
                                                     chunks=(1, 1), compression="gzip", compression_opts=4)

    cur_tr_ind = 0
    cur_val_ind = 0
    num_data_to_sample_in_cat = num_train_data_to_sample_in_each_cat + num_val_data_to_sample_in_each_cat
    file_list = os.listdir(data_path)
    if part == 1:
        file_inds = range(113)
    elif part == 2:
        file_inds = range(113, 226)
    elif part == 3:
        file_inds = range(226, 340)

    for file_ind in file_inds:
        cat_name = file_list[file_ind]
        data_in_cur_cat = pd.read_csv(os.path.join(data_path, cat_name))
        inds = np.random.choice(data_in_cur_cat.shape[0], size=num_data_to_sample_in_cat, replace=False)
        tr_inds = inds[0:num_train_data_to_sample_in_each_cat]
        val_inds = inds[num_train_data_to_sample_in_each_cat:num_data_to_sample_in_cat]
        for ind in tr_inds:
            feature_inds = set_feature_mat(ast.literal_eval(data_in_cur_cat.loc[ind, 'drawing']))
            feature = np.zeros(num_feature, dtype=np.bool)
            feature[feature_inds] = 1
            tr_data_pointer[cur_tr_ind] = feature
            tr_label_pointer[cur_tr_ind] = file_ind
            cur_tr_ind += 1

        for ind in val_inds:
            feature_inds = set_feature_mat(ast.literal_eval(data_in_cur_cat.loc[ind, 'drawing']))
            feature = np.zeros(num_feature, dtype=np.bool)
            feature[feature_inds] = 1
            val_data_pointer[cur_val_ind] = feature
            val_label_pointer[cur_val_ind] = file_ind
            cur_val_ind += 1
        print(cat_name + '_num_data=' + str(data_in_cur_cat.shape[0]))

    file_handler.flush()
    file_handler.close()

BASE_SIZE = 256
# Code in this function is copied from https://www.kaggle.com/gaborfodor/greyscale-mobilenet-lb-0-892/notebook
def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img

def torch_vision_transform(stroke, img_size):
    drawing = draw_cv2(stroke, size=img_size, time_color=False)
    drawing = drawing / 255  # normalize between 0 and 1
    drawing = np.repeat(drawing[np.newaxis, :, :], 3, axis=0)  # copy along second and third dimension
    # transform image values according to https://pytorch.org/docs/stable/torchvision/models.html
    drawing[:,:,0] = (drawing[:,:,0]-0.485)/0.229
    drawing[:,:,1] = (drawing[:,:,1]-0.456)/0.224
    drawing[:,:,2] = (drawing[:,:,2]-0.406)/0.225
    return drawing

def stroke_to_drawing_generator(data_path, saving_path, img_size, num_cats, num_train_data_to_sample_in_each_cat, num_val_data_to_sample_in_each_cat):
    file_handler = h5py.File(os.path.join(saving_path, 'quick_draw_stroke_data.hdf5'), 'w')
    tr_data_pointer = file_handler.create_dataset('tr_data', (num_cats * num_train_data_to_sample_in_each_cat, 3, img_size, img_size),
                                                  dtype='float32', chunks=(1, 3, img_size, img_size), compression="gzip", compression_opts=4)
    val_data_pointer = file_handler.create_dataset('val_data', (num_cats * num_val_data_to_sample_in_each_cat, 3, img_size, img_size),
                                                   dtype='float32', chunks=(1, 3, img_size, img_size), compression="gzip", compression_opts=4)

    tr_label_pointer = file_handler.create_dataset('tr_label', (num_cats * num_train_data_to_sample_in_each_cat, 1),
                                                   dtype='int64',  chunks=(1, 1), compression="gzip", compression_opts=4)
    val_label_pointer = file_handler.create_dataset('val_label', (num_cats * num_val_data_to_sample_in_each_cat, 1),
                                                    dtype='int64', chunks=(1, 1), compression="gzip", compression_opts=4)
    cur_tr_ind = 0
    cur_val_ind = 0
    num_data_to_sample_in_cat = num_train_data_to_sample_in_each_cat + num_val_data_to_sample_in_each_cat
    file_list = os.listdir(data_path)
    print('found ' + str(len(file_list)) + ' files')

    for file_ind in range(num_cats):
        cat_name = file_list[file_ind]
        if cat_name == '.DS_Store': continue
        data_in_cur_cat = pd.read_csv(os.path.join(data_path, cat_name))
        inds = np.random.choice(data_in_cur_cat.shape[0], size=num_data_to_sample_in_cat, replace=False)
        tr_inds = inds[0:num_train_data_to_sample_in_each_cat]
        val_inds = inds[num_train_data_to_sample_in_each_cat:num_data_to_sample_in_cat]
        for ind in tr_inds:
            tr_data_pointer[cur_tr_ind] = torch_vision_transform(ast.literal_eval(data_in_cur_cat.loc[ind, 'drawing']), img_size)
            tr_label_pointer[cur_tr_ind] = file_ind
            cur_tr_ind += 1

        for ind in val_inds:
            val_data_pointer[cur_val_ind] = torch_vision_transform(ast.literal_eval(data_in_cur_cat.loc[ind, 'drawing']), img_size)
            val_label_pointer[cur_val_ind] = file_ind
            cur_val_ind += 1
        print(cat_name + '_num_data=' + str(data_in_cur_cat.shape[0]))

    file_handler.flush()
    file_handler.close()


if __name__ == "__main__":
    # sample_and_compose_data(dir_loc='./cv_simplified', num_tr_data_to_sample=5000, num_val_data_to_sample=100,
    #                         save_loc='./cv_one_file')
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=int, default=1)
    args = parser.parse_args()

    slurm_tmp_dir_loc = os.environ.get('SLURM_TMPDIR', '.')
    # sample_from_all_data(data_path=os.path.join(slurm_tmp_dir_loc, 'quick_draw_data'),
    #                      saving_path = slurm_tmp_dir_loc,
    #                      num_train_data_to_sample_in_each_cat=20000,
    #                      num_val_data_to_sample_in_each_cat=500, part=args.part)

    stroke_to_drawing_generator(data_path=os.path.join(slurm_tmp_dir_loc, 'quick_draw_data'),
                                saving_path=slurm_tmp_dir_loc,
                                img_size=224,
                                num_cats=340,
                                num_train_data_to_sample_in_each_cat=20000,
                                num_val_data_to_sample_in_each_cat=500
                                )