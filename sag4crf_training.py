import os
import ast
import numpy as np
import pandas as pd
import build_feature

class sag4crf:

    def __init__(self, data_dir, fold_num, regularization_param, step_size, maximum_iteration, err_tolerance, maximum_iteration_on_one_category_multiplier):
        self.data_dir = data_dir
        self.reg_lam = regularization_param
        self.max_iter = maximum_iteration
        self.delta = err_tolerance
        self.fold_num = fold_num
        self.max_iter_multiplier = maximum_iteration_on_one_category_multiplier
        self.alpha = step_size

        self.init_all_cats(data_dir)
        self.cur_cat = 0
        self.read_category(self.cur_cat)

        self.weights = np.zeros((self.num_cats, 851968))
        self.tot_data_seen = 0

    def init_all_cats(self,data_dir):
        self.cat_names = self.get_categories(data_dir)
        self.num_cats = len(self.cat_names)
        self.cat_visit_freq = np.zeros(self.num_cats)

    def get_categories(self,data_dir):
        cat_names = []
        for cat_name in os.listdir(data_dir):
            cat_names.append(cat_name)
        return cat_names

    def read_category(self, cur_cat):
        self.cur_cat_path = os.path.join(self.data_dir, self.cat_names[cur_cat])
        self.cur_tr_data_path = self.cur_cat_path + '/tr' + str(self.fold_num) + '.csv'
        self.cur_tr_data_fold = pd.read_csv(self.cur_tr_data_path)
        self.cur_tr_fold_size = self.cur_tr_data_fold.shape[0]
        self.cur_tr_fold_counter = 0
        self.cur_tr_fold_sample_freq_counter = np.zeros(self.cur_tr_fold_size)
        self.max_iter_on_cat = self.max_iter_multiplier * self.cur_tr_fold_size
        self.probs = (np.zeros(851968) if self.cat_visit_freq[self.cur_cat] == 0 else np.load(self.cur_cat_path+'/probs.npy'))
        self.cat_visit_freq[self.cur_cat] += 1

    def update(self):
        self.cur_cat= (self.cur_cat+1 if self.cur_cat<self.num_cats else 0)
        np.save(self.probs, self.cur_cat_path+'/probs.npy')
        self.read_category(self.cur_cat)

    def compute_d(self, old_d, data_id, feature_i):
        Z = np.exp(self.weights @ feature_i)
        new_prob = Z[self.cur_cat]/np.sum(Z)
        d = old_d + feature_i*(self.probs[data_id] - new_prob)
        self.probs[data_id] = new_prob
        return d

    def custom_random_sampler(self):
        data_id = np.random.randint(self.cur_tr_fold_size)
        x_i = self.cur_tr_data_fold.loc[data_id,'drawing']
        x_i = ast.literal_eval(x_i)
        y_i = self.cur_cat
        self.cur_tr_fold_counter += 1
        if self.cat_visit_freq[self.cur_cat] == 1 and self.cur_tr_fold_sample_freq_counter[data_id] == 0:
            self.tot_data_seen += 1
            self.cur_tr_fold_sample_freq_counter[data_id] = 1
        return data_id,x_i, y_i

    def sag_training(self):
        iter = 0
        d = np.zeros(851968)
        w = np.zeros(851968)
        while iter<self.max_iter:
            data_id,x_i,y_i = self.custom_random_sampler()
            feat_i = build_feature.set_feature_mat(x_i,256)
            d = self.compute_d(d,data_id,feat_i)
            w = (1-self.alpha*self.reg_lam)*w -(self.alpha/self.tot_data_seen)*d

            if self.cur_tr_fold_counter > self.max_iter_on_cat:
                self.update()
                iter += 1
        return w

sth = sag4crf('cv_simplified',1,0.1,1,1000,1e-5,3)
sth.sag_training()