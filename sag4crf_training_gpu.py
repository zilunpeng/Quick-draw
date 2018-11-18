import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import build_feature
from average_predictions import mapk
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class sag4crf:

    def __init__(self, data_dir, fold_num, regularization_param, step_size, maximum_iteration, err_tolerance):
        self.data_dir = data_dir
        self.reg_lam = regularization_param
        self.max_iter = maximum_iteration
        self.delta = err_tolerance
        self.fold_num = fold_num
        self.alpha = step_size
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        self.init_all_cats(data_dir)
        self.cur_cat = 0
        self.read_category(self.cur_cat)

        self.weights = tf.get_variable("weights",shape=[self.num_cats, 851968],dtype=tf.float64,initializer=tf.zeros_initializer)
        # with tf.device('/cpu:0'):
        #     self.sess.run(self.weights.initializer)
        self.sess.run(self.weights.initializer)
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
        self.cur_tr_data_path = self.cur_cat_path + '/tr' + str(self.fold_num) + '.pkl'
        self.cur_tr_data_fold = pd.read_pickle(self.cur_tr_data_path)
        self.cur_tr_fold_size = self.cur_tr_data_fold.shape[0]
        self.cur_tr_fold_seq = np.random.permutation(self.cur_tr_fold_size)
        self.probs = (np.zeros(851968) if self.cat_visit_freq[self.cur_cat] == 0 else np.load(self.cur_cat_path+'/probs.npy'))
        self.cat_visit_freq[self.cur_cat] += 1
        self.init_val_fold()

    def init_val_fold(self):
        self.cur_val_data_path = self.cur_cat_path + '/val' + str(self.fold_num) + '.pkl'
        self.cur_val_data_fold = pd.read_pickle(self.cur_val_data_path)
        self.cur_val_data_size = self.cur_val_data_fold.shape[0]

    def update_category(self):
        self.cur_cat= (self.cur_cat+1 if self.cur_cat<self.num_cats-1 else 0)
        np.save(file=self.cur_cat_path+'/probs.npy',arr=self.probs)
        self.read_category(self.cur_cat)

    def compute_d(self, old_d, data_id, feature_i):
        # with tf.device('/cpu:0'):
        #     Z = tf.exp(tf.tensordot(self.weights, feature_i, axes=[[1],[0]]))
        #     new_prob = Z[self.cur_cat] / tf.reduce_sum(Z)
        #     d = old_d + feature_i * (self.probs[data_id] - new_prob)
        Z = tf.exp(tf.tensordot(self.weights, feature_i, axes=[[1], [0]]))
        new_prob = Z[self.cur_cat] / tf.reduce_sum(Z)
        d = old_d + feature_i * (self.probs[data_id] - new_prob)
        self.probs[data_id] = self.sess.run(new_prob)
        return self.sess.run(d)

    def custom_random_sampler(self, data_id):
        x_i = self.cur_tr_data_fold.loc[data_id,'drawing']
        y_i = self.cur_cat
        if self.cat_visit_freq[self.cur_cat] == 1:
            self.tot_data_seen += 1
        return x_i, y_i

    def get_val_err(self):
        predictions = []
        # with tf.device('/cpu:0'):
        #     for i, val_data in self.cur_val_data_fold.iterrows():
        #         val_data = val_data['drawing']
        #         val_data = tf.convert_to_tensor(build_feature.set_feature_mat(val_data,256))
        #         Z = tf.exp(tf.tensordot(self.weights, val_data, axes=[[1],[0]]))
        #         _, predictions_i = tf.nn.top_k(Z, k=3, sorted=True)
        #         predictions.append(predictions_i)
        #     predictions = self.sess.run(predictions)
        for i, val_data in self.cur_val_data_fold.iterrows():
            val_data = val_data['drawing']
            val_data = tf.convert_to_tensor(build_feature.set_feature_mat(val_data, 256))
            Z = tf.exp(tf.tensordot(self.weights, val_data, axes=[[1], [0]]))
            _, predictions_i = tf.nn.top_k(Z, k=3, sorted=True)
            predictions.append(predictions_i)
        predictions = self.sess.run(predictions)
        return mapk(actual=np.matrix(np.ones((self.cur_val_data_size),dtype=np.int8)*self.cur_cat), predicted=np.array(predictions), k=3)

    def sag_training(self):
        iter = 0
        epoch = 0
        d = np.zeros(851968)
        then = time.time()
        print('start training')
        while epoch<self.max_iter:
            data_id = self.cur_tr_fold_seq[iter]
            x_i,y_i = self.custom_random_sampler(data_id)
            feat_i = tf.convert_to_tensor(build_feature.set_feature_mat(x_i,256))
            d = self.compute_d(d,data_id,feat_i)
            w = (1-self.alpha*self.reg_lam)*self.weights[self.cur_cat,:] - (self.alpha/self.tot_data_seen)*d
            w = self.sess.run(w)
            tf.scatter_update(self.weights, indices=self.cur_cat, updates=w)
            iter += 1

            if iter >= self.cur_tr_fold_size:
                now = time.time()
                print('finished training on category ' + self.cat_names[self.cur_cat] + '. Took %.2f'%(now-then) + 'seconds. Start validating.')
                val_err = self.get_val_err()
                print('epoch=%d. trained on category '%(epoch) + self.cat_names[self.cur_cat] + '. NLL on validation set is %.5f'%(val_err))
                self.update_category()
                then = time.time()
                iter = 0
                epoch += 1
        return w

crf = sag4crf(data_dir='cv_simplified',fold_num=1,regularization_param=0.001,step_size=0.000001,maximum_iteration=3*340,err_tolerance=0.00001)
crf.sag_training()