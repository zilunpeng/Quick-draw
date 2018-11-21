import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import build_feature
from average_predictions import mapk

class sag4crf:

    def __init__(self, data_dir, fold_num, regularization_param, step_size, maximum_iteration, err_tolerance):
        self.data_dir = data_dir
        self.reg_lam = tf.constant(regularization_param)
        self.alpha = tf.constant(step_size)
        self.max_iter = maximum_iteration
        self.delta = err_tolerance
        self.fold_num = fold_num
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        self.init_all_cats(data_dir)
        self.cur_cat = 0
        self.read_category(self.cur_cat)

        self.weights = tf.Variable(initial_value=tf.zeros([self.num_cats, 851968]))
        self.d = tf.Variable(initial_value=tf.zeros([851968]))
        self.sess.run(tf.global_variables_initializer())
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

    def get_new_prob(self, feature_i):
        probs = tf.tensordot(self.weights, feature_i, axes=[[1], [0]])
        probs = tf.nn.softmax(probs)
        return probs[self.cur_cat]

    def get_w(self, old_prob, feature_i, new_prob, total_data_seen):
        d = self.d.assign(self.d + feature_i * (old_prob - new_prob))
        return (tf.constant(1.0)-self.alpha*self.reg_lam)*self.weights[self.cur_cat,:] - (self.alpha/tf.convert_to_tensor(total_data_seen, dtype=tf.float32))*d

    def update_w(self, w):
        return tf.scatter_update(self.weights, indices=self.cur_cat, updates=w)

    def custom_random_sampler(self, data_id):
        x_i = self.cur_tr_data_fold.loc[data_id,'drawing']
        if self.cat_visit_freq[self.cur_cat] == 1:
            self.tot_data_seen += 1
        return x_i

    def get_predictions_i(self, val_data):
        Z = tf.exp(tf.tensordot(self.weights, val_data, axes=[[1], [0]]))
        _, predictions_i = tf.nn.top_k(Z, k=3, sorted=True)

    def sag_training(self):
        iter = 0
        epoch = 0

        feat_i_ph = tf.placeholder(dtype=tf.float32)
        get_new_prob = self.get_new_prob(feat_i_ph)
        old_prob_ph = tf.placeholder(dtype=tf.float32)
        new_prob_ph = tf.placeholder(dtype=tf.float32)
        tot_data_seen_ph = tf.placeholder(dtype=tf.float32)
        get_w = self.get_w(old_prob_ph, feat_i_ph, new_prob_ph,tot_data_seen_ph)
        w_ph = tf.placeholder(dtype=tf.float32)
        update_w = self.update_w(w_ph)
        val_data_i = tf.placeholder(dtype=tf.float32)
        get_predictions_i = self.get_predictions_i(val_data_i)

        then = time.time()
        print('start training')
        while epoch<self.max_iter:
            data_id = self.cur_tr_fold_seq[iter]
            x_i = self.custom_random_sampler(data_id)
            feat_i = build_feature.set_feature_mat(x_i,256)

            new_prob = self.sess.run(get_new_prob, {feat_i_ph: feat_i})
            w = self.sess.run(get_w, {old_prob_ph:self.probs[data_id], new_prob_ph:new_prob, feat_i_ph:feat_i, tot_data_seen_ph:self.tot_data_seen})
            self.sess.run(update_w, {w_ph:w})
            if iter%200 == 0: print('iter=%d'%(iter)+' prob=%.7f'%(new_prob))

            iter += 1
            if iter >= self.cur_tr_fold_size:
                now = time.time()
                print('finished training on category ' + self.cat_names[self.cur_cat] + '. Took %.2f'%(now-then) + 'seconds. Start validating.')
                predictions = []
                for i, val_data in self.cur_val_data_fold.iterrows():
                    val_data = build_feature.set_feature_mat(val_data['drawing'], 256)
                    predictions.append(self.sess.run(get_predictions_i, {val_data_i:val_data}))
                val_err = mapk(actual=np.matrix(np.ones((self.cur_val_data_size), dtype=np.int8) * self.cur_cat), predicted=np.array(predictions), k=3)
                print('epoch=%d. trained on category '%(epoch) + self.cat_names[self.cur_cat] + '. NLL on validation set is %.5f'%(val_err))

                self.update_category()
                then = time.time()
                iter = 0
                epoch += 1
        return w

crf = sag4crf(data_dir='cv_simplified',fold_num=1,regularization_param=0.001,step_size=0.000001,maximum_iteration=3*340,err_tolerance=0.00001)
crf.sag_training()