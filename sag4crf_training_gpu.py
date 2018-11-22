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
        self.reg_lam = regularization_param
        self.alpha = step_size
        self.max_iter = maximum_iteration
        self.delta = err_tolerance
        self.fold_num = fold_num
        self.init_all_cats(data_dir)

        self.tot_data_seen = 0
        self.cur_cat = 0
        self.read_category(self.cur_cat)

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
        self.cat_visit_freq[self.cur_cat] += 1
        self.init_val_fold()

    def init_val_fold(self):
        self.cur_val_data_path = self.cur_cat_path + '/val' + str(self.fold_num) + '.pkl'
        self.cur_val_data_fold = pd.read_pickle(self.cur_val_data_path)
        self.cur_val_data_size = self.cur_val_data_fold.shape[0]

    def update_category(self):
        self.cur_cat= (self.cur_cat+1 if self.cur_cat<self.num_cats-1 else 0)
        self.read_category(self.cur_cat)

    def build_training_session(self, tr_graph, weights_ph, d_ph, probs_ph):
        tr_sess = tf.Session(graph=tr_graph, config=tf.ConfigProto(log_device_placement=True))
        tr_sess.run(tf.variables_initializer([weights_ph, d_ph, probs_ph]))
        return tr_sess

    def build_training_graph(self):
        g = tf.Graph()
        with g.as_default():
            alpha = tf.constant(self.alpha, dtype=tf.float32)
            reg_lam = tf.constant(self.reg_lam, dtype=tf.float32)

            weights_ph = tf.Variable(tf.zeros([self.num_cats, 851968], dtype=tf.float32))
            probs_ph = tf.Variable(tf.zeros([self.num_cats, 306026], dtype=tf.float32))
            d_ph = tf.Variable(tf.zeros([851968], dtype=tf.float32))
            feat_i_ph = tf.placeholder(tf.float32, shape=(851968))
            data_id_ph = tf.placeholder(tf.int32)
            cur_cat_ph = tf.placeholder(tf.int32)
            total_data_seen_ph = tf.placeholder(tf.float32)

            probs = tf.tensordot(weights_ph, feat_i_ph, axes=[[1], [0]])
            probs = tf.nn.softmax(probs)
            new_prob = probs[cur_cat_ph]
            update_d = d_ph.assign(d_ph+ feat_i_ph * (probs_ph[cur_cat_ph, data_id_ph] - new_prob))
            w = (tf.constant(1.0)-alpha*reg_lam)*weights_ph[cur_cat_ph,:] - (alpha/total_data_seen_ph)*update_d
            update_weights = tf.scatter_update(weights_ph, indices=cur_cat_ph, updates=w)
            update_probs = probs_ph[cur_cat_ph, data_id_ph].assign(new_prob)
        return g, update_weights, update_probs, data_id_ph, cur_cat_ph, feat_i_ph, total_data_seen_ph, weights_ph, d_ph, probs_ph

    def build_validating_session(self, val_graph):
        val_sess = tf.Session(graph=val_graph, config=tf.ConfigProto(log_device_placement=True))
        return val_sess

    def build_validating_graph(self):
        g = tf.Graph()
        with g.as_default():
            feat_i_ph = tf.placeholder(tf.float32, shape=(851968))
            weights_ph = tf.placeholder(tf.float32, shape=(self.num_cats, 851968))
            Z = tf.exp(tf.tensordot(weights_ph, feat_i_ph, axes=[[1], [0]]))
            _, predictions_i = tf.nn.top_k(Z, k=3, sorted=True)
        return g, predictions_i, feat_i_ph, weights_ph

    def get_val_acc(self, val_sess, predictions_i, feat_i_ph, weights_ph, weights):
        predictions = []
        for i, val_data in self.cur_val_data_fold.iterrows():
            val_data = build_feature.set_feature_mat(val_data['drawing'], 256)
            predictions.append(val_sess.run(predictions_i, feed_dict={feat_i_ph: val_data, weights_ph:weights}))
        return mapk(actual=np.matrix(np.ones((self.cur_val_data_size), dtype=np.int8) * self.cur_cat), predicted=np.array(predictions), k=3)

    def custom_random_sampler(self, data_id):
        x_i = self.cur_tr_data_fold.loc[data_id,'drawing']
        if self.cat_visit_freq[self.cur_cat] == 1:
            self.tot_data_seen += 1
        return x_i

    def sag_training(self):
        iter = 0
        epoch = 0

        tr_graph, update_weights, update_probs, data_id_ph, cur_cat_ph, feat_i_ph, total_data_seen_ph, weights_ph, d_ph, probs_ph = self.build_training_graph()
        tr_sess = self.build_training_session(tr_graph, weights_ph, d_ph, probs_ph)

        val_graph, predictions_i, feat_i_val_ph, weights_val_ph = self.build_validating_graph()
        val_sess = self.build_validating_session(val_graph)

        then = time.time()
        print('start training')
        while epoch<self.max_iter:
            data_id = self.cur_tr_fold_seq[iter]
            x_i = self.custom_random_sampler(data_id)
            feat_i = build_feature.set_feature_mat(x_i,256)
            tr_sess.run([update_weights, update_probs], feed_dict={data_id_ph:data_id, cur_cat_ph:self.cur_cat, feat_i_ph:feat_i, total_data_seen_ph:self.tot_data_seen})

            if iter%200 == 0: print('iter=%d'%(iter))
            iter += 1
            if iter >= self.cur_tr_fold_size:
                now = time.time()
                print('finished training on category ' + self.cat_names[self.cur_cat] + '. Took %.2f'%(now-then) + 'seconds. Start validating.')
                weights = tr_sess.run(weights_ph)
                print('epoch=%d. trained on category '%(epoch) + self.cat_names[self.cur_cat] + '. score on validation set is %.5f'%(self.get_val_acc(val_sess, predictions_i, feat_i_val_ph, weights_val_ph, weights)))
                self.update_category()
                then = time.time()
                iter = 0
                epoch += 1

crf = sag4crf(data_dir='cv_simplified',fold_num=1,regularization_param=0.001,step_size=0.0001,maximum_iteration=3*340,err_tolerance=0.00001)
crf.sag_training()