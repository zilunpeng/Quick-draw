import os
import time
import numpy as np
import pandas as pd
import build_feature
from average_predictions import mapk
import torch
import torch.nn as nn
from torch.utils import data
import argparse
import torch.optim as optim
from utils import save_checkpoint
from utils import parse_validation_data_labels
import wandb
import h5py
from torch.nn.functional import softmax as Softmax

from adam_training_100_cats import Image_dataset as Validation_dataset

def validate(crf, validate_data_true_label, validate_loader, val_dataset_size, device):
    all_predictions = []
    for image_batch, data_ids, _ in validate_loader:
        all_predictions.extend(crf.make_predictions(image_batch).cpu().numpy())
    all_predictions = np.squeeze(np.array(all_predictions))
    return mapk(actual=validate_data_true_label, predicted=parse_validation_data_labels(all_predictions), k=3)

class Image_dataset(data.Dataset):
    def __init__(self, data, label, data_size, num_cats, device):
        self.data = data
        self.labels = label
        self.data_size = data_size
        self.num_cats = num_cats
        self.device = device

    def __len__(self):
        return self.data_size

    # Get feature from the dataset,
    def __getitem__(self, index):
        feature = torch.tensor(self.data[index], dtype=torch.float32, device=self.device)
        label_as_vec = torch.zeros((self.num_cats,1), device=self.device)
        label = self.labels[index][0]
        label_as_vec[label] = 1
        return feature, label_as_vec, label, index

class CRF:
    def __init__(self, num_cats, num_features, num_tr_data, lr, l2_reg_param, line_search, device):
        self.crf_weights = torch.randn((num_cats, num_features), dtype=torch.float32, device=device)
        self.probs = torch.zeros((num_tr_data, num_cats, 1), device=device)
        self.is_sample_visited = np.zeros(num_tr_data, dtype=np.bool)
        self.num_sample_visited = 0
        self.full_grad = 0
        self.lr = lr
        self.l2_reg_param = l2_reg_param
        self.num_cats = num_cats
        self.line_search = line_search
        self.line_search_lr = 1
        self.device = device

    def update(self, image_feature, label_vec, label, index):
        if self.is_sample_visited[index] == False:
            self.num_sample_visited += 1
            self.is_sample_visited[index] = True
        new_prob = Softmax(torch.mv(self.crf_weights, image_feature), dim=0).unsqueeze(-1)
        repeated_image_feature = image_feature.repeat(self.num_cats, 1)
        self.full_grad = self.full_grad - repeated_image_feature*(self.probs[index] - new_prob)
        if self.line_search:
            self.line_search_lr = self.lipschitz_line_search(image_feature, repeated_image_feature, label_vec, label, new_prob)
            self.crf_weights = (1-self.l2_reg_param*self.line_search_lr)*self.crf_weights - (self.line_search_lr/self.num_sample_visited)*self.full_grad
        else:
            self.crf_weights = (1-self.l2_reg_param*self.lr)*self.crf_weights - (self.lr/self.num_sample_visited)*self.full_grad
        self.probs[index] = new_prob

    def lipschitz_line_search(self, image_feature, repeated_image_feature, img_label_vec, img_label, new_prob):
        L_g = self.line_search_lr
        old_func_val = -torch.log(new_prob[img_label])
        stoc_grad = repeated_image_feature*(img_label_vec - new_prob)
        flat_stoc_grad = stoc_grad.view(-1)
        l2_norm_stoc_grad = torch.dot(flat_stoc_grad, flat_stoc_grad)
        prob = Softmax(torch.mv(self.crf_weights + (1/L_g)*stoc_grad, image_feature), dim=0)
        new_func_val = -torch.log(prob[img_label])
        while new_func_val >= (old_func_val - (1/(2*L_g))*l2_norm_stoc_grad):
            L_g = L_g * 2
            prob = Softmax(torch.mv(self.crf_weights + (1/L_g)*stoc_grad, image_feature), dim=0)
            new_func_val = -torch.log(prob[img_label])
        return L_g

    def make_predictions(self, image_batch):
        return torch.topk(torch.exp(torch.mm(image_batch, torch.t(self.crf_weights))), 3)[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_loc', default='./cv_simplified')
    parser.add_argument('--checkpoint_loading_path', default='./saved_model')
    parser.add_argument('--checkpoint_saving_path', default='./saved_model')
    parser.add_argument('--on_cluster', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--line_search', action='store_true', default=False)
    parser.add_argument('--wandb_logging', action='store_true', default=False)
    parser.add_argument('--wandb_exp_name')
    parser.add_argument('--step_size', type=float, default=1e-06)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--regularization_param', type=float, default=1e-03)
    parser.add_argument('--max_pass', type=int, default=10)
    parser.add_argument('--opt_method', default='adam')
    args = parser.parse_args()

    if args.on_cluster:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.wandb_logging:
        wandb.init(project='quick_draw_crf', name=args.wandb_exp_name)
        wandb.config.update({"step_size": args.step_size, "l2_regularize": args.regularization_param,
                             "opt_method":'sag'})

    torch.manual_seed(1)
    num_features = 851968
    num_cats = 113
    num_val_data_per_cat = 500

    data_fh = h5py.File(args.dataset_loc, 'r')
    num_tr_data = len(data_fh['tr_data'])
    tr_dataset = Image_dataset(data=data_fh['tr_data'], label=data_fh['tr_label'], data_size=num_tr_data, num_cats=num_cats, device=device)
    val_dataset = Validation_dataset(data=data_fh['val_data'], label=data_fh['val_label'], num_cats=num_cats, num_data_per_cat=num_val_data_per_cat, num_features=num_features, device=device)
    train_loader = data.DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    validate_loader = data.DataLoader(val_dataset, batch_size=args.batch_size)
    validate_data_true_label = parse_validation_data_labels(data_fh['val_label'][:]) # Assumed no zero entries here!

    crf = CRF(num_cats, num_features, num_tr_data, args.step_size, args.regularization_param, args.line_search, device)

    iter = 0
    for num_pass in range(args.max_pass):
        for image_batch, label_vecs, labels, indicies in train_loader:
            num_data_in_batch = image_batch.shape[0]
            for i in range(num_data_in_batch):
                crf.update(image_batch[i], label_vecs[i], labels[i], indicies[i])
                iter += 1
                if iter % 1000 == 0:
                    val_err = validate(crf, validate_data_true_label, validate_loader, val_dataset.data_size, device)
                    if args.wandb_logging:
                        grad = torch.abs(crf.full_grad/crf.num_sample_visited)
                        step_size = 1/crf.line_search_lr if args.line_search else crf.lr
                        wandb.log({'val_err': val_err, 'grad_l1': torch.sum(torch.sum(grad)), 'grad_l_inf':torch.max(grad), 'step_size':step_size})

if __name__ == '__main__':
    main()