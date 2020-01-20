#! /usr/bin/env python
import os
import time
import numpy as np
import pandas as pd
import build_feature
from average_predictions import mapk
import torch
from torch.utils import data
import argparse
from sag4crf_training import Image_dataset
import torch.optim as optim
from orion.client import report_results

def parse_validation_data_labels(labels):
    if len(labels.shape) == 1: labels = labels[:, np.newaxis]
    return labels.tolist()

def train(weight_param, regularize_param, num_cats, device, train_loader, optimizer):
    for image_batch, data_ids, data_labels in train_loader:
        def closure():
            optimizer.zero_grad()
            image_batch_size = image_batch.shape[0]
            probs = torch.zeros((image_batch_size, num_cats), device=device)
            for i in range(image_batch_size):
                img_feature = image_batch[i, :]
                probs[i,:] = torch.nn.functional.softmax(torch.mv(weight_param, img_feature), dim=0)
            loss = torch.nn.functional.nll_loss(probs, data_labels.to(device)) + (regularize_param/2)*torch.sum(weight_param**2, dtype=torch.float32)
            loss.backward()
            return loss
        optimizer.step(closure)

        # image_batch_size = image_batch.shape[0]
        # probs = torch.zeros((image_batch_size, num_cats), device=device)
        # for i in range(image_batch_size):
        #     img_feature = image_batch[i, :]
        #     probs[i, :] = torch.nn.functional.softmax(torch.mv(weight_param, img_feature), dim=0)
        # loss = torch.nn.functional.nll_loss(probs, data_labels.to(device)) + (regularize_param/2)*torch.sum(weight_param**2, dtype=torch.float32)
        # print('loss = %.3f' % (loss.item()))

    return weight_param

def validate(weight_param, validate_data_true_label, validate_loader, val_dataset_size, device):
    all_predictions = torch.zeros((val_dataset_size, 3), dtype=torch.int16, device=device)
    i = 0
    for image_batch, data_ids, _ in validate_loader:
        image_batch_size = image_batch.shape[0]
        for j in range(image_batch_size):
            img_feature = image_batch[j, :]
            prediction = torch.topk(torch.exp(torch.mv(weight_param, img_feature)), 3)[1]
            all_predictions[i, :] = prediction
            i += 1
    return mapk(actual=validate_data_true_label, predicted=parse_validation_data_labels(all_predictions.cpu().numpy()), k=3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_loc', default='./cv_simplified')
    parser.add_argument('--checkpoint_loading_path', default='./saved_model')
    parser.add_argument('--on_cluster', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--step_size', type=float, default=1e-06)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--regularization_param', type=float, default=1e-03)
    parser.add_argument('--max_epoch', type=int, default=100)
    args = parser.parse_args()

    if args.on_cluster:
        dataset_loc = os.environ.get('SLURM_TMPDIR', '.')
        dataset_loc = os.path.join(dataset_loc, args.dataset_loc)
        device = torch.device('cuda')
    else:
        dataset_loc = args.dataset_loc
        device = torch.device('cpu')

    torch.manual_seed(1)
    num_cats = 5
    num_features = 851968
    tr_dataset = Image_dataset(data_path=os.path.join(dataset_loc, 'tr.pkl'), num_features=num_features, device=device)
    val_dataset = Image_dataset(data_path=os.path.join(dataset_loc, 'val.pkl'), num_features=num_features, device=device)
    train_loader = data.DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    validate_loader = data.DataLoader(val_dataset, batch_size=args.batch_size)
    validate_data_true_label = parse_validation_data_labels(pd.read_pickle(os.path.join(args.dataset_loc, 'val.pkl'))['cat_ind'].values)

    weight_param = torch.randn((num_cats, num_features), dtype=torch.float64, device=device, requires_grad=True)
    optimizer = optim.LBFGS([weight_param], lr=args.step_size, max_iter=20, tolerance_grad=1e-10, history_size=10)

    for i in range(args.max_epoch):
        weight_param = train(weight_param, args.regularization_param, num_cats, device, train_loader, optimizer)
        val_err = validate(weight_param, validate_data_true_label, validate_loader, val_dataset.data_size, device)
        print('epoch=%d. ' % (i) + '. validation error = %.3f' % (val_err))

    report_results([dict(
        name='validation result',
        type='objective',
        value= -val_err)])

if __name__ == '__main__':
    main()