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
from sag4crf_training import Image_dataset
import torch.optim as optim
from utils import save_checkpoint
from utils import parse_validation_data_labels
import wandb

def validate(crf, validate_data_true_label, validate_loader, val_dataset_size, device):
    all_predictions = []
    for image_batch, data_ids, _ in validate_loader:
        all_predictions.extend(crf.make_predictions(image_batch).cpu().numpy())
    all_predictions = np.squeeze(np.array(all_predictions))
    return mapk(actual=validate_data_true_label, predicted=parse_validation_data_labels(all_predictions), k=3)

class CRF(nn.Module):
    def __init__(self, initial_weights, num_features, num_cats):
        super(CRF, self).__init__()
        self.crf_weights = initial_weights
        self.linear = nn.Linear(num_features, num_cats, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss()

    def forward(self, image_batch, data_labels, regularization_param, device, num_cats=5):
        probs = self.log_softmax(self.linear(image_batch))
        return self.nll_loss(probs, data_labels.to(device))

    def make_predictions(self, image_batch):
        return torch.topk(torch.exp(self.linear(image_batch)), 3)[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_loc', default='./cv_simplified')
    parser.add_argument('--checkpoint_loading_path', default='./saved_model')
    parser.add_argument('--checkpoint_saving_path', default='./saved_model')
    parser.add_argument('--on_cluster', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--wandb_logging', action='store_true', default=False)
    parser.add_argument('--wandb_exp_name')
    parser.add_argument('--step_size', type=float, default=1e-06)
    parser.add_argument('--weight_decay', type=float, default=1e-06)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--regularization_param', type=float, default=1e-03)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--opt_method', default='adam')
    args = parser.parse_args()

    if args.on_cluster:
        dataset_loc = os.environ.get('SLURM_TMPDIR', '.')
        dataset_loc = os.path.join(dataset_loc, args.dataset_loc)
        device = torch.device('cuda')
    else:
        dataset_loc = args.dataset_loc
        device = torch.device('cpu')

    if args.wandb_logging:
        wandb.init(project='quick_draw_crf', name=args.wandb_exp_name)
        wandb.config.update({"step_size": args.step_size, "weight_decay": args.weight_decay, "opt_method":args.opt_method})

    torch.manual_seed(1)
    num_cats = 5
    num_features = 851968
    tr_dataset = Image_dataset(data_path=os.path.join(dataset_loc, 'tr.pkl'), num_features=num_features, device=device)
    val_dataset = Image_dataset(data_path=os.path.join(dataset_loc, 'val.pkl'), num_features=num_features, device=device)
    train_loader = data.DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    validate_loader = data.DataLoader(val_dataset, batch_size=args.batch_size)
    validate_data_true_label = parse_validation_data_labels(pd.read_pickle(os.path.join(args.dataset_loc, 'val.pkl'))['cat_ind'].values)

    weight_param = torch.randn((num_cats, num_features), dtype=torch.float64, device=device, requires_grad=True)
    crf = CRF(weight_param, num_features, num_cats).to(device)
    if args.opt_method == 'adam':
        optimizer = optim.Adam(crf.parameters(), lr=args.step_size, weight_decay=args.weight_decay)
    elif args.opt_method == 'sgd':
        optimizer = optim.SGD(crf.parameters(), lr=args.step_size, weight_decay=args.weight_decay)
    crf.name = 'crf_' + args.opt_method

    print('initial validation error = %.3f' % (validate(crf, validate_data_true_label, validate_loader, val_dataset.data_size, device)))

    for epoch in range(args.max_epoch):
        optimizer.zero_grad()
        i = 1
        for image_batch, data_ids, data_labels in train_loader:
            loss = crf(image_batch, data_labels, args.regularization_param, device)
            loss.backward()
            if i % 10 == 0:
                optimizer.step()
                adam_grad_l1 = torch.sum(torch.sum(torch.abs(crf.linear.weight.grad)))
                optimizer.zero_grad()
            i += 1
        val_err = validate(crf, validate_data_true_label, validate_loader, val_dataset.data_size, device)
        if args.wandb_logging:
            wandb.log({'val_err':val_err, 'tr_loss':loss, 'adam_grad_l1':adam_grad_l1})
        else:
            print('epoch=%d. ' % (epoch) + '. validation error = %.3f' % (val_err))

    save_checkpoint(crf, args.checkpoint_saving_path, epoch)

if __name__ == '__main__':
    main()