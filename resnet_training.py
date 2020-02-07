import os
import time
import numpy as np
import pandas as pd
import build_feature
from average_predictions import mapk
import torch
import torch.nn as nn
import torchvision
from torch.utils import data
import argparse
import torch.optim as optim
from utils import parse_validation_data_labels
import wandb
import h5py

class Image_dataset(data.Dataset):
    def __init__(self, data, label, data_size):
        self.data = data
        self.labels = label
        self.data_size = data_size

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return self.data[index], self.labels[index][0]

def validate(resnet, validate_data_true_label, validate_loader, val_dataset_size, device):
    all_predictions = []
    resnet.eval()
    for image_batch, _ in validate_loader:
        image_batch = image_batch.to(device)
        outputs = resnet(image_batch)
        _, predicted = torch.topk(outputs.data, 3)
        all_predictions.extend(predicted.cpu().numpy())
    all_predictions = np.squeeze(np.array(all_predictions))
    return mapk(actual=validate_data_true_label, predicted=parse_validation_data_labels(all_predictions), k=3)

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
    parser.add_argument('--num_iter_accum_grad', type=int, default=1)
    parser.add_argument('--max_pass', type=int, default=10)
    parser.add_argument('--opt_method', default='adam')
    args = parser.parse_args()

    device = torch.device('cuda') if args.on_cluster else torch.device('cpu')
    if args.on_cluster: os.environ['TORCH_HOME'] = os.environ.get('SLURM_TMPDIR', '.')

    if args.wandb_logging:
        wandb.init(project='quick_draw_crf', name=args.wandb_exp_name)
        wandb.config.update({"step_size": args.step_size, "opt_method":args.opt_method,
                            "num_data_used_calc_grad": args.batch_size * args.num_iter_accum_grad})

    num_cats = 3
    data_fh = h5py.File(args.dataset_loc, 'r')
    num_tr_data = len(data_fh['tr_data'])
    tr_dataset = Image_dataset(data=data_fh['tr_data'], label=data_fh['tr_label'], data_size=num_tr_data)
    val_dataset = Image_dataset(data=data_fh['val_data'], label=data_fh['val_label'], data_size=len(data_fh['val_data']))
    train_loader = data.DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    validate_loader = data.DataLoader(val_dataset, batch_size=args.batch_size)
    validate_data_true_label = parse_validation_data_labels(data_fh['val_label'][:])  # Assumed no zero entries here!

    resnet = torchvision.models.resnet18(pretrained=False, num_classes=num_cats).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=args.step_size)
    #optimizer = optim.SGD(resnet.parameters(), lr=args.step_size, momentum=0.9
    for epoch in range(args.max_pass):
        optimizer.zero_grad()
        i = 1
        for image_batch, data_labels in train_loader:
            image_batch = image_batch.to(device)
            data_labels = data_labels.to(device)
            resnet.train()
            outputs = resnet(image_batch)
            loss = criterion(outputs, data_labels)
            tr_loss = loss.item()
            loss.backward()
            if i % args.num_iter_accum_grad == 0:
                optimizer.step()
                optimizer.zero_grad()

                val_err = validate(resnet, validate_data_true_label, validate_loader, val_dataset.data_size, device)
                if args.wandb_logging:
                    wandb.log({'val_err': val_err, 'tr_loss': tr_loss})
                else:
                    print('epoch=%d. ' % (epoch) + '. validation error = %.3f' % (val_err))

            i += 1


if __name__ == '__main__':
    main()