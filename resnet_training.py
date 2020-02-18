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
    parser.add_argument('--model_loading_path', default='./saved_model')
    parser.add_argument('--model_saving_path', default='./saved_model')
    parser.add_argument('--load_model', action='store_true', default=False)
    parser.add_argument('--on_cluster', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--line_search', action='store_true', default=False)
    parser.add_argument('--wandb_logging', action='store_true', default=False)
    parser.add_argument('--wandb_exp_name')
    parser.add_argument('--step_size', type=float, default=1e-06)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--num_iter_accum_grad', type=int, default=1)
    parser.add_argument('--max_pass', type=int, default=10)
    parser.add_argument('--opt_method', default='adam')
    parser.add_argument('--model_name', default='resnet18')
    args = parser.parse_args()

    device = torch.device('cuda') if args.on_cluster else torch.device('cpu')
    if args.on_cluster: os.environ['TORCH_HOME'] = os.environ.get('SLURM_TMPDIR', '.')

    if args.wandb_logging:
        wandb.init(project='quick_draw_crf', name=args.wandb_exp_name)
        wandb.config.update({"step_size": args.step_size, "opt_method":args.opt_method,
                            "num_data_used_calc_grad": args.batch_size * args.num_iter_accum_grad,
                             'model_name':args.model_name})

    num_cats = 340
    num_tr_files = 20
    val_data_fh = h5py.File(os.path.join(args.dataset_loc, 'quick_draw_resnet_val_data.hdf5'), 'r')
    val_dataset = Image_dataset(data=val_data_fh['val_data'], label=val_data_fh['val_label'], data_size=len(val_data_fh['val_data']))
    validate_loader = data.DataLoader(val_dataset, batch_size=args.batch_size)
    validate_data_true_label = parse_validation_data_labels(val_data_fh['val_label'][:])  # Assumed no zero entries here!

    if args.model_name == 'resnet18':
        resnet = torchvision.models.resnet18(pretrained=False, num_classes=num_cats).to(device)
    elif args.model_name == 'resnet34':
        resnet = torchvision.models.resnet34(pretrained=False, num_classes=num_cats).to(device)
    elif args.model_name == 'resnet50':
        resnet = torchvision.models.resnet50(pretrained=False, num_classes=num_cats).to(device)
    optimizer = optim.Adam(resnet.parameters(), lr=args.step_size, weight_decay=args.weight_decay)

    if args.load_model:
        checkpoint = torch.load(args.model_loading_path)
        resnet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        resnet.train()
    else:
        epoch = 0
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(resnet.parameters(), lr=args.step_size, momentum=0.9)
    while epoch < args.max_pass:
        optimizer.zero_grad()
        i = 1

        tr_data_fh = h5py.File(os.path.join(args.dataset_loc, 'quick_draw_resnet_data_'+str(np.random.randint(0,num_tr_files))+'.hdf5'), 'r')
        tr_dataset = Image_dataset(data=tr_data_fh['tr_data'], label=tr_data_fh['tr_label'], data_size=len(tr_data_fh['tr_data']))
        train_loader = data.DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)

        for image_batch, data_labels in train_loader:
            image_batch = image_batch.to(device)
            data_labels = data_labels.to(device)
            resnet.train()
            outputs = resnet(image_batch)
            loss = criterion(outputs, data_labels)
            loss.backward()
            if i % args.num_iter_accum_grad == 0:
                optimizer.step()
                optimizer.zero_grad()

                val_err = validate(resnet, validate_data_true_label, validate_loader, val_dataset.data_size, device)
                if args.wandb_logging:
                    wandb.log({'val_err': val_err, 'tr_loss': loss.item()})
                else:
                    print('epoch=%d. ' % (epoch) + '. validation error = %.3f' % (val_err))
            i += 1
        torch.save({
            'epoch': epoch,
            'model_state_dict': resnet.state_dict(),
            'optimizer_state_dict':optimizer.state_dict()
        }, os.path.join(args.model_saving_path, args.model_name+'.pt'))
        epoch += 1

if __name__ == '__main__':
    main()