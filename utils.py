import os
import torch
import numpy as np
from average_predictions import mapk

def save_checkpoint(model, model_dir, epoch):
    model_name = model.name+'_epoch='+str(epoch)
    path = os.path.join(model_dir, model_name)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({'state': model.state_dict(), 'epoch': epoch}, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model_name, path=path
    ))

def parse_validation_data_labels(labels):
    if len(labels.shape) == 1: labels = labels[:, np.newaxis]
    return labels.tolist()

def validate(crf, validate_data_true_label, validate_loader, val_dataset_size, device):
    all_predictions = []
    for image_batch, data_ids, _ in validate_loader:
        all_predictions.extend(crf.make_predictions(image_batch).cpu().numpy())
    all_predictions = np.squeeze(np.array(all_predictions))
    return mapk(actual=validate_data_true_label, predicted=parse_validation_data_labels(all_predictions), k=3)