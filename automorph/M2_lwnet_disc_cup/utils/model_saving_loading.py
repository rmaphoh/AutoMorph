import os
import os.path as osp
import torch
import argparse

def save_model(path, model, optimizer, stats= None):
    os.makedirs(path, exist_ok=True)
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stats':stats
            }, osp.join(path, 'model_checkpoint.pth'))

def load_model(model, experiment_path, device='cpu', with_opt=False):
    checkpoint_path = osp.join(experiment_path, 'model_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if with_opt:
        return model, checkpoint['stats'], checkpoint['optimizer_state_dict']
    return model, checkpoint['stats']

def str2bool(v):
    # as seen here: https://stackoverflow.com/a/43357954/3208255
    if isinstance(v, bool):
       return v
    if v.lower() in ('true','yes'):
        return True
    elif v.lower() in ('false','no'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected.')