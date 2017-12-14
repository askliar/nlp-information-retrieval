import torch
import shutil

def save_checkpoint(state, is_best, config):
    torch.save(state, config.save_path_latest)
    if is_best:
        shutil.copyfile(config.save_path_latest, config.save_path_best)