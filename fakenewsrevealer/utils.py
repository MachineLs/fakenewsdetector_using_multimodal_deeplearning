import numpy as np
import torch
import os

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['name'], param_group['lr'])


class CheckpointSaving:

    def __init__(self, path='checkpoint.pt', verbose=True, trace_func=print):
        self.best_score = None
        self.val_acc_max = 0
        self.path = path
        self.verbose = verbose
        self.trace_func = trace_func

    def __call__(self, val_acc, model):
        if self.best_score is None:
            self.best_score = val_acc
            self.save_checkpoint(val_acc, model)
        elif val_acc > self.best_score:
            self.best_score = val_acc
            self.save_checkpoint(val_acc, model)

    def save_checkpoint(self, val_acc, model):
        if self.verbose:
            self.trace_func(
                f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}). Model saved ...')
        torch.save(model.state_dict(), self.path)
        self.val_acc_max = val_acc


class EarlyStopping:

    def __init__(self, patience=10, verbose=False, delta=0.000001, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.verbose:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            # self.save_checkpoint(val_loss, model)
            self.val_loss_min = val_loss
            self.counter = 0

    # def save_checkpoint(self, val_loss, model):
    #     if self.verbose:
    #         self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Model saved ...')
    #     torch.save(model.state_dict(), self.path)
    #     self.val_loss_min = val_loss


def make_directory(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory failed")
    else:
        print("Successfully created the directory")