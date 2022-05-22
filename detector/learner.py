import gc
import itertools
import random

import numpy as np
import optuna
import pandas as pd
import torch

from evaluation import multiclass_acc
from model import FakeNewsModel, calculate_loss
from utils import AvgMeter, print_lr, EarlyStopping, CheckpointSaving


def batch_constructor(config, batch):
    b = {}
    for k, v in batch.items():
        if k != 'text':
            b[k] = v.to(config.device)
    return b


def train_epoch(config, model, train_loader, optimizer, scalar):
    loss_meter = AvgMeter('train')
    # tqdm_object = tqdm(train_loader, total=len(train_loader))

    targets = []
    predictions = []
    for index, batch in enumerate(train_loader):
        batch = batch_constructor(config, batch)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            output, score = model(batch)
            loss = calculate_loss(model, score, batch['label'])
        scalar.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        if (index + 1) % 2:
            scalar.step(optimizer)
            # loss.backward()
            # optimizer.step()
            scalar.update()

        count = batch["id"].size(0)
        loss_meter.update(loss.detach(), count)

        # tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

        prediction = output.detach()
        predictions.append(prediction)

        target = batch['label'].detach()
        targets.append(target)

    return loss_meter, targets, predictions


def validation_epoch(config, model, validation_loader):
    loss_meter = AvgMeter('validation')

    targets = []
    predictions = []
    # tqdm_object = tqdm(validation_loader, total=len(validation_loader))
    for batch in validation_loader:
        batch = batch_constructor(config, batch)
        with torch.no_grad():
            output, score = model(batch)
            loss = calculate_loss(model, score, batch['label'])

            count = batch["id"].size(0)
            loss_meter.update(loss.detach(), count)

            # tqdm_object.set_postfix(validation_loss=loss_meter.avg)

            prediction = output.detach()
            predictions.append(prediction)

            target = batch['label'].detach()
            targets.append(target)

    return loss_meter, targets, predictions


def supervised_train(config, train_loader, validation_loader, trial=None):
    torch.cuda.empty_cache()
    checkpoint_path2 = checkpoint_path = str(config.output_path) + '/checkpoint.pt'
    if trial:
        checkpoint_path2 = str(config.output_path) + '/checkpoint_' + str(trial.number) + '.pt'

    torch.manual_seed(27)
    random.seed(27)
    np.random.seed(27)
    
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    scalar = torch.cuda.amp.GradScaler()
    model = FakeNewsModel(config).to(config.device)

    params = [
        {"params": model.image_encoder.parameters(), "lr": config.image_encoder_lr, "name": 'image_encoder'},
        {"params": model.text_encoder.parameters(), "lr": config.text_encoder_lr, "name": 'text_encoder'},
        {"params": itertools.chain(model.image_projection.parameters(), model.text_projection.parameters()),
         "lr": config.head_lr, "weight_decay": config.head_weight_decay, 'name': 'projection'},
        {"params": model.classifier.parameters(), "lr": config.classification_lr,
         "weight_decay": config.classification_weight_decay,
         'name': 'classifier'}
    ]
    optimizer = torch.optim.AdamW(params, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.factor,
                                                              patience=config.patience // 5, verbose=True)
    early_stopping = EarlyStopping(patience=config.patience, delta=config.delta, path=checkpoint_path, verbose=True)
    checkpoint_saving = CheckpointSaving(path=checkpoint_path, verbose=True)

    train_losses, train_accuracies = [], []
    validation_losses, validation_accuracies = [], []

    validation_accuracy, validation_loss = 0, 1
    for epoch in range(config.epochs):
        print(f"Epoch: {epoch + 1}")
        gc.collect()

        model.train()
        train_loss, train_truth, train_pred = train_epoch(config, model, train_loader, optimizer, scalar)
        model.eval()
        with torch.no_grad():
            validation_loss, validation_truth, validation_pred = validation_epoch(config, model, validation_loader)

        train_accuracy = multiclass_acc(train_truth, train_pred)
        validation_accuracy = multiclass_acc(validation_truth, validation_pred)
        print_lr(optimizer)
        print('Training Loss:', train_loss, 'Training Accuracy:', train_accuracy)
        print('Validation Loss', validation_loss, 'Validation Accuracy:', validation_accuracy)

        if lr_scheduler:
            lr_scheduler.step(validation_loss.avg)
        if early_stopping:
            early_stopping(validation_loss.avg, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        if checkpoint_saving:
            checkpoint_saving(validation_accuracy, model)

        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        validation_accuracies.append(validation_accuracy)
        validation_losses.append(validation_loss)

        if trial:
            trial.report(validation_accuracy, epoch)
            if trial.should_prune():
                print('trial pruned')
                raise optuna.exceptions.TrialPruned()

        print()

    if checkpoint_saving:
        model = FakeNewsModel(config).to(config.device)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            validation_loss, validation_truth, validation_pred = validation_epoch(config, model, validation_loader)
        validation_accuracy = multiclass_acc(validation_pred, validation_truth)
        if trial and validation_accuracy >= config.wanted_accuracy:
            loss_accuracy = pd.DataFrame(
                {'train_loss': train_losses, 'train_accuracy': train_accuracies, 'validation_loss': validation_losses,
                 'validation_accuracy': validation_accuracies})
            torch.save({'model_state_dict': model.state_dict(),
                        'parameters': str(config),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_accuracy': loss_accuracy}, checkpoint_path2)

    if not checkpoint_saving:
        loss_accuracy = pd.DataFrame(
            {'train_loss': train_losses, 'train_accuracy': train_accuracies, 'validation_loss': validation_losses,
             'validation_accuracy': validation_accuracies})
        torch.save(model.state_dict(), checkpoint_path)
        if trial and validation_accuracy >= config.wanted_accuracy:
            torch.save({'model_state_dict': model.state_dict(),
                        'parameters': str(config),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_accuracy': loss_accuracy}, checkpoint_path2)
    try:
        del train_loss
        del train_truth
        del train_pred
        del validation_loss
        del validation_truth
        del validation_pred
        del train_losses
        del train_accuracies
        del validation_losses
        del validation_accuracies
        del loss_accuracy
        del scalar
        del model
        del params
        gc.collect()
        torch.cuda.empty_cache
    except:
        print('Error in deleting caches')
        pass

    return validation_accuracy

