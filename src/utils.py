import os
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def val_worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)

def train(model, device, loss_fxn, optimizer, data_loader, history, epoch, model_dir, classes):
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}')

    running_loss = 0.
    y_true, y_hat = [], []
    for i, (x, y) in pbar:
        x = x.to(device)
        y = y.to(device)

        out = model(x)

        loss = loss_fxn(out, y)

        for param in model.parameters():
            param.grad = None
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        y_hat.append(out.sigmoid().detach().cpu().numpy())
        y_true.append(y.detach().cpu().numpy())

        pbar.set_postfix({'loss': running_loss / (i + 1)})

    # Collect true and predicted labels into flat numpy arrays
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)

    # Compute metrics
    aucs = roc_auc_score(y_true, y_hat, average=None, multi_class='ovr')
    aps = average_precision_score(y_true, y_hat, average=None)
    precs, recalls, f1s, _ = precision_recall_fscore_support(y_true, y_hat.round())

    print(f'Mean AUC: {aucs.mean():.3f} | mAP: {aps.mean():.3f} | Mean F1: {f1s.mean():.3f} | Mean Precision: {precs.mean():.3f} | Mean Recall: {recalls.mean():.3f}')
    print(aucs)

    current_metrics = pd.DataFrame([[epoch, 'train', running_loss / (i + 1), aucs.mean(), aps.mean(), f1s.mean()]], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    return history.append(current_metrics)

def validate(model, device, loss_fxn, optimizer, data_loader, history, epoch, model_dir, early_stopping_dict, best_model_wts, classes):
    model.eval()
    
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[VAL] Epoch {epoch}')

    running_loss = 0.
    y_true, y_hat = [], []
    with torch.no_grad():
        for i, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)

            out = model(x)

            loss = loss_fxn(out, y)

            running_loss += loss.item()

            y_hat.append(out.sigmoid().detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())

            pbar.set_postfix({'loss': running_loss / (i + 1)})

    # Collect true and predicted labels into flat numpy arrays
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)

    # Compute metrics
    aucs = roc_auc_score(y_true, y_hat, average=None, multi_class='ovr')
    aps = average_precision_score(y_true, y_hat, average=None)
    precs, recalls, f1s, _ = precision_recall_fscore_support(y_true, y_hat.round())

    print(f'Mean AUC: {aucs.mean():.3f} | mAP: {aps.mean():.3f} | Mean F1: {f1s.mean():.3f} | Mean Precision: {precs.mean():.3f} | Mean Recall: {recalls.mean():.3f}')
    print(aucs)

    val_loss = running_loss / (i + 1)

    current_metrics = pd.DataFrame([[epoch, 'val', val_loss, aucs.mean(), aps.mean(), f1s.mean()]], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    if val_loss < early_stopping_dict['best_loss']:
        print(f'--- EARLY STOPPING: Loss has improved from {round(early_stopping_dict["best_loss"], 3)} to {round(val_loss, 3)}! Saving weights. ---')
        early_stopping_dict['epochs_no_improve'] = 0
        early_stopping_dict['best_loss'] = val_loss
        best_model_wts = deepcopy(model.state_dict())
        torch.save({'weights': best_model_wts, 'optimizer': optimizer.state_dict(), 'epoch': epoch}, os.path.join(model_dir, f'chkpt.pt'))
    else:
        print(f'--- EARLY STOPPING: AUC has not improved from {round(early_stopping_dict["best_loss"], 3)} ---')
        early_stopping_dict['epochs_no_improve'] += 1

    return history.append(current_metrics), early_stopping_dict, best_model_wts

def evaluate(model, device, loss_fxn, dataset, split, batch_size, history, model_dir, weights, n_TTA=0):
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(weights)  # load best weights
    else:
        model.load_state_dict(weights)  # load best weights
    model.eval()

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8 if split == 'test' else 2, pin_memory=True, worker_init_fn=val_worker_init_fn)

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[{split.upper()}] EVALUATION')

    running_loss = 0.
    y_true, y_hat = [], []
    with torch.no_grad():
        for i, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)

            if n_TTA > 0:
                out = torch.stack([model(x[:, tta_copy, :, :, :]) for tta_copy in range(n_TTA)], dim=0).sigmoid().mean(dim=0)
                y_hat.append(out.detach().cpu().numpy())
            else:
                out = model(x)
                y_hat.append(out.sigmoid().detach().cpu().numpy())

            loss = loss_fxn(out, y)

            running_loss += loss.item()

            y_true.append(y.detach().cpu().numpy())

            pbar.set_postfix({'loss': running_loss / (i + 1)})

    # Collect true and predicted labels into flat numpy arrays
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)

    # Compute metrics
    aucs = roc_auc_score(y_true, y_hat, average=None, multi_class='ovr')
    aps = average_precision_score(y_true, y_hat, average=None)
    precs, recalls, f1s, _ = precision_recall_fscore_support(y_true, y_hat.round())

    print(f'Mean AUC: {aucs.mean():.3f} | mAP: {aps.mean():.3f} | Mean F1: {f1s.mean():.3f} | Mean Precision: {precs.mean():.3f} | Mean Recall: {recalls.mean():.3f}')
    print(aucs)

    # Collect and save true and predicted disease labels for test set
    pred_df = pd.DataFrame(y_hat, columns=dataset.CLASSES)
    true_df = pd.DataFrame(y_true, columns=dataset.CLASSES)
    pred_df.to_csv(os.path.join(model_dir, f'{split}_pred.csv'), index=False)
    true_df.to_csv(os.path.join(model_dir, f'{split}_true.csv'), index=False)

    # Plot loss curves
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'loss'], label='train')
    ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'loss'], label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.savefig(os.path.join(model_dir, 'loss.png'), dpi=300, bbox_inches='tight')
    
    # Plot AUROC learning curves
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'auc'], label='train')
    ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'auc'], label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUROC')
    ax.legend()
    fig.savefig(os.path.join(model_dir, 'auc.png'), dpi=300, bbox_inches='tight')
    
    # Plot mAP learning curves
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'mAP'], label='train')
    ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'mAP'], label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.legend()
    fig.savefig(os.path.join(model_dir, 'mAP.png'), dpi=300, bbox_inches='tight')

    # Create summary text file describing final performance
    summary = f'Mean AUC: {aucs.mean():.3f}\n\n'

    summary += 'Class:| AUC\n'
    for i, c in enumerate(dataset.CLASSES):
        summary += f'{c}:| {aucs[i]:.3f}\n'
    summary += '\n'
    
    summary += f'mAP: {aps.mean():.3f}\n\n'

    summary += 'Class:| AP\n'
    for i, c in enumerate(dataset.CLASSES):
        summary += f'{c}:| {aps[i]:.3f}\n'
    summary += '\n'

    f = open(os.path.join(model_dir, f'{split}_summary.txt'), 'w')
    f.write(summary)
    f.close()

def evaluate_minimal(model, device, loss_fxn, dataset, split, batch_size, model_dir, n_TTA=0, orig_cls_idx=None):
    """Evaluate PyTorch model on test set of NIH ChestXRay14 dataset. Saves training history csv, summary text file, training curves, etc.
    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        ls : int
            Ratio of label smoothing to apply during loss computation
        batch_size : int
        history : pandas DataFrame
            Data frame containing history of training metrics
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        weights : PyTorch state_dict
            Model weights from best epoch
        n_TTA : int
            Number of augmented copies to use for test-time augmentation (0-K)
        fusion : bool
            Whether or not fusion is being performed (image + metadata inputs)
        meta_only : bool
            Whether or not to train on *only* metadata as input
    """
    model.eval()

    ## INFERENCE
    data_loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8 if split == 'test' else 2, pin_memory=True, worker_init_fn=val_worker_init_fn)

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[{split.upper()}] EVALUATION')

    running_loss = 0.
    y_true, y_hat = [], []
    with torch.no_grad():
        for i, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)

            if n_TTA > 0:
                out = torch.stack([model(x[:, tta_copy, :, :, :]) for tta_copy in range(n_TTA)], dim=0).sigmoid().mean(dim=0)
                y_hat.append(out.detach().cpu().numpy())
            else:
                out = model(x)
                y_hat.append(out.sigmoid().detach().cpu().numpy())

            loss = loss_fxn(out, y)

            running_loss += loss.item()

            y_true.append(y.detach().cpu().numpy())

            pbar.set_postfix({'loss': running_loss / (i + 1)})

    # Collect true and predicted labels into flat numpy arrays
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)

    if orig_cls_idx is not None:
        y_true = y_true[:, orig_cls_idx]
        y_hat = y_hat[:, orig_cls_idx]

    # Compute metrics
    aucs = roc_auc_score(y_true, y_hat, average=None, multi_class='ovr')
    precs, recalls, f1s, _ = precision_recall_fscore_support(y_true, y_hat.round())

    print(f'Mean AUC: {aucs.mean():.3f} | Mean F1: {f1s.mean():.3f} | Mean Precision: {precs.mean():.3f} | Mean Recall: {recalls.mean():.3f}')
    print(aucs)

    # Collect and save true and predicted disease labels for test set
    pred_df = pd.DataFrame(y_hat, columns=dataset.CLASSES)
    true_df = pd.DataFrame(y_true, columns=dataset.CLASSES)
    # true_df = pd.DataFrame(LabelBinarizer().fit(range(len(dataset.CLASSES))).transform(y_true), columns=dataset.CLASSES)

    pred_df.to_csv(os.path.join(model_dir, f'{split}_pred.csv'), index=False)
    true_df.to_csv(os.path.join(model_dir, f'{split}_true.csv'), index=False)
    
    # Create summary text file describing final performance
    summary = f'Mean AUC: {aucs.mean():.3f}\n\n'

    summary += 'Class:| AUC\n'
    for i, c in enumerate(dataset.CLASSES):
        summary += f'{c}:| {aucs[i]:.3f}\n'
    summary += '\n'
    
    f = open(os.path.join(model_dir, f'{split}_summary.txt'), 'w')
    f.write(summary)
    f.close()

def evaluate_prune(model, device, dataset, split, batch_size, n_TTA=0):
    model.eval()

    data_loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=val_worker_init_fn)

    y_true, y_hat = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)

            if n_TTA > 0:
                out = torch.stack([model(x[:, tta_copy, :, :, :]) for tta_copy in range(n_TTA)], dim=0).sigmoid().mean(dim=0)
                y_hat.append(out.detach().cpu().numpy())
            else:
                out = model(x)
                y_hat.append(out.sigmoid().detach().cpu().numpy())

            y_true.append(y.detach().cpu().numpy())

    # Collect true and predicted labels into flat numpy arrays
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)

    y_true = pd.DataFrame(y_true, columns=dataset.CLASSES)
    y_hat = pd.DataFrame(y_hat, columns=dataset.CLASSES)

    y_hat['label'] = y_true.apply(lambda x: np.where(x == 1)[0], axis=1)
    y_hat['label'] = y_hat['label'].apply(lambda x: '|'.join(list(np.array(dataset.CLASSES)[np.array(np.matrix(x)).squeeze(axis=0)])))

    return y_hat

