import os
import shutil

import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
from sklearn.utils import class_weight

from datasets import NIH_CXR_Dataset, MIMIC_CXR_Dataset
from utils import set_seed, worker_init_fn, val_worker_init_fn, train, validate, evaluate

def main(args):
    # Set model/output directory name
    MODEL_NAME = args.dataset
    MODEL_NAME += f'_{args.model_name}'
    MODEL_NAME += f'_{args.loss}'
    MODEL_NAME += f'_rand' if args.rand_init else ''
    MODEL_NAME += f'_lr-{args.lr}'
    MODEL_NAME += f'_bs-{args.batch_size}'
    MODEL_NAME += f'x{args.n_gpu}' if args.n_gpu != 1 else ''
    MODEL_NAME += f'_seed-{args.seed}' if args.seed != 0 else ''

    # Create output directory for model (and delete if already exists)
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    model_dir = os.path.join(args.out_dir, MODEL_NAME)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    # Set all seeds for reproducibility
    set_seed(args.seed)

    # Create datasets + loaders
    subset = False
    if args.dataset == 'nih-cxr-lt':
        dataset = NIH_CXR_Dataset
        N_CLASSES = 20
    elif args.dataset == 'mimic-cxr-lt':
        dataset = MIMIC_CXR_Dataset
        N_CLASSES = 19
    else:
        sys.exit(-1)

    train_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='train')
    val_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='val')
    test_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='test', n_TTA=args.n_TTA)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=val_worker_init_fn)

    # Create csv documenting training history
    history = pd.DataFrame(columns=['epoch', 'phase', 'loss', 'auc', 'mAP', 'f1'])
    history.to_csv(os.path.join(model_dir, 'history.csv'), index=False)

    # Set device
    device = torch.device('cuda:0')

    # Instantiate model
    model = torchvision.models.resnet50(pretrained=(not args.rand_init))
    model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)
    print(model)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=[0, 1]).to(device)
    else:
        model = model.to(device)

    # Set loss function and optimizer
    loss_fxn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train with early stopping
    epoch = 1
    early_stopping_dict = {'best_loss': 1e8, 'epochs_no_improve': 0}
    best_model_wts = None
    while epoch <= args.max_epochs and early_stopping_dict['epochs_no_improve'] <= args.patience:
        history = train(model=model, device=device, loss_fxn=loss_fxn, optimizer=optimizer, data_loader=train_loader, history=history, epoch=epoch, model_dir=model_dir, classes=train_dataset.CLASSES, mixup=args.mixup, mixup_alpha=args.mixup_alpha, orig_cls_idx=train_dataset.orig_cls_idx, PIE=args.PIE)
        history, early_stopping_dict, best_model_wts = validate(model=model, device=device, loss_fxn=loss_fxn, optimizer=optimizer, data_loader=val_loader, history=history, epoch=epoch, model_dir=model_dir, early_stopping_dict=early_stopping_dict, best_model_wts=best_model_wts, classes=val_dataset.CLASSES, orig_cls_idx=train_dataset.orig_cls_idx)

        epoch += 1
    
    # Evaluate on test set
    evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=test_dataset, split='test', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts, n_TTA=args.n_TTA)

if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/ssd1/greg/NIH_CXR/images', type=str)
    parser.add_argument('--label_dir', default='labels', type=str)
    parser.add_argument('--out_dir', default='/ssd1/greg/nih_results/', type=str,
                        help="path to directory where results and model weights will be saved (will create model-specific directory within out_dir)")

    parser.add_argument('--dataset', required=True, type=str, choices=['nih-cxr-lt', 'mimic-cxr-lt'])
    parser.add_argument('--loss', default='ce', type=str, choices=['ce'])
    parser.add_argument('--model_name', default='resnet50', type=str, choices=['resnet50'])
    parser.add_argument('--max_epochs', default=60, type=int, help="maximum number of epochs to train")
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--patience', default=15, type=int, help="early stopping 'patience' during training")
    parser.add_argument('--rand_init', default=False, action='store_true')
    parser.add_argument('--n_TTA', default=0, type=int, help="number of augmented copies to use during test-time augmentation (TTA), default 0")
    parser.add_argument('--seed', default=0, type=int, help="set random seed")
    parser.add_argument('--n_gpu', default=1, type=int)

    args = parser.parse_args()

    print(args)

    main(args)