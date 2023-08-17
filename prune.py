import os
import shutil
import sys

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.utils.prune as prune
import torchvision

from datasets import NIH_CXR_Dataset, MIMIC_CXR_Dataset
from utils import evaluate_prune

def main(args):
    set_seed(0)

    if args.dataset == 'mimic-cxr-lt':
        N_CLASSES = 19
        dataset = MIMIC_CXR_Dataset
        dataset_name = 'mimic'
    elif args.dataset == 'nih-cxr-lt':
        N_CLASSES = 20
        dataset = NIH_CXR_Dataset
        dataset_name = 'nih'
    else:
        sys.exit(-1)
    
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    device = 'cuda:0'

    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)

    weights = torch.load(os.path.join(args.model_dir, 'chkpt.pt'))['weights']
    msg = model.load_state_dict(weights, strict=True)
    model = model.to(device)

    # Get predictions on validation set
    val_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=val_worker_init_fn)

    res = evaluate_prune(model=model, device=device, dataset=val_dataset, split='val', batch_size=args.batch_size, n_TTA=0)
    res.to_pickle(os.path.join(out_dir, f'{dataset_name}-cxr-lt_resnet50_seed-0_prune-0_val.pkl'))

    # Get predictions on test set at various sparsity ratios
    test_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=val_worker_init_fn)

    sparsity_ratios = np.array(range(0, 100, 5)) / 100
    for seed in tqdm.tqdm(range(args.n_seeds), desc='SEEDS COMPLETED'):
        # Instantiate model
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)

        extra_str = f'_seed-{seed}' if seed != 0 else ''
        weights = torch.load(os.path.join(args.model_dir + extra_str, 'chkpt.pt'))['weights']

        msg = model.load_state_dict(weights, strict=True)
        model = model.to(device)

        # Run inference on test set and save predictions
        res = evaluate_prune(model=model, device=device, dataset=test_dataset, split='test', batch_size=args.batch_size, n_TTA=0)
        res.to_pickle(os.path.join(out_dir, f'{dataset_name}-cxr-lt_resnet50_seed-{seed}_prune-0.pkl'))

        for ratio in tqdm.tqdm(sparsity_ratios[1:], desc='PRUNING RATIOS COMPLETED'):
            # Re-instantiate model
            model = torchvision.models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)

            msg = model.load_state_dict(weights, strict=True)  # reload best weights
            model = model.to(device)

            if args.prune_type == 'L1':
                ## L1 PRUNING ##
                params_to_prune = []
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        params_to_prune.append((module, 'weight'))
                    elif isinstance(module, torch.nn.Linear):
                        params_to_prune.append((module, 'weight'))

                prune.global_unstructured(
                    params_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=ratio
                )
            elif args.prune_type == 'random':
                ## RANDOM PRUNING ##
                params_to_prune = []
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        params_to_prune.append((module, 'weight'))
                    elif isinstance(module, torch.nn.Linear):
                        params_to_prune.append((module, 'weight'))

                prune.global_unstructured(
                    params_to_prune,
                    pruning_method=prune.RandomUnstructured,
                    amount=ratio
                )
            else:
                sys.exit(-1)

            # Run inference on test set with pruned model and save predictions
            res = evaluate_prune(model=model, device=device, dataset=test_dataset, split='test', batch_size=args.batch_size, n_TTA=0)
            res.to_pickle(os.path.join(out_dir, f'{dataset_name}-cxr-lt_resnet50_seed-{seed}_prune-{ratio}.pkl'))

if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/ssd1/greg/NIH_CXR/images', type=str)
    parser.add_argument('--label_dir', default='labels/', type=str)
    parser.add_argument('--out_dir', required=True, type=str, help='path to directory where results will be saved')
    parser.add_argument('--model_dir', required=True, default='trained_models', type=str, help='path to directory with model weights')
    parser.add_argument('--dataset', default='nih-cxr-lt', type=str, choices=['nih-cxr-lt', 'mimic-cxr-lt'])

    parser.add_argument('--prune_type', type=str, default='L1', choices=['L1', 'random'])
    parser.add_argument('--n_seeds', type=int, default=30)
    parser.add_argument('--model_name', default='resnet50', type=str, choices=['resnet50'])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--seed', default=0, type=int, help="set random seed")

    args = parser.parse_args()

    print(args)

    main(args)