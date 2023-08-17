import os
import shutil
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torchvision
import torch
import tqdm
from scipy.spatial.distance import pdist, cosine, euclidean, correlation
from scipy.stats import spearmanr, ttest_ind
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

def evaluate(y_true, y_hat, classes):
    y_true_ = y_true[classes]
    y_hat_ = y_hat[classes]

    aucs = roc_auc_score(y_true_, y_hat_, average=None, multi_class='ovr')
    aps = average_precision_score(y_true_, y_hat_, average=None)

    precs, recalls, f1s, _ = precision_recall_fscore_support(y_true_, y_hat_.round(), zero_division=0)  # TODO: get class-optimal thresholds based on val preds

    return pd.DataFrame({'aucs': aucs, 'aps': aps, 'precs': precs, 'recalls': recalls, 'f1s': f1s})

def main(args):
    NIH_CLASSES = [
        'No Finding',
        'Infiltration',
        'Effusion',
        'Atelectasis',
        'Nodule',
        'Mass',
        'Consolidation',
        'Pneumothorax',
        'Pleural Thickening',
        'Cardiomegaly',
        'Emphysema',
        'Edema',
        'Fibrosis',
        'Subcutaneous Emphysema',
        'Pneumonia',
        'Tortuous Aorta',
        'Calcification of the Aorta',
        'Pneumoperitoneum',
        'Hernia',
        'Pneumomediastinum'
    ]
    MIMIC_CLASSES = [
        'Support Devices',
        'Lung Opacity',
        'Cardiomegaly',  # shared
        'Pleural Effusion',  # shared
        'Atelectasis',  # shared
        'Pneumonia',  # shared
        'Edema',  # shared
        'No Finding',  # shared
        'Enlarged Cardiomediastinum',
        'Consolidation',  # shared
        'Pneumothorax',  # shared
        'Fracture',
        'Calcification of the Aorta',  # shared
        'Tortuous Aorta',  # shared
        'Subcutaneous Emphysema',  # shared
        'Lung Lesion',
        'Pneumomediastinum',  # shared
        'Pneumoperitoneum',  # shared
        'Pleural Other'
    ]

    ### OVERALL ANALYSIS OF THE EFFECT OF PRUNING ###

    # Set sparsity ratios [0, 0.05, ..., 0.9, 0.95]
    sparsity_ratios = np.array(range(0, 100, 5)) / 100

    # Read in NIH test set ground truth
    nih_y_test = pd.read_csv(os.path.join(args.nih_model_dir, 'test_true.csv'))

    # Read in all NIH predictions (stratified by random seed + pruning ratio) + perform evaluation
    nih_pred_dict = {}
    for seed in tqdm.tqdm(range(args.n_seeds), desc='Reading in pruning predictions'):
        for ratio in sparsity_ratios:
            if ratio == 0:
                y_hat = pd.read_pickle(os.path.join(args.nih_l1_prune_dir, f'nih-cxr-lt_resnet50_seed-{seed}_prune-{int(ratio)}.pkl'))
            else:
                y_hat = pd.read_pickle(os.path.join(args.nih_l1_prune_dir, f'nih-cxr-lt_resnet50_seed-{seed}_prune-{ratio}.pkl'))

            res = evaluate(nih_y_test, y_hat, NIH_CLASSES)
            nih_pred_dict[str(seed), str(int(ratio*100)), 'L1'] = res

            if args.nih_rand_prune_dir != '':
                if ratio == 0:
                    y_hat = pd.read_pickle(os.path.join(args.nih_rand_prune_dir, f'nih-cxr-lt_resnet50_seed-{seed}_prune-{int(ratio)}.pkl'))
                else:
                    y_hat = pd.read_pickle(os.path.join(args.nih_rand_prune_dir, f'nih-cxr-lt_resnet50_seed-{seed}_prune-{ratio}.pkl'))

                res = evaluate(nih_y_test, y_hat, NIH_CLASSES)
                nih_pred_dict[str(seed), str(int(ratio*100)), 'Random'] = res

    # Read in MIMIC test set ground truth
    mimic_y_test = pd.read_csv(os.path.join(args.mimic_model_dir, 'test_true.csv'))

    # Read in all MIMIC predictions (stratified by random seed + pruning ratio) + perform evaluation
    mimic_pred_dict = {}
    for seed in tqdm.tqdm(range(args.n_seeds), desc='Reading in pruning predictions'):
        for ratio in sparsity_ratios:
            if ratio == 0:
                y_hat = pd.read_pickle(os.path.join(args.mimic_l1_prune_dir, f'mimic-cxr-lt_resnet50_seed-{seed}_prune-{int(ratio)}.pkl'))
            else:
                y_hat = pd.read_pickle(os.path.join(args.mimic_l1_prune_dir, f'mimic-cxr-lt_resnet50_seed-{seed}_prune-{ratio}.pkl'))

            res = evaluate(mimic_y_test, y_hat, MIMIC_CLASSES)
            mimic_pred_dict[str(seed), str(int(ratio*100)), 'L1'] = res

            if args.mimic_rand_prune_dir != '':
                if ratio == 0:
                    y_hat = pd.read_pickle(os.path.join(args.mimic_rand_prune_dir, f'mimic-cxr-lt_resnet50_seed-{seed}_prune-{int(ratio)}.pkl'))
                else:
                    y_hat = pd.read_pickle(os.path.join(args.mimic_rand_prune_dir, f'mimic-cxr-lt_resnet50_seed-{seed}_prune-{ratio}.pkl'))

                res = evaluate(mimic_y_test, y_hat, MIMIC_CLASSES)
                mimic_pred_dict[str(seed), str(int(ratio*100)), 'Random'] = res

    # Create data frame of overall metrics by dataset, seed, and sparsity ratio
    overall_metrics = pd.concat([nih_overall_metrics, mimic_overall_metrics], axis=0)
    overall_metrics['dataset'] = ['NIH-CXR-LT'] * nih_overall_metrics.shape[0] + ['MIMIC-CXR-LT'] * mimic_overall_metrics.shape[0]

    # Plot overall effect of pruning on both datasets
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    sns.lineplot(data=overall_metrics[overall_metrics['prune'] == 'L1'], ax=ax, x='ratio', y='mAP', hue='dataset', estimator=np.median, err_style=None, ci=None)
    ax.set_xlabel('Sparsity Ratio', fontsize=11)
    ax.set_ylabel('Mean AP', fontsize=11)
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.set_ylim([0.05, 0.375])
    ax.legend(title='', fontsize=11)
    fig.tight_layout()
    fig.savefig('overall_prune.pdf', bbox_inches='tight')

    ### (END) OVERALL ANALYSIS OF THE EFFECT OF PRUNING ###

    ### CLASS-LEVEL ANALYSIS OF THE EFFECT OF PRUNING ###

    # Gather class-level NIH performance metrics by seed and sparsity ratio
    seeds = []
    ratios = []
    aucs = []
    aps = []
    f1s = []
    labels = []
    prunes = []
    nih_prune_types = ['L1', 'Random'] if args.nih_rand_prune_dir != '' else ['L1']
    for seed in range(args.n_seeds):
        for ratio in sparsity_ratios:
            for prune in nih_prune_types:
                metrics = nih_pred_dict[str(seed), str(int(ratio*100)), prune]

                for i, c in enumerate(NIH_CLASSES):
                    seeds.append(seed)
                    ratios.append(ratio)    
                    
                    aucs.append(metrics['aucs'][i])
                    aps.append(metrics['aps'][i])
                    f1s.append(metrics['f1s'][i])
                    labels.append(c)
                    prunes.append(prune)
    nih_class_metrics = pd.DataFrame({'seed': seeds, 'ratio': ratios, 'prune': prunes, 'label': labels, 'auc': aucs, 'ap': aps, 'f1': f1s})

    # Get relative change in performance metrics btw uncompressed and pruned models for each sparsity ratio for NIH
    nih_diff_class_metrics = []
    for ratio in sparsity_ratios[1:]:
        diff_data = (nih_class_metrics[nih_class_metrics['ratio'] == ratio][['auc', 'ap', 'f1']].reset_index(drop=True) - nih_class_metrics[nih_class_metrics['ratio'] == 0][['auc', 'ap', 'f1']].reset_index(drop=True)) / (nih_class_metrics[nih_class_metrics['ratio'] == 0][['auc', 'ap', 'f1']].reset_index(drop=True) + 1e-12)
        diff_data[['seed', 'label', 'prune']] = nih_class_metrics[nih_class_metrics['ratio'] == 0][['seed', 'label', 'prune']].reset_index(drop=True)
        diff_data['ratio'] = ratio

        nih_diff_class_metrics.append(diff_data)
    nih_diff_class_metrics = pd.concat(nih_diff_class_metrics, axis=0).reset_index(drop=True)

    # Gather class-level MIMIC performance metrics by seed and sparsity ratio
    seeds = []
    ratios = []
    aucs = []
    aps = []
    f1s = []
    labels = []
    prunes = []
    mimic_prune_types = ['L1', 'Random'] if args.mimic_rand_prune_dir != '' else ['L1']
    for seed in range(args.n_seeds):
        for ratio in sparsity_ratios:
            for prune in mimic_prune_types:
                metrics = mimic_pred_dict[str(seed), str(int(ratio*100)), prune]

                for i, c in enumerate(MIMIC_CLASSES):
                    seeds.append(seed)
                    ratios.append(ratio)    
                    
                    aucs.append(metrics['aucs'][i])
                    aps.append(metrics['aps'][i])
                    f1s.append(metrics['f1s'][i])
                    labels.append(c)
                    prunes.append(prune)
    mimic_class_metrics = pd.DataFrame({'seed': seeds, 'ratio': ratios, 'prune': prunes, 'label': labels, 'auc': aucs, 'ap': aps, 'f1': f1s})

    # Get relative change in performance metrics btw uncompressed and pruned models for each sparsity ratio for NIH
    mimic_diff_class_metrics = []
    for ratio in sparsity_ratios[1:]:
        diff_data = (mimic_class_metrics[mimic_class_metrics['ratio'] == ratio][['auc', 'ap', 'f1']].reset_index(drop=True) - mimic_class_metrics[mimic_class_metrics['ratio'] == 0][['auc', 'ap', 'f1']].reset_index(drop=True)) / (mimic_class_metrics[mimic_class_metrics['ratio'] == 0][['auc', 'ap', 'f1']].reset_index(drop=True) + 1e-12)
        diff_data[['seed', 'label', 'prune']] = mimic_class_metrics[mimic_class_metrics['ratio'] == 0][['seed', 'label', 'prune']].reset_index(drop=True)
        diff_data['ratio'] = ratio

        mimic_diff_class_metrics.append(diff_data)
    mimic_diff_class_metrics = pd.concat(mimic_diff_class_metrics, axis=0).reset_index(drop=True)

    # Plot representative "forgettability curves" for each dataset: relative change in AP (median across 30 runs) for a given class across sparsity ratios
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3.5), sharey=True)

    keep_classes = ['No Finding', 'Infiltration', 'Emphysema', 'Subcutaneous Emphysema', 'Hernia', 'Pneumomediastinum']
    sns.lineplot(ax=ax[0], data=nih_diff_class_metrics[(nih_diff_class_metrics['prune'] == 'L1') & (nih_diff_class_metrics['label'].isin(keep_classes))], estimator=np.median, x='ratio', y='ap', hue='label', hue_order=keep_classes, ci=None, linewidth=3, alpha=0.75)
    ax[0].set_xlabel('Sparsity Ratio', fontsize=13)
    ax[0].set_ylabel('Relative Change in AP', fontsize=13)
    ax[0].set_title('NIH-CXR-LT', fontsize=14)

    keep_classes_ = ['No Finding', 'Infiltration', 'Emphysema', 'Subcutaneous\nEmphysema', 'Hernia', 'Pneumomediastinum']
    leg = ax[0].legend(labels=[c for c in keep_classes_], loc='lower left', ncol=1, fontsize=10)
    leg.set_title('Class', prop={'size': 11})
    ax[0].set_xlim([0.25, 0.975])

    keep_classes = ['Support Devices', 'Pneumonia', 'No Finding', 'Fracture', 'Subcutaneous Emphysema', 'Pneumomediastinum']
    sns.lineplot(ax=ax[1], data=mimic_diff_class_metrics[(mimic_diff_class_metrics['prune'] == 'L1') & (mimic_diff_class_metrics['label'].isin(keep_classes))], estimator=np.median, x='ratio', y='ap', hue='label', hue_order=keep_classes, ci=None, linewidth=3, alpha=0.75)
    ax[1].set_xlabel('Sparsity Ratio', fontsize=13)
    ax[1].set_ylabel('Relative Change in AP', fontsize=13)
    ax[1].set_title('MIMIC-CXR-LT', fontsize=14)

    keep_classes_ = ['Support Devices', 'Pneumonia', 'No Finding', 'Fracture', 'Subcutaneous\nEmphysema', 'Pneumomediastinum']
    leg = ax[1].legend(labels=[c for c in keep_classes_], loc='lower left', ncol=1, fontsize=10)
    leg.set_title('Class', prop={'size': 11})
    ax[1].set_xlim([0.25, 0.975])

    fig.tight_layout()
    fig.savefig('forgettability_curves_subset.pdf', bbox_inches='tight')

    ### (END) CLASS-LEVEL ANALYSIS OF THE EFFECT OF PRUNING ###

    ### RELATIONSHIP BETWEEN CLASS FREQUENCY AND "FORGETTABILITY"/IMPACT OF PRUNING ###

    # Load NIH test labels
    nih_y_test = pd.read_csv(os.path.join(args.label_dir, f'121722_nih-cxr-lt_labels_test.csv'))
    nih_label_df = pd.DataFrame(nih_y_test[NIH_CLASSES].sum(0).astype(int), columns=['test_freq'])

    nih_y_train = pd.read_csv(os.path.join(args.label_dir, f'121722_nih-cxr-lt_labels_train.csv'))

    nih_label_df['train_freq'] = nih_y_train[NIH_CLASSES].sum(0)

    # Co-occurrence matrix of NIH train dataset
    coocurrence = np.dot(nih_y_train[NIH_CLASSES].transpose(), nih_y_train[NIH_CLASSES])
    coocurrence = coocurrence / np.diagonal(coocurrence)[:, None]

    nih_train_co_mtx = pd.DataFrame(coocurrence, index=NIH_CLASSES, columns=NIH_CLASSES)
    nih_train_w_co_mtx = nih_train_co_mtx * nih_label_df['train_freq']  # "weighted" co-occurrence matrix

    # Co-occurrence matrix of NIH test dataset
    coocurrence = np.dot(nih_y_test[NIH_CLASSES].transpose(), nih_y_test[NIH_CLASSES])
    coocurrence = coocurrence / np.diagonal(coocurrence)[:, None]

    nih_test_co_mtx = pd.DataFrame(coocurrence, index=NIH_CLASSES, columns=NIH_CLASSES)
    nih_test_w_co_mtx = nih_test_co_mtx * nih_label_df['test_freq']  # "weighted" co-occurrence matrix

    # Get median (across 30 runs) relative change in performance for each class and sparsity ratio for NIH
    nih_delta_df = nih_diff_class_metrics.groupby(['label', 'ratio', 'prune'], as_index=False).agg({'auc': 'median', 'ap': 'median', 'prune': 'first'})

    # For each NIH class, get first sparsity ratio at which performance drops by 20%
    drop_20_dict = {}
    for c in NIH_CLASSES:
        sub_df = nih_delta_df[(nih_delta_df['label'] == c) & (nih_delta_df['prune'] == 'L1')].reset_index(drop=True)

        first_drop_idx = np.where(sub_df['ap'] < -0.2)[0][0]
        drop_20_dict[c] = sub_df.loc[first_drop_idx, 'ratio']

    nih_delta_df['test_freq'] = nih_delta_df['label'].map(nih_label_df['test_freq'])
    nih_delta_df['train_freq'] = nih_delta_df['label'].map(nih_label_df['train_freq'])
    nih_delta_df['first_drop_20_percent'] = nih_delta_df['label'].map(drop_20_dict)

    # For NIH, extract subset of relative performance changes at 95% sparsity under L1 pruning
    nih_extreme_delta_df_L1 = nih_delta_df[(nih_delta_df['ratio'] == 0.95) & (nih_delta_df['prune'] == 'L1')]
    nih_extreme_delta_df_L1['log_train_freq'] = np.log(nih_extreme_delta_df_L1['train_freq'])

    # Load MIMIC test data
    mimic_y_test = pd.read_csv(os.path.join(args.label_dir, f'121722_mimic-cxr-lt_labels_test.csv'))
    mimic_label_df = pd.DataFrame(mimic_y_test[MIMIC_CLASSES].sum(0).astype(int), columns=['test_freq'])

    mimic_y_train = pd.read_csv(os.path.join(args.label_dir, f'121722_mimic-cxr-lt_labels_train.csv'))

    mimic_label_df['train_freq'] = mimic_y_train[MIMIC_CLASSES].sum(0)

    # Co-occurrence matrix of MIMIC train dataset
    coocurrence = np.dot(mimic_y_train[MIMIC_CLASSES].transpose(), mimic_y_train[MIMIC_CLASSES])
    coocurrence = coocurrence / np.diagonal(coocurrence)[:, None]

    mimic_train_co_mtx = pd.DataFrame(coocurrence, index=MIMIC_CLASSES, columns=MIMIC_CLASSES)
    mimic_train_w_co_mtx = mimic_train_co_mtx * mimic_label_df['train_freq']  # "weighted" co-occurrence matrix

    # Co-occurrence matrix of MIMIC test dataset
    coocurrence = np.dot(mimic_y_test[MIMIC_CLASSES].transpose(), mimic_y_test[MIMIC_CLASSES])
    coocurrence = coocurrence / np.diagonal(coocurrence)[:, None]

    mimic_test_co_mtx = pd.DataFrame(coocurrence, index=MIMIC_CLASSES, columns=MIMIC_CLASSES)
    mimic_test_w_co_mtx = mimic_test_co_mtx * mimic_label_df['test_freq']  # "weighted" co-occurrence matrix

    # Get median (across 30 runs) relative change in performance for each class and sparsity ratio for MIMIC
    mimic_delta_df = mimic_diff_class_metrics.groupby(['label', 'ratio', 'prune'], as_index=False).agg({'auc': 'median', 'ap': 'median', 'prune': 'first'})

    # For each MIMIC class, get first sparsity ratio at which performance drops by 20%
    drop_20_dict = {}
    for c in MIMIC_CLASSES:
        sub_df = mimic_delta_df[(mimic_delta_df['label'] == c) & (mimic_delta_df['prune'] == 'L1')].reset_index(drop=True)

        first_drop_idx = np.where(sub_df['ap'] < -0.2)[0][0]
        drop_20_dict[c] = sub_df.loc[first_drop_idx, 'ratio']

    # For MIMIC, extract subset of relative performance changes at 95% sparsity under L1 pruning
    mimic_delta_df['test_freq'] = mimic_delta_df['label'].map(mimic_label_df['test_freq'])
    mimic_delta_df['train_freq'] = mimic_delta_df['label'].map(mimic_label_df['train_freq'])
    mimic_delta_df['first_drop_20_percent'] = mimic_delta_df['label'].map(drop_20_dict)

    # For MIMIC, extract subset of relative performance changes at 95% sparsity under L1 pruning
    mimic_extreme_delta_df_L1 = mimic_delta_df[(mimic_delta_df['ratio'] == 0.95) & (mimic_delta_df['prune'] == 'L1')]
    mimic_extreme_delta_df_L1['log_train_freq'] = np.log(mimic_extreme_delta_df_L1['train_freq'])
    mimic_extreme_delta_df_L1.sort_values('ap')

    # Plot association between class frequency (log-transformed) and first sparsity ratio with a 20% drop in AP for each class and dataset
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 5))

    sns.regplot(ax=ax[0,0], data=nih_extreme_delta_df_L1, x='first_drop_20_percent', y='log_train_freq', scatter=True, ci=None, truncate=False, scatter_kws={'s': 75, 'color': 'tab:blue'}, line_kws={'linestyle': '--', 'color': 'orange'})
    ax[0,0].set_xlabel('')
    ax[0,0].set_ylabel('Log Class Frequency', fontsize=13)

    ax[0,0].text(0.7, 10.3, r'$r=0.64, \rho=0.61$', fontsize=11)

    ax[0,0].annotate('No Finding', xy=(0.90, 10.706050), xycoords='data', xytext=(-20, -15), textcoords='offset points', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightgrey', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'), fontsize=9)
    ax[0,0].annotate('Hernia', xy=(0.7, 4.867534), xycoords='data', xytext=(25, -5), textcoords='offset points', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightgrey', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'), fontsize=9)

    sns.regplot(ax=ax[0,1], data=nih_extreme_delta_df_L1, x='ap', y='log_train_freq', scatter=True, ci=None, color='orange', truncate=False, scatter_kws={'s': 75, 'color': 'tab:blue'}, line_kws={'linestyle': '--', 'color': 'orange'})
    ax[0,1].set_xlabel('')
    ax[0,1].set_ylabel('')
    ax[0,1].set_yticklabels([])

    ax[0,1].text(-0.98, 10.3, r'$r=0.80, \rho=0.75$', fontsize=11)


    ax[0,1].annotate('No Finding', xy=(-0.412296, 10.706050), xycoords='data', xytext=(-20, -15), textcoords='offset points', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightgrey', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'), fontsize=9)
    ax[0,1].annotate('Hernia', xy=(-0.978412, 4.867534), xycoords='data', xytext=(40, -5), textcoords='offset points', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightgrey', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'), fontsize=9)

    sns.regplot(ax=ax[1,0], data=mimic_extreme_delta_df_L1, x='first_drop_20_percent', y='log_train_freq', scatter=True, ci=None, truncate=False, scatter_kws={'s': 75, 'color': 'tab:blue'}, line_kws={'linestyle': '--', 'color': 'orange'})
    ax[1,0].set_xlabel('First Sparsity Ratio with\na 20% Drop in AP', fontsize=12)
    ax[1,0].set_ylabel('Log Class Frequency', fontsize=13)
    ax[1,0].set_ylim([5.4, 12])

    ax[1,0].text(0.6, 11.25, r'$r=0.82, \rho=0.93$', fontsize=11)

    ax[1,0].annotate('Support\nDevices', xy=(0.95, 11.206957), xycoords='data', xytext=(0, -55), textcoords='offset points', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightgrey', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'), fontsize=8)
    ax[1,0].annotate('Pneumoperitoneum', xy=(0.6, 6.104793), xycoords='data', xytext=(0, 35), textcoords='offset points', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightgrey', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'), fontsize=8)

    sns.regplot(ax=ax[1,1], data=mimic_extreme_delta_df_L1, x='ap', y='log_train_freq', scatter=True, ci=None, color='orange', truncate=False, scatter_kws={'s': 75, 'color': 'tab:blue'}, line_kws={'linestyle': '--', 'color': 'orange'})
    ax[1,1].set_xlabel('Relative Change in AP\nat 95% Sparsity', fontsize=12)
    ax[1,1].set_ylabel('')
    ax[1,1].set_yticklabels([])
    ax[1,1].set_ylim([5.4, 12])

    ax[1,1].text(-0.98, 11.25, r'$r=0.76, \rho=0.75$', fontsize=11)

    ax[1,1].annotate('Support\nDevices', xy=(-0.366953, 11.206957), xycoords='data', xytext=(-5, -55), textcoords='offset points', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightgrey', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'), fontsize=8)
    ax[1,1].annotate('Pneumoperitoneum', xy=(-0.935696, 6.104793), xycoords='data', xytext=(20, 5), textcoords='offset points', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightgrey', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'), fontsize=8)

    ax[0,0].set_title('NIH-CXR-LT', fontsize=14)
    ax[0,1].set_title('NIH-CXR-LT', fontsize=14)
    ax[1,0].set_title('MIMIC-CXR-LT', fontsize=14)
    ax[1,1].set_title('MIMIC-CXR-LT', fontsize=14)

    fig.tight_layout()
    fig.savefig('class_prune_correlations.pdf', bbox_inches='tight')

    ### (END) RELATIONSHIP BETWEEN CLASS FREQUENCY AND "FORGETTABILITY"/IMPACT OF PRUNING ###

    ### RELATIONSHIP BETWEEN CLASS CO-OCCURRENCE AND IMPACT OF PRUNING ON CLASS PERFORMANCE ###

    # For NIH, get vector of median (across 30 runs) relative changes in AP at each sparsity ratio for L1 pruning ["forgettability curve"]
    nih_rel_delta_ap_vector_dict = {}
    for c in NIH_CLASSES:
        rel_delta_ap_vector = nih_delta_df.loc[(nih_delta_df['label'] == c) & (nih_delta_df['prune'] == 'L1'), 'ap'].values
        nih_rel_delta_ap_vector_dict[c] = rel_delta_ap_vector
    nih_rel_delta_ap_vector_dict

    # Get NIH train and test set co-occurrence matrices
    nih_train_coocurrence = np.dot(nih_y_train[NIH_CLASSES].transpose(), nih_y_train[NIH_CLASSES])
    nih_train_coocurrence = pd.DataFrame(nih_train_coocurrence, index=NIH_CLASSES, columns=NIH_CLASSES)
    nih_test_coocurrence = np.dot(nih_y_test[NIH_CLASSES].transpose(), nih_y_test[NIH_CLASSES])
    nih_test_coocurrence = pd.DataFrame(nih_test_coocurrence, index=NIH_CLASSES, columns=NIH_CLASSES)

    # For each unique pair of NIH classes, compute the Intersection over Union (IoU),  Euclidean distance between their forgettability curves ["FCD"], and frequency of each class
    disease_As = []
    disease_Bs = []
    FCDs = []
    train_freq_As = []
    train_freq_Bs = []
    train_ious = []
    test_freq_As = []
    test_freq_Bs = []
    test_ious = []
    for comb in combinations(NIH_CLASSES, 2):

        rel_delta_ap_vector_A = nih_rel_delta_ap_vector_dict[comb[0]]
        rel_delta_ap_vector_B = nih_rel_delta_ap_vector_dict[comb[1]]

        fcd = euclidean(rel_delta_ap_vector_A, rel_delta_ap_vector_B)

        train_iou = nih_train_coocurrence.loc[comb[0], comb[1]] / (nih_train_coocurrence[comb[0]].sum() + nih_train_coocurrence[comb[1]].sum() - nih_train_coocurrence.loc[comb[0], comb[1]])
        test_iou = nih_test_coocurrence.loc[comb[0], comb[1]] / (nih_test_coocurrence[comb[0]].sum() + nih_test_coocurrence[comb[1]].sum() - nih_test_coocurrence.loc[comb[0], comb[1]])

        train_freq_A = nih_train_coocurrence[comb[0]].sum()
        train_freq_B = nih_train_coocurrence[comb[1]].sum()
        test_freq_A = nih_test_coocurrence[comb[0]].sum()
        test_freq_B = nih_test_coocurrence[comb[1]].sum()

        disease_As.append(comb[0])
        disease_Bs.append(comb[1])
        FCDs.append(fcd)
        train_ious.append(train_iou)
        test_ious.append(test_iou)
        train_freq_As.append(train_freq_A)
        train_freq_Bs.append(train_freq_B)
        test_freq_As.append(test_freq_A)
        test_freq_Bs.append(test_freq_B)
    nih_df = pd.DataFrame({'disease_A': disease_As, 'disease_B': disease_Bs, 'train_freq_A': train_freq_As, 'train_freq_B': train_freq_Bs, 'test_freq_A': test_freq_As,
                           'test_freq_B': test_freq_Bs, 'train_iou': train_ious, 'test_iou': test_ious, 'FCD': FCDs})

    # Compute difference in (log-transformed) class frequency for each pair of NIH classes
    nih_df['train_freq_diff'] = (nih_df['train_freq_A'] - nih_df['train_freq_B']).abs()
    nih_df['log_train_freq_A'] = np.log(nih_df['train_freq_A'])
    nih_df['log_train_freq_B'] = np.log(nih_df['train_freq_B'])
    nih_df['log_train_freq_diff'] = (nih_df['log_train_freq_A'] - nih_df['log_train_freq_B']).abs()

    nih_df['test_freq_diff'] = (nih_df['test_freq_A'] - nih_df['test_freq_B']).abs()
    nih_df['log_test_freq_A'] = np.log(nih_df['test_freq_A'])
    nih_df['log_test_freq_B'] = np.log(nih_df['test_freq_B'])
    nih_df['log_test_freq_diff'] = (nih_df['log_test_freq_A'] - nih_df['log_test_freq_B']).abs()

    # Compute scaled IoU (for aid of visualization) for each pair of NIH classes
    nih_df['scaled_train_iou'] = nih_df['train_iou'] ** (1/4)
    nih_df['scaled_test_iou'] = nih_df['test_iou'] ** (1/4)

    # Plot relationship btw FCD and (a) absolute difference in log class frequency and (b) scaled IoU between classes for each unique pair of NIH classes
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), sharey=True)
    sns.regplot(ax=ax[0], data=nih_df, x='log_test_freq_diff', y='FCD', scatter=True, ci=None, truncate=False, scatter_kws={'s': 50, 'color': 'tab:blue'}, line_kws={'linestyle': '--', 'color': 'orange'})
    ax[0].set_xlabel('|Difference in Log Class Frequency|', fontsize=13)
    ax[0].set_ylabel('Forgettability Curve\nDissimilarity (FCD)', fontsize=13)
    ax[0].text(0, 1.42, r'$r=0.71, \rho=0.64$', fontsize=12)

    ax[0].scatter(0.091192, 0.062937, s=50, color='red', label='(Emphysema, Subcutaneous Emphysema)')
    ax[0].scatter(2.044142, 0.177101, s=50, color='black', label='(Emphysema, Pneumomediastinum)')
    ax[0].scatter(4.583549, 1.473196, s=50, color='orange', label='(Infiltration, Hernia)')

    sns.regplot(ax=ax[1], data=nih_df, x='scaled_test_iou', y='FCD', scatter=True, ci=None, truncate=False, scatter_kws={'s': 50, 'color': 'tab:blue'}, line_kws={'linestyle': '--', 'color': 'orange'})
    ax[1].set_xlabel('(IoU Between Classes)$^{1/4}$', fontsize=13)
    ax[1].set_ylabel('')
    ax[1].text(0.325, 1.42, r'$r=$-$0.43, \rho=$-$0.47$', fontsize=12)

    ax[1].scatter(0.625797, 0.062937, s=50, color='red', label='(Emphysema, Subcutaneous Emphysema)')
    ax[1].scatter(0.372540, 0.177101, s=50, color='black', label='(Emphysema, Pneumomediastinum)')
    ax[1].scatter(0.153988, 1.473196, s=50, color='orange', label='(Infiltration, Hernia)')
    ax[1].legend()
    ax[1].get_legend().remove()

    fig.tight_layout()

    handles, labels = ax[1].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.26)
    fig.legend(handles, labels, fontsize=9, loc='lower center', ncol=3)

    fig.savefig('nih_mutual_curve_corr.pdf', bbox_inches='tight')

    # For MIMIC, get vector of median (across 30 runs) relative changes in AP at each sparsity ratio for L1 pruning ["forgettability curve"]
    mimic_rel_delta_ap_vector_dict = {}
    for c in MIMIC_CLASSES:
        rel_delta_ap_vector = mimic_delta_df.loc[(mimic_delta_df['label'] == c) & (mimic_delta_df['prune'] == 'L1'), 'ap'].values
        mimic_rel_delta_ap_vector_dict[c] = rel_delta_ap_vector

    # Get MIMIC train and test set co-occurrence matrices
    mimic_train_coocurrence = np.dot(mimic_y_train[MIMIC_CLASSES].transpose(), mimic_y_train[MIMIC_CLASSES])
    mimic_train_coocurrence = pd.DataFrame(mimic_train_coocurrence, index=MIMIC_CLASSES, columns=MIMIC_CLASSES)
    mimic_test_coocurrence = np.dot(mimic_y_test[MIMIC_CLASSES].transpose(), mimic_y_test[MIMIC_CLASSES])
    mimic_test_coocurrence = pd.DataFrame(mimic_test_coocurrence, index=MIMIC_CLASSES, columns=MIMIC_CLASSES)

    # For each unique pair of MIMIC classes, compute the Intersection over Union (IoU),  Euclidean distance between their forgettability curves ["FCD"], and frequency of each class
    disease_As = []
    disease_Bs = []
    FCDs = []
    train_freq_As = []
    train_freq_Bs = []
    train_ious = []
    test_freq_As = []
    test_freq_Bs = []
    test_ious = []
    for comb in combinations(MIMIC_CLASSES, 2):

        rel_delta_ap_vector_A = mimic_rel_delta_ap_vector_dict[comb[0]]
        rel_delta_ap_vector_B = mimic_rel_delta_ap_vector_dict[comb[1]]

        fcd = euclidean(rel_delta_ap_vector_A, rel_delta_ap_vector_B)

        train_iou = mimic_train_coocurrence.loc[comb[0], comb[1]] / (mimic_train_coocurrence[comb[0]].sum() + mimic_train_coocurrence[comb[1]].sum() - mimic_train_coocurrence.loc[comb[0], comb[1]])
        test_iou = mimic_test_coocurrence.loc[comb[0], comb[1]] / (mimic_test_coocurrence[comb[0]].sum() + mimic_test_coocurrence[comb[1]].sum() - mimic_test_coocurrence.loc[comb[0], comb[1]])

        train_freq_A = mimic_train_coocurrence[comb[0]].sum()
        train_freq_B = mimic_train_coocurrence[comb[1]].sum()
        test_freq_A = mimic_test_coocurrence[comb[0]].sum()
        test_freq_B = mimic_test_coocurrence[comb[1]].sum()

        disease_As.append(comb[0])
        disease_Bs.append(comb[1])
        FCDs.append(fcd)
        train_ious.append(train_iou)
        test_ious.append(test_iou)
        train_freq_As.append(train_freq_A)
        train_freq_Bs.append(train_freq_B)
        test_freq_As.append(test_freq_A)
        test_freq_Bs.append(test_freq_B)
    mimic_df = pd.DataFrame({'disease_A': disease_As, 'disease_B': disease_Bs, 'train_freq_A': train_freq_As, 'train_freq_B': train_freq_Bs, 'test_freq_A': test_freq_As,
                             'test_freq_B': test_freq_Bs, 'train_iou': train_ious, 'test_iou': test_ious, 'FCD': FCDs})

    # Compute difference in (log-transformed) class frequency for each pair of MIMIC classes
    mimic_df['log_train_freq_A'] = np.log(mimic_df['train_freq_A'])
    mimic_df['log_train_freq_B'] = np.log(mimic_df['train_freq_B'])
    mimic_df['log_train_freq_diff'] = (mimic_df['log_train_freq_A'] - mimic_df['log_train_freq_B']).abs()

    mimic_df['log_test_freq_A'] = np.log(mimic_df['test_freq_A'])
    mimic_df['log_test_freq_B'] = np.log(mimic_df['test_freq_B'])
    mimic_df['log_test_freq_diff'] = (mimic_df['log_test_freq_A'] - mimic_df['log_test_freq_B']).abs()

    # Compute scaled IoU (for aid of visualization) for each pair of MIMIC classes
    mimic_df['scaled_train_iou'] = mimic_df['train_iou'] ** (1/4)
    mimic_df['scaled_test_iou'] = mimic_df['test_iou'] ** (1/4)

    # Plot relationship btw FCD and (a) absolute difference in log class frequency and (b) scaled IoU between classes for each unique pair of NIH classes
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharey=True)
    sns.regplot(ax=ax[0], data=mimic_df, x='log_test_freq_diff', y='FCD', scatter=True, ci=None, truncate=False, scatter_kws={'s': 50, 'color': 'tab:blue'}, line_kws={'linestyle': '--', 'color': 'orange'})
    ax[0].set_xlabel('|Difference in Log Class Frequency|', fontsize=13)
    ax[0].set_ylabel('Forgettability Curve\nDissimilarity (FCD)', fontsize=13)
    ax[0].text(0, 1.5, r'$r=0.59, \rho=0.60$' + '\n' + r'$(P \ll 0.001)$', fontsize=12)

    sns.regplot(ax=ax[1], data=mimic_df, x='scaled_test_iou', y='FCD', scatter=True, ci=None, truncate=False, scatter_kws={'s': 50, 'color': 'tab:blue'}, line_kws={'linestyle': '--', 'color': 'orange'})
    ax[1].set_xlabel('(IoU Between Classes)$^{1/4}$', fontsize=13)
    ax[1].set_ylabel('')
    ax[1].text(0.27, 1.5, r'$r=$-$0.28, \rho=$-$0.30$' + '\n' + r'             (P<0.001)', fontsize=12)
    fig.tight_layout()

    fig.savefig('mimic_mutual_curve_corr.pdf', bbox_inches='tight')

    ### (END) RELATIONSHIP BETWEEN CLASS CO-OCCURRENCE AND IMPACT OF PRUNING ON CLASS PERFORMANCE ###

    ### DISTRIBUTION OF WEIGHT MAGNITUDES IN NIH- AND MIMIC-TRAINED MODELS ###

    if args.nih_model_dir != '' and args.mimic_model_dir != '':
        # Load NIH-pretrained model (seed 0)
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 20)

        weights = torch.load(os.path.join(args.nih_model_dir, 'chkpt.pt'))['weights']
        msg = model.load_state_dict(weights, strict=True)
        print(msg)

        # Collect weight magnitudes throughout NIH-trained network
        abs_weights_nih = []
        for i, (name, W) in enumerate(model.named_parameters()):
            if 'conv' in name or 'fc' in name:
                abs_weights_nih.append(W.data.detach().abs().numpy().ravel())
        abs_weights_nih = np.concatenate(abs_weights_nih)

        # Load MIMIC-pretrained model (seed 0)
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 19)

        weights = torch.load(os.path.join(args.mimic_model_dir, 'chkpt.pt'))['weights']
        msg = model.load_state_dict(weights, strict=True)
        print(msg)

        # Collect weight magnitudes throughout MIMIC-trained network
        abs_weights_mimic = []
        for i, (name, W) in enumerate(model.named_parameters()):
            if 'conv' in name or 'fc' in name:
                abs_weights_mimic.append(W.data.detach().abs().numpy().ravel())
        abs_weights_mimic = np.concatenate(abs_weights_mimic)

        # Plot histogram of weight magnitudes for NIH- and MIMIC-trained models (in log-scale for visibility)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), sharey=True)

        sns.histplot(abs_weights_nih, ax=ax[0], bins=50)
        ax[0].set_yscale('log')
        ax[0].set_xlabel('Weight Magnitude', fontsize=12)
        ax[0].set_ylabel('Count', fontsize=12)
        ax[0].set_title('NIH-CXR-LT', fontsize=14)

        sns.histplot(abs_weights_mimic, ax=ax[1], bins=50)
        ax[1].set_yscale('log')
        ax[1].set_xlabel('Weight Magnitude', fontsize=12)
        ax[1].set_ylabel('Count', fontsize=12)
        ax[1].set_title('MIMIC-CXR-LT', fontsize=14)

        fig.tight_layout()
        fig.savefig('weight_magnitudes.pdf', bbox_inches='tight')

    ### (END) DISTRIBUTION OF WEIGHT MAGNITUDES IN NIH- AND MIMIC-TRAINED MODELS ###

if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--label_dir', type=str, default='../labels')

    parser.add_argument('--nih_model_dir', type=str, default='../trained_models/nih-cxr-lt_resnet50_ce_lr-0.0001_bs-256')
    parser.add_argument('--nih_l1_prune_dir', type=str, default='../nih-cxr-lt_L1-prune_preds')
    parser.add_argument('--nih_rand_prune_dir', type=str, default='')  # ex: 'nih-cxr-lt_rand-prune_preds'

    parser.add_argument('--mimic_model_dir', type=str, default='../trained_models/mimic-cxr-lt_resnet50_ce_lr-0.0001_bs-256')
    parser.add_argument('--mimic_l1_prune_dir', type=str, default='../mimic-cxr-lt_L1-prune_preds')
    parser.add_argument('--mimic_rand_prune_dir', type=str, default='')  # ex: 'mimic-cxr-lt_rand-prune_preds'

    parser.add_argument('--n_seeds', type=int, default=30)

    args = parser.parse_args()
    print(args)

    main(args)