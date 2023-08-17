import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, roc_curve, f1_score
import seaborn as sns
import numpy as np
import tqdm
import shutil
from scipy.stats import ttest_ind

def main(args):
    if args.dataset_name == 'nih':
        CLASSES = [
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
    else:
        CLASSES = [
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

    sparsity_ratios = np.array(range(0, 100, 5)) / 100

    # For each sparsity ratio, ensemble (average predictions) across all 30 runs
    avg_preds_dict = {}
    for ratio in sparsity_ratios:
        y_hats = []
        for seed in range(args.n_seeds):
            if ratio == 0:
                y_hat = pd.read_pickle(os.path.join(prune_dir, f'{dataset_name}-cxr-lt_resnet50_seed-{seed}_prune-{int(ratio)}.pkl'))
            else:
                y_hat = pd.read_pickle(os.path.join(prune_dir, f'{dataset_name}-cxr-lt_resnet50_seed-{seed}_prune-{ratio}.pkl'))

            y_hats.append(y_hat[CLASSES].values)
            
        y_hats = pd.DataFrame(np.stack(y_hats, axis=0).mean(axis=0), columns=CLASSES)

        avg_preds_dict[str(int(ratio * 100))] = y_hats

    # Compute sample-wise correlation between uncompressed model predictions and k-sparse model predictions for each k=0.05,0.10,...,0.90,0.95
    corr_dict = {}
    for ratio in tqdm.tqdm(sparsity_ratios[1:]):
        corrs = avg_preds_dict['0'].corrwith(avg_preds_dict[str(int(ratio*100))], method='spearman', axis=1)
        corr_dict[str(int(ratio*100))] = corrs

    # Organize information into DataFrame with image ID, sparsity ratio, and correlation btw uncompressed and k-sparse predictions
    corr_df = []
    for ratio in sparsity_ratios[1:]:
        corrs = corr_dict[str(int(ratio*100))]
        df = pd.DataFrame({'id': list(range(corrs.size)), 'ratio': ratio, 'corr': corrs})
        corr_df.append(df)
    corr_df = pd.concat(corr_df, axis=0)

    # Define pruning-identified exemplars (PIEs) as test images falling in the bottom 5th percentile of correlation btw uncompressed predictions and 90%-sparsified model predictions
    corrs = corr_dict['90']

    pie_idx = np.argsort(corrs)[:int(0.05*corrs.size)]
    print(pie_idx.size)

    # Read in test set ground truth
    y_test = pd.read_csv(os.path.join(label_dir, f'{dataset_name}-cxr-lt_labels_test.csv'))

    # Obtain ratio and relative change in class frequency for PIEs compared to non-PIEs
    freq_df = pd.DataFrame({'label': CLASSES, 'pie_freq': y_test.loc[pie_idx, CLASSES].sum(0) / y_test.loc[pie_idx].shape[0],
                            'non_pie_freq': y_test.drop(index=pie_idx)[CLASSES].sum(0) / y_test.drop(index=pie_idx).shape[0]})
    freq_df['rel_delta_freq'] = (freq_df['pie_freq'] - freq_df['non_pie_freq']) / freq_df['non_pie_freq']
    freq_df['freq_ratio'] = freq_df['pie_freq'] / freq_df['non_pie_freq']

    # For each class, plot ratio of class frequency in the subset of PIEs vs. the subset of non-PIEs
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    sns.barplot(ax=ax, data=freq_df, x='label', y='freq_ratio', color='tab:blue')
    ax.axhline(1, linestyle='--', color='black')
    ax.set_xticklabels([s[:7] for s in CLASSES], rotation=45)
    ax.set_xlabel('Class', fontsize=13)
    ax.set_ylabel('PIE:Non-PIE Ratio', fontsize=13)
    plt.show()
    fig.tight_layout()
    fig.savefig(f'{dataset_name}_pie_freq_ratio.pdf', bbox_inches='tight')

    # Get number of distinct classes per test set image in PIEs vs. non-PIEs
    pie_n_diseases, pie_counts = np.unique(y_test.loc[pie_idx, 'n_diseases'], return_counts=True)
    non_pie_n_diseases, non_pie_counts = np.unique(y_test.drop(pie_idx)['n_diseases'], return_counts=True)

    # Bin into 0,1,2,3,4+ disease classes per image
    n_diseases = [str(i) for i in range(4)] + ['4+']
    non_pie_counts_truncated = non_pie_counts[:5]
    non_pie_counts_truncated[-1] += np.sum(non_pie_counts[5:])

    pie_counts_truncated = pie_counts[:5]
    pie_counts_truncated[-1] += np.sum(pie_counts[5:])

    # Compute relative frequency of each disease number bin for PIEs vs. non-PIEs
    pie_freqs = (pie_counts_truncated / y_test.loc[pie_idx].shape[0])
    non_pie_freqs = (non_pie_counts_truncated / y_test.drop(index=pie_idx).shape[0])

    # Obtain ratio and relative change in disease number bin frequency (e.g., ratio of test images containing exactly 2 diseases) for PIEs compared to non-PIEs
    pie_n_dis_df = pd.DataFrame({'n_diseases': n_diseases, 'freq': pie_freqs, 'pie': 'PIE'})
    non_pie_n_dis_df = pd.DataFrame({'n_diseases': n_diseases, 'freq': non_pie_freqs, 'pie': 'Non-PIE'})
    n_dis_df = pd.concat([pie_n_dis_df, non_pie_n_dis_df], axis=0)
    n_dis_df['rel_delta_freq'] = (n_dis_df['pie_freq'] - n_dis_df['non_pie_freq']) / n_dis_df['non_pie_freq']
    n_dis_df['freq_ratio'] = n_dis_df['pie_freq'] / n_dis_df['non_pie_freq']

    # For each class, plot ratio of disease number bin frequency (ratio of images containing k diseases for k=0,1,2,3,4+) in the subset of PIEs vs. the subset of non-PIEs
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    sns.barplot(ax=ax, data=n_dis_df, x='n_diseases', y='freq_ratio', color='tab:blue')
    ax.axhline(1, linestyle='--', color='black')
    ax.set_xlabel('# Diseases Present', fontsize=13)
    ax.set_ylabel('PIE:Non-PIE Ratio', fontsize=13)
    plt.show()
    fig.tight_layout()
    fig.savefig(f'{dataset_name}_pie_n_disease_ratio.pdf', bbox_inches='tight')

if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--label_dir', type=str, default='../labels')

    parser.add_argument('--dataset_name', type=str, default='nih', choices=['nih', 'mimic'])
    parser.add_argument('--prune_dir', type=str, default='../nih-cxr-lt_L1-prune_preds')  # for MIMIC-CXR-LT, can use '../mimic-cxr-lt_L1-prune_preds'
    parser.add_argument('--prune_type', type=str, default='L1', choices=['random', 'L1'])

    parser.add_argument('--n_seeds', type=int, default=30)

    args = parser.parse_args()
    print(args)

    main(args)