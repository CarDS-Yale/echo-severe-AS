import os

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
from math import ceil
from utils import get_date, bootstrap_threshold

np.random.seed(0)

# With early stopping by val AUC
model_dirs = [
    '/home/gih5/echo-severe-AS/ssl_ft_results/3dresnet18_pretr_ssl-100222_mi-simclr+fo_ssl_aug_clip-len-16-stride-1_num-clips-4_cw_lr-0.1_30ep_patience-5_bs-88_ls-0.1_drp-0.25_val-auc',
    '/home/gih5/echo-severe-AS/kinetics_ft_results/3dresnet18_pretr_aug_clip-len-16-stride-1_num-clips-4_cw_lr-0.0001_30ep_patience-5_bs-88_ls-0.1_drp-0.25_val-auc',
    '/home/gih5/echo-severe-AS/random_ft_results/3dresnet18_rand_aug_clip-len-16-stride-1_num-clips-4_cw_lr-0.0001_30ep_patience-5_bs-88_ls-0.1_drp-0.25_val-auc'
]
data_dir = '/home/gih5/mounts/nfs_echo_yale/031522_echo_avs_preprocessed'

metrics = ['auroc', 'aupr']
threshold = 0.6075266666666667
alpha = 0.95
n_samples = 10000

for cohort in ['051823_full_test_2016-2020', '100122_test_2021', 'cedars']:
    if cohort == 'cedars':
        pred_df = pd.read_csv('/home/gih5/echo-severe-AS/YaleASInference_2-28-23_clean.csv')
        pred_df = pred_df[['study_uid', 'ensemble', 'SevereAS']]
        
        # Average video-level predictions into study-level predictions
        pred_df = pred_df.groupby('study_uid', as_index=False).agg({'ensemble': np.mean, 'SevereAS': 'first'}).reset_index(drop=True)

        acc_num_df = pd.read_csv(f'052123_{cohort}_prevalence-0.015_cohort.csv')
        new_acc_nums = acc_num_df['acc_num'].values

        pred_df = pred_df[pred_df['study_uid'].isin(new_acc_nums)].reset_index(drop=True)

        pred_df = pred_df.rename(columns={'study_uid': 'acc_num', 'ensemble': 'y_hat', 'SevereAS': 'y_true'})
    else:
        y_hats = []
        for model_dir in model_dirs:
            pred_df = pd.read_csv(os.path.join(model_dir, f'{cohort}_video_preds.csv'))

            pred_df = pred_df.groupby('acc_num', as_index=False).agg({'y_hat': np.mean, 'y_true': 'first'}).reset_index(drop=True)

            y_hats.append(pred_df['y_hat'])
        pred_df['y_hat'] = np.array(y_hats).mean(axis=0)  # ensemble across models

        if cohort == '051823_full_test_2016-2020':
            acc_num_df = pd.read_csv(f'052123_{cohort}_prevalence-0.015_cohort.csv')
            new_acc_nums = acc_num_df['acc_num'].values

            pred_df = pred_df[pred_df['acc_num'].isin(new_acc_nums)].reset_index(drop=True)

    fprs, tprs, _ = roc_curve(pred_df['y_true'], pred_df['y_hat'])

    precs, recalls, thrs = precision_recall_curve(pred_df['y_true'], pred_df['y_hat'])

    auroc = auc(fprs, tprs)
    aupr = auc(recalls, precs)

    lb_dict, ub_dict = bootstrap_threshold(pred_df, metrics, threshold, alpha=alpha, n_samples=n_samples)

    # ROC plot
    roc_plot, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.plot(fprs, tprs, lw=2, label=f'AUC: {auroc:.3f} ({lb_dict["auroc"]:.3f}, {ub_dict["auroc"]:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('1 - Specificity', fontsize=13)
    ax.set_ylabel('Sensitivity', fontsize=13)
    ax.legend(loc="lower right", fontsize=11)

    roc_plot.tight_layout()
    roc_plot.savefig(f'{get_date()}_roc_{cohort}.pdf', bbox_inches='tight')

    # PR plot
    pr_plot, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.plot(recalls, precs, lw=2, label=f'AUC: {aupr:.3f} ({lb_dict["aupr"]:.3f}, {ub_dict["aupr"]:.3f})')
    ax.axhline(y=pred_df['y_true'].sum()/pred_df.shape[0], color='navy', lw=2, linestyle='--')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.legend(loc="upper right", fontsize=11)

    pr_plot.tight_layout()
    pr_plot.savefig(f'{get_date()}_pr_{cohort}.pdf', bbox_inches='tight')
