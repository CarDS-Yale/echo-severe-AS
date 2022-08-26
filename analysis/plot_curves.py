import os

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score

from utils import get_date

for split in ['test', 'ext_test']:
    MODELS = ['SSL', 'Kinetics-400', 'Random', 'Ensemble']

    model_dirs = [
        '/home/gih5/echo_avs/simclr+tshuffle_ft_results/3dresnet18_pretr_ssl-simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_pad-hflip-rot+temporal-correction_aug-heavy_clip-len-16-stride-1_num-clips-4_cw_lr-0.1_30ep_patience-5_bs-88_ls-0.1_drp-0.25',
        '/home/gih5/echo_avs/kinetics_ft_results/3dresnet18_pretr_aug-heavy_clip-len-16-stride-1_num-clips-4_cw_lr-0.0001_30ep_patience-5_bs-88_ls-0.1_drp-0.25',
        '/home/gih5/echo_avs/rand_ft_results/3dresnet18_rand_aug-heavy_clip-len-16-stride-1_num-clips-4_cw_lr-0.0001_30ep_patience-5_bs-88_ls-0.1_drp-0.25',
    ]

    roc_dict = {}
    for model in MODELS:
        roc_dict[model] = {}

    pr_dict = {}
    for model in MODELS:
        pr_dict[model] = {}

    for model_name, model_dir in zip(MODELS, model_dirs):
        pred_df = pd.read_csv(os.path.join(model_dir, f'{split}_preds.csv'))

        fprs, tprs, _ = roc_curve(pred_df['y_true'], pred_df['y_hat'])

        precs, recalls, thrs = precision_recall_curve(pred_df['y_true'], pred_df['y_hat'])

        roc_dict[model_name]['fprs'] = fprs
        roc_dict[model_name]['tprs'] = tprs
        roc_dict[model_name]['auc'] = auc(fprs, tprs)

        pr_dict[model_name]['precs'] = precs
        pr_dict[model_name]['recalls'] = recalls
        pr_dict[model_name]['auc'] = auc(recalls, precs)

    y_hats = []
    for model_dir in model_dirs:
        pred_df = pd.read_csv(os.path.join(model_dir, f'{split}_preds.csv'))

        y_hats.append(pred_df['y_hat'])

    y_hat = np.array(y_hats).mean(0)
    pred_df['y_hat'] = y_hat

    fprs, tprs, _ = roc_curve(pred_df['y_true'], pred_df['y_hat'])

    precs, recalls, thrs = precision_recall_curve(pred_df['y_true'], pred_df['y_hat'])

    roc_dict['Ensemble']['fprs'] = fprs
    roc_dict['Ensemble']['tprs'] = tprs
    roc_dict['Ensemble']['auc'] = auc(fprs, tprs)

    pr_dict['Ensemble']['precs'] = precs
    pr_dict['Ensemble']['recalls'] = recalls
    pr_dict['Ensemble']['auc'] = auc(recalls, precs)

    # ROC plot
    roc_plot, ax = plt.subplots(1, 1, figsize=(6, 6))

    for model in MODELS:
        ax.plot(roc_dict[model]['fprs'], roc_dict[model]['tprs'], lw=2, label=f'{model} (AUC: {roc_dict[model]["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('1 - Specificity', fontsize=13)
    ax.set_ylabel('Sensitivity', fontsize=13)
    ax.legend(loc="lower right", fontsize=11)

    roc_plot.tight_layout()
    roc_plot.savefig(f'{get_date()}_roc_{split}.pdf', bbox_inches='tight')

    # PR plot
    pr_plot, ax = plt.subplots(1, 1, figsize=(6, 6))

    for model in MODELS:
        ax.plot(pr_dict[model]['recalls'], pr_dict[model]['precs'], lw=2, label=f'{model} (AUC: {pr_dict[model]["auc"]:.3f})')
    ax.axhline(y=pred_df['y_true'].sum()/pred_df.shape[0], color='navy', lw=2, linestyle='--')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.legend(loc="lower right" if split == 'test' else "upper right", fontsize=11)

    pr_plot.tight_layout()
    pr_plot.savefig(f'{get_date()}_pr_{split}.pdf', bbox_inches='tight')
