import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr
from math import ceil
from analysis.utils import get_date

model_dirs = [
    '/home/gih5/echo-severe-AS/ssl_ft_results/3dresnet18_pretr_ssl-100222_mi-simclr+fo_ssl_aug_clip-len-16-stride-1_num-clips-4_cw_lr-0.1_30ep_patience-5_bs-88_ls-0.1_drp-0.25_val-auc',
    '/home/gih5/echo-severe-AS/kinetics_ft_results/3dresnet18_pretr_aug_clip-len-16-stride-1_num-clips-4_cw_lr-0.0001_30ep_patience-5_bs-88_ls-0.1_drp-0.25_val-auc',
    '/home/gih5/echo-severe-AS/random_ft_results/3dresnet18_rand_aug_clip-len-16-stride-1_num-clips-4_cw_lr-0.0001_30ep_patience-5_bs-88_ls-0.1_drp-0.25_val-auc'
]

for cohort in ['051823_full_test_2016-2020', '100122_test_2021', 'cedars']:
    if cohort == 'cedars':
        pred_df = pd.read_csv('YaleASInference_2-28-23_PseudoID_clean.csv')

        preds = pred_df.groupby(by='study_uid', as_index=False).agg({'ensemble': np.mean, 'SevereAS': 'first', 'AS_severity': 'first'})
        print(preds.shape)

        ## DOWNSAMPLE TO 1.5% PREVALENCE ##
        acc_num_df = pd.read_csv(f'analysis/052123_{cohort}_prevalence-0.015_cohort.csv')
        preds = preds[preds['study_uid'].isin(acc_num_df['acc_num'])].reset_index(drop=True)

        print(preds)
        print(preds['AS_severity'].value_counts())

        preds['AS_severity_simple'] = preds['AS_severity'].apply(lambda x: 'Non-Severe' if x not in ['None', 'Severe'] else x)

        print(preds['AS_severity_simple'].value_counts())

        ## VIOLINT PLOT OF PROBABILITIES BY AS SEVERITY (SIMPLIFIED) ##
        levels = ['None', 'Non-Severe', 'Severe']

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax = sns.violinplot(ax=ax, data=preds, x='AS_severity_simple', y='ensemble', order=levels)
        ax.set_xticklabels(levels, fontsize=12)

        ax.set_xlabel('Aortic Stenosis Severity', fontsize=13)
        ax.set_ylabel('Model Prediction', fontsize=13)
        ax.set_ylim(top=1.25)

        for i, label in enumerate(levels):
            freq = (preds['AS_severity_simple'] == label).sum()
            ax.text(i, 1.1035, f'N={freq}', horizontalalignment='center', fontsize=11)  # , weight='semibold'

        ax.text(-0.22, 1.2, 'P<0.001', fontsize=14, horizontalalignment='center')

        fig.tight_layout()
        fig.savefig(f'{get_date()}_cedars_preds_by_AS_severity_simple.pdf', bbox_inches='tight')

        mapping_dict = {}
        for i, label in enumerate(levels):
            mapping_dict[label] = i
        ordinal_encoding = preds['AS_severity_simple'].map(mapping_dict)

        print(spearmanr(preds['ensemble'], ordinal_encoding))

        ## VIOLINT PLOT OF PROBABILITIES BY AS SEVERITY ##
        levels = ['None', 'Mild', 'Mild-Moderate', 'Moderate', 'Moderate-Severe', 'Severe']

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax = sns.violinplot(ax=ax, data=preds, x='AS_severity', y='ensemble', order=levels)
        ax.set_xticklabels(['None', 'Mild', 'Mild-\nModerate', 'Moderate', 'Moderate-\nSevere', 'Severe'], rotation=45)

        ax.set_xlabel('Aortic Stenosis Severity', fontsize=13)
        ax.set_ylabel('Model Prediction', fontsize=13)
        ax.set_ylim(bottom=-0.195, top=1.25)

        for i, label in enumerate(levels):
            freq = (preds['AS_severity'] == label).sum()
            ax.text(i, 1.1035, f'N={freq}', horizontalalignment='center', fontsize=11)  # , weight='semibold'

        ax.text(0.05, 1.19, 'P<0.001', fontsize=13, horizontalalignment='center')

        fig.tight_layout()
        fig.savefig(f'{get_date()}_cedars_preds_by_AS_severity.pdf', bbox_inches='tight')

        mapping_dict = {}
        for i, label in enumerate(levels):
            mapping_dict[label] = i
        ordinal_encoding = preds['AS_severity'].map(mapping_dict)

        print(spearmanr(preds['ensemble'], ordinal_encoding))
    else:
        data_dir = '~/mounts/nfs_echo_yale/031522_echo_avs_preprocessed/'

        print('---', cohort, '---')

        labels = pd.read_csv(os.path.join(data_dir, cohort + '.csv'))
        acc_num_dict = labels.groupby('acc_num').agg({'av_stenosis': 'first'}).to_dict()

        if cohort == '100122_test_2021':
            preds = pd.read_csv('101822_ensemble_' + cohort + '_preds.csv')
        else:   # if cohort == '051823_full_test_2016-2020':
            y_hats = []
            for model_dir in model_dirs:
                preds = pd.read_csv(os.path.join(model_dir, f'{cohort}_video_preds.csv'))
                preds = preds.groupby('acc_num', as_index=False).agg({'y_hat': np.mean, 'y_true': 'first'}).reset_index(drop=True)

                y_hats.append(preds['y_hat'])
            preds['y_hat'] = np.array(y_hats).mean(axis=0)  # ensemble across models

            acc_num_df = pd.read_csv(f'analysis/052123_{cohort}_prevalence-0.015_cohort.csv')
            preds = preds[preds['acc_num'].isin(acc_num_df['acc_num'])].reset_index(drop=True)

        preds['AS_bin'] = preds['acc_num'].map(acc_num_dict['av_stenosis'])

        full_labels = pd.read_csv('echo_train_test_extract.csv')
        full_acc_num_dict = full_labels.set_index('AccessionNumber').to_dict()

        preds['AS'] = preds['acc_num'].map(full_acc_num_dict['AVStenosis'])

        levels = ['None', 'Non-Severe', 'Severe']
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax = sns.violinplot(ax=ax, data=preds, x='AS_bin', y='y_hat', order=levels)
        ax.set_xlabel('Aortic Stenosis Severity', fontsize=13)
        ax.set_xticklabels(levels, fontsize=12)
        ax.set_ylabel('Model Prediction', fontsize=13)
        ax.set_ylim(top=1.25)

        for i, label in enumerate(levels):
            freq = (preds['AS_bin'] == label).sum()
            if cohort == '100122_test_2021':
                ax.text(i, 1.1, f'N={freq}', horizontalalignment='center', fontsize=11)  # , weight='semibold'
            else:
                ax.text(i, 1.125, f'N={freq}', horizontalalignment='center', fontsize=11)  # , weight='semibold'

        ax.text(0.01, 1.2, 'P<0.001', fontsize=14, horizontalalignment='right')

        fig.tight_layout()
        fig.savefig(f'{get_date()}_{cohort}_preds_by_AS_simple.pdf', bbox_inches='tight')

        ## STATISTICAL TEST ##
        from scipy.stats import spearmanr
        ordinal_encoding = preds['AS_bin'].map({'None': 0, 'Non-Severe': 1, 'Severe': 2})

        print(spearmanr(preds['y_hat'], ordinal_encoding))

        levels = ['None', 'Sclerosis without Stenosis', 'Mild', 'Mild-Mod', 'Moderate', 'Mod-Sev', 'Severe']
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax = sns.violinplot(ax=ax, data=preds, x='AS', y='y_hat', order=levels)
        ax.set_xticklabels(['None', 'Sclerosis w/o\nStenosis', 'Mild', 'Mild-\nModerate', 'Moderate', 'Moderate-\nSevere', 'Severe'], rotation=45)

        ax.set_xlabel('Aortic Stenosis Severity', fontsize=13)
        ax.set_ylabel('Model Prediction', fontsize=13)
        ax.set_ylim(bottom=-0.195, top=1.25)

        for i, label in enumerate(levels):
            freq = (preds['AS'] == label).sum()
            if cohort == '100122_test_2021':
                ax.text(i, 1.1, f'N={freq}', horizontalalignment='center', fontsize=11)  # , weight='semibold'
            else:
                ax.text(i, 1.125, f'N={freq}', horizontalalignment='center', fontsize=11)  # , weight='semibold'

        ax.text(0.1, 1.19, 'P<0.001', fontsize=13, horizontalalignment='center')

        fig.tight_layout()
        fig.savefig(f'{get_date()}_{cohort}_preds_by_AS.pdf', bbox_inches='tight')

        ## STATISTICAL TEST ##
        mapping_dict = {}
        for i, label in enumerate(levels):
            mapping_dict[label] = i
        ordinal_encoding = preds['AS'].map(mapping_dict)

        print(spearmanr(preds['y_hat'], ordinal_encoding))


        print(preds.groupby('AS').agg({'y_hat': ['mean', 'std']}))
