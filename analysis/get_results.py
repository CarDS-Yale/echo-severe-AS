import os

import argparse
import numpy as np
import pandas as pd
import tqdm
from math import ceil
from utils import bootstrap_threshold, get_date, get_metrics_threshold

def main(args):
    np.random.seed(0)

    # With early stopping by val AUC
    model_dirs = [
        '/home/gih5/echo-severe-AS/ssl_ft_results/3dresnet18_pretr_ssl-100222_mi-simclr+fo_ssl_aug_clip-len-16-stride-1_num-clips-4_cw_lr-0.1_30ep_patience-5_bs-88_ls-0.1_drp-0.25_val-auc',
        '/home/gih5/echo-severe-AS/kinetics_ft_results/3dresnet18_pretr_aug_clip-len-16-stride-1_num-clips-4_cw_lr-0.0001_30ep_patience-5_bs-88_ls-0.1_drp-0.25_val-auc',
        '/home/gih5/echo-severe-AS/random_ft_results/3dresnet18_rand_aug_clip-len-16-stride-1_num-clips-4_cw_lr-0.0001_30ep_patience-5_bs-88_ls-0.1_drp-0.25_val-auc'
    ]
    data_dir = '/home/gih5/mounts/nfs_echo_yale/031522_echo_avs_preprocessed'

    metrics = ['auroc', 'aupr', 'f1', 'precision', 'npv', 'recall', 'specificity']
    threshold = 0.6075266666666667
    
    if args.cohort == '100122_test_2021':
        labels = pd.read_csv(os.path.join(data_dir, args.cohort + '.csv'))
        study_labels = labels.groupby('acc_num', as_index=False).first()

        acc_num_dict = {'all': study_labels['acc_num'].values}

        prevalences = ['all']
    elif args.cohort == '051823_full_test_2016-2020':
        labels = pd.read_csv(os.path.join(data_dir, args.cohort + '.csv'))
        study_labels = labels.groupby('acc_num', as_index=False).first()

        acc_num_dict = {'all': study_labels['acc_num'].values}

        prevalences = ['0.015', 'all']

        for prevalence_str in prevalences[:-1]:
            acc_num_df = pd.read_csv(f'052123_{args.cohort}_prevalence-{prevalence_str}_cohort.csv')
            acc_num_dict[prevalence_str] = acc_num_df['acc_num'].values
    else:  # cohort = args.cedars
        labels = pd.read_csv('/home/gih5/echo-severe-AS/YaleASInference_2-28-23_PseudoID_clean.csv')
        study_labels = labels.groupby(by='study_uid', as_index=False).first()

        acc_num_dict = {'all': study_labels['study_uid'].values}

        prevalences = ['0.015', '0.05', '0.1', '0.15', '0.2', 'all']

        for prevalence_str in prevalences[:-1]:
            acc_num_df = pd.read_csv(f'052123_{args.cohort}_prevalence-{prevalence_str}_cohort.csv')
            acc_num_dict[prevalence_str] = acc_num_df['acc_num'].values

    results_df = pd.DataFrame(index=['severe_AS'] + metrics, columns=prevalences)

    if args.cohort == 'cedars':
        orig_pred_df = pd.read_csv('/home/gih5/echo-severe-AS/YaleASInference_2-28-23_clean.csv')
        orig_pred_df = orig_pred_df[['study_uid', 'ensemble', 'SevereAS']]
        
        if args.eval_type == 'studies':
            # Average video-level predictions into study-level predictions
            orig_pred_df = orig_pred_df.groupby('study_uid', as_index=False).agg({'ensemble': np.mean, 'SevereAS': 'first'}).reset_index(drop=True)
        elif args.eval_type == 'one_video':
            # Randomly sample one video per study
            orig_pred_df = orig_pred_df.groupby('study_uid', as_index=False).apply(lambda x: x.sample(1, random_state=0)).reset_index(drop=True)

        orig_pred_df = orig_pred_df.rename(columns={'study_uid': 'acc_num', 'ensemble': 'y_hat', 'SevereAS': 'y_true'})
    else:
        y_hats = []
        for model_dir in model_dirs:
            orig_pred_df = pd.read_csv(os.path.join(model_dir, f'{args.cohort}_video_preds.csv'))

            if args.eval_type == 'studies':
                orig_pred_df = orig_pred_df.groupby('acc_num', as_index=False).agg({'y_hat': np.mean, 'y_true': 'first'}).reset_index(drop=True)
            elif args.eval_type == 'one_video':
                # Randomly sample one video per study
                orig_pred_df = orig_pred_df.groupby('acc_num', as_index=False).apply(lambda x: x.sample(1, random_state=0)).reset_index(drop=True)

            y_hats.append(orig_pred_df['y_hat'])
        orig_pred_df['y_hat'] = np.array(y_hats).mean(axis=0)  # ensemble across models

    for prevalence_str in tqdm.tqdm(prevalences):
        pred_df = orig_pred_df.copy()
        if prevalence_str != 'all':
            pred_df = pred_df[pred_df['acc_num'].isin(acc_num_dict[prevalence_str])].reset_index(drop=True)


        original_metric_dict = get_metrics_threshold(pred_df['y_true'], pred_df['y_hat'], threshold)

        lb_dict, ub_dict = bootstrap_threshold(pred_df, metrics, threshold, alpha=args.alpha, n_samples=args.n_samples)

        n_pos = (pred_df['y_true'] == 1).sum()
        results_df.loc['severe_AS', prevalence_str] = f'{n_pos}/{pred_df.shape[0]} ({(n_pos/pred_df.shape[0])*100:.1f}%)'
        for metric in metrics:
            result_str = f'{original_metric_dict[metric]:.3f} ({lb_dict[metric]:.3f}, {ub_dict[metric]:.3f})'

            results_df.loc[metric, prevalence_str] = result_str

            print(result_str)

    print(results_df)

    results_df.to_csv(f'{get_date()}_main_results_{args.cohort}_{args.eval_type}.tsv', index=True, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cohort', type=str, default='051823_full_test_2016-2020', choices=['051823_full_test_2016-2020', '100122_test_2021', 'cedars'])
    parser.add_argument('--eval_type', type=str, default='studies', choices=['studies', 'videos', 'one_video'])
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--n_samples', type=int, default=10000)

    args = parser.parse_args()

    print(args)

    main(args)
