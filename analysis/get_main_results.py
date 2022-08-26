import os

import argparse
import numpy as np
import pandas as pd

def main(args):
    METRICS = ['auroc', 'aupr', 'f1', 'precision', 'recall', 'spec@90sens', 'partial_aupr', 'prec@90sens']
    MODELS = ['SSL', 'Kinetics-400', 'Random', 'Ensemble']

    results_df = pd.DataFrame(index=METRICS, columns=MODELS)

    model_dirs = [
        '/home/gih5/echo_avs/simclr+tshuffle_ft_results/3dresnet18_pretr_ssl-simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_pad-hflip-rot+temporal-correction_aug-heavy_clip-len-16-stride-1_num-clips-4_cw_lr-0.1_30ep_patience-5_bs-88_ls-0.1_drp-0.25',
        '/home/gih5/echo_avs/kinetics_ft_results/3dresnet18_pretr_aug-heavy_clip-len-16-stride-1_num-clips-4_cw_lr-0.0001_30ep_patience-5_bs-88_ls-0.1_drp-0.25',
        '/home/gih5/echo_avs/rand_ft_results/3dresnet18_rand_aug-heavy_clip-len-16-stride-1_num-clips-4_cw_lr-0.0001_30ep_patience-5_bs-88_ls-0.1_drp-0.25',
    ]

    for model_name, model_dir in zip(MODELS, model_dirs):
        print('-'*5, model_dir, '-'*5)

        if args.video_level:
            pred_df = pd.read_csv(os.path.join(model_dir, f'{args.split}_video_preds.csv'))
        else:
            pred_df = pd.read_csv(os.path.join(model_dir, f'{args.split}_preds.csv'))

        original_metric_dict = get_metrics(pred_df['y_true'], pred_df['y_hat'])

        lb_dict, ub_dict = bootstrap(pred_df, METRICS, alpha=args.alpha, n_samples=args.n_samples)

        for metric in METRICS:
            result_str = f'{original_metric_dict[metric]:.3f} ({lb_dict[metric]:.3f}, {ub_dict[metric]:.3f})'
            print(f'{metric}: {result_str}')

            results_df.loc[metric, model_name] = result_str

    y_hats = []
    for model_dir in model_dirs:
        if args.video_level:
            pred_df = pd.read_csv(os.path.join(model_dir, f'{args.split}_video_preds.csv'))
        else:
            pred_df = pd.read_csv(os.path.join(model_dir, f'{args.split}_preds.csv'))

        y_hats.append(pred_df['y_hat'])

    print('-'*5, 'ENSEMBLE', '-'*5)
    y_hat = np.array(y_hats).mean(0)
    pred_df['y_hat'] = y_hat

    original_metric_dict = get_metrics(pred_df['y_true'], pred_df['y_hat'])

    lb_dict, ub_dict = bootstrap(pred_df, METRICS, alpha=args.alpha, n_samples=args.n_samples)

    for metric in METRICS:
        result_str = f'{original_metric_dict[metric]:.3f} ({lb_dict[metric]:.3f}, {ub_dict[metric]:.3f})'
        print(f'{metric}: {result_str}')

        results_df.loc[metric, 'Ensemble'] = result_str

    print(results_df)

    if args.video_level:
        results_df.to_csv(f'{get_date()}_main_video_results_{args.split}.tsv', index=True, sep='\t')
    else:
        results_df.to_csv(f'{get_date()}_main_results_{args.split}.tsv', index=True, sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--split', type=str, default='test', choices=['test', 'ext_test'])
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--video_level', action='store_true', default=False)

    args = parser.parse_args()

    print(args)

    main(args)
