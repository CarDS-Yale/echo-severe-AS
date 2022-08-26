import os

import argparse
import numpy as np
import pandas as pd

def main(args):
    METRICS = ['auroc', 'aupr', 'f1', 'precision', 'recall', 'spec@90sens', 'partial_aupr', 'prec@90sens']
    MODELS = ['SSL', 'Kinetics-400', 'Random', 'Ensemble']

    index = ['1', '2-3', '4-5', '6+']
    results_df = pd.DataFrame(index=index, columns=['n'] + MODELS)

    model_dirs = [
        '/home/gih5/echo_avs/simclr+tshuffle_ft_results/3dresnet18_pretr_ssl-simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_pad-hflip-rot+temporal-correction_aug-heavy_clip-len-16-stride-1_num-clips-4_cw_lr-0.1_30ep_patience-5_bs-88_ls-0.1_drp-0.25',
        '/home/gih5/echo_avs/kinetics_ft_results/3dresnet18_pretr_aug-heavy_clip-len-16-stride-1_num-clips-4_cw_lr-0.0001_30ep_patience-5_bs-88_ls-0.1_drp-0.25',
        '/home/gih5/echo_avs/rand_ft_results/3dresnet18_rand_aug-heavy_clip-len-16-stride-1_num-clips-4_cw_lr-0.0001_30ep_patience-5_bs-88_ls-0.1_drp-0.25',
    ]

    for model_name, model_dir in zip(MODELS, model_dirs):
        print('-'*5, model_dir, '-'*5)

        pred_df = pd.read_csv(os.path.join(model_dir, f'{args.split}_video_preds.csv'))

        freq_df = pred_df['acc_num'].value_counts(ascending=True)

        for i, count in enumerate([1, [2, 3], [4, 5], 6]):
            if isinstance(count, list):
                acc_nums = freq_df[freq_df.isin(count)].index
            elif count == 6:
                acc_nums = freq_df[freq_df >= count].index
            else:
                acc_nums = freq_df[freq_df == count].index
            sub_df = pred_df[pred_df['acc_num'].isin(acc_nums)]
            sub_df = sub_df.groupby('acc_num').agg({'y_true': np.mean, 'y_hat': np.mean})

            original_metric_dict = get_metrics(sub_df['y_true'], sub_df['y_hat'])

            lb_dict, ub_dict = bootstrap(sub_df, METRICS, alpha=args.alpha, n_samples=args.n_samples)

            if isinstance(count, list):
                print(f'----- {count[0]}-{count[1]} VIDEOS (n={acc_nums.size}) -----')
            elif count == 6:
                print(f'----- {count}+ VIDEOS (n={acc_nums.size}) -----')
            else:
                print(f'----- {count} VIDEOS (n={acc_nums.size}) -----')
            
            for metric in METRICS:
                result_str = f'{original_metric_dict[metric]:.3f} ({lb_dict[metric]:.3f}, {ub_dict[metric]:.3f})'
                print(f'{metric}: {result_str}')

            print(f'{original_metric_dict["auroc"]:.3f} ({lb_dict["auroc"]:.3f}, {ub_dict["auroc"]:.3f})')
            results_df.loc[index[i], model_name] = f'{original_metric_dict["auroc"]:.3f} ({lb_dict["auroc"]:.3f}, {ub_dict["auroc"]:.3f})'
            results_df.loc[index[i], 'n'] = acc_nums.size

    y_hats = []
    for model_dir in model_dirs:
        pred_df = pd.read_csv(os.path.join(model_dir, f'{args.split}_video_preds.csv'))

        y_hats.append(pred_df['y_hat'])

    print('-'*5, 'ENSEMBLE', '-'*5)
    y_hat = np.array(y_hats).mean(0)
    pred_df['y_hat'] = y_hat

    freq_df = pred_df['acc_num'].value_counts(ascending=True)
    for i, count in enumerate([1, [2, 3], [4, 5], 6]):
        if isinstance(count, list):
            acc_nums = freq_df[freq_df.isin(count)].index
        elif count == 6:
            acc_nums = freq_df[freq_df >= count].index
        else:
            acc_nums = freq_df[freq_df == count].index
        sub_df = pred_df[pred_df['acc_num'].isin(acc_nums)]
        sub_df = sub_df.groupby('acc_num').agg({'y_true': np.mean, 'y_hat': np.mean})

        original_metric_dict = get_metrics(sub_df['y_true'], sub_df['y_hat'])

        lb_dict, ub_dict = bootstrap(sub_df, METRICS, alpha=args.alpha, n_samples=args.n_samples)

        if isinstance(count, list):
            print(f'----- {count[0]}-{count[1]} VIDEOS (n={acc_nums.size}) -----')
        elif count == 6:
            print(f'----- {count}+ VIDEOS (n={acc_nums.size}) -----')
        else:
            print(f'----- {count} VIDEOS (n={acc_nums.size}) -----')
        
        for metric in METRICS:
            result_str = f'{original_metric_dict[metric]:.3f} ({lb_dict[metric]:.3f}, {ub_dict[metric]:.3f})'
            print(f'{metric}: {result_str}')

        print(f'{original_metric_dict["auroc"]:.3f} ({lb_dict["auroc"]:.3f}, {ub_dict["auroc"]:.3f})')
        results_df.loc[index[i], 'Ensemble'] = f'{original_metric_dict["auroc"]:.3f} ({lb_dict["auroc"]:.3f}, {ub_dict["auroc"]:.3f})'
        results_df.loc[index[i], 'n'] = acc_nums.size

        print(results_df)

    results_df.to_csv(f'{get_date()}_results_by_num_videos_{args.split}.tsv', index=True, sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--split', type=str, default='test', choices=['test', 'ext_test'])
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--n_samples', type=int, default=10000)

    args = parser.parse_args()

    print(args)

    main(args)
