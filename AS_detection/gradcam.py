import os

import argparse
import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm

from medcam import medcam

from dataset import EchoDataset

def main(args):
    if args.model_name == 'ssl':
        model_dir = '/home/gih5/echo-severe-AS/ssl_ft_results/3dresnet18_pretr_ssl-100222_mi-simclr+fo_ssl_aug_clip-len-16-stride-1_num-clips-4_cw_lr-0.1_30ep_patience-5_bs-88_ls-0.1_drp-0.25_val-auc'
    elif args.model_name == 'kinetics':
        model_dir = '/home/gih5/echo-severe-AS/kinetics_ft_results/3dresnet18_pretr_aug_clip-len-16-stride-1_num-clips-4_cw_lr-0.0001_30ep_patience-5_bs-88_ls-0.1_drp-0.25_val-auc'
    elif args.model_name == 'random':
        model_dir = '/home/gih5/echo-severe-AS/random_ft_results/3dresnet18_rand_aug_clip-len-16-stride-1_num-clips-4_cw_lr-0.0001_30ep_patience-5_bs-88_ls-0.1_drp-0.25_val-auc'
    else:
        sys.exit('args.model_name must be one of ["ssl", "kinetics", "random"]')

    # Get acc nums for 5 true positives, 1 true negative, and 1 false positive
    ssl_pred_df = pd.read_csv(os.path.join('/home/gih5/echo-severe-AS/ssl_ft_results/3dresnet18_pretr_ssl-100222_mi-simclr+fo_ssl_aug_clip-len-16-stride-1_num-clips-4_cw_lr-0.1_30ep_patience-5_bs-88_ls-0.1_drp-0.25_val-auc', '100122_test_2016-2020_video_preds.csv'))

    # Get 5 most confident true positives
    tp_acc_nums = ssl_pred_df.loc[ssl_pred_df['y_true'] == 1].sort_values(by='y_hat', ascending=False)['acc_num'][:5].values.tolist()

    # Get most confident true negative
    tn_acc_num = ssl_pred_df.loc[ssl_pred_df['y_true'] == 0].sort_values(by='y_hat', ascending=True)['acc_num'][:1].values.tolist()

    # Get most confident false positive
    fp_acc_num = ssl_pred_df.loc[ssl_pred_df['y_true'] == 0].sort_values(by='y_hat', ascending=False)['acc_num'][:1].values.tolist()
    acc_nums = tp_acc_nums + tn_acc_num + fp_acc_num

    # Initialize model
    model = torchvision.models.video.r3d_18(pretrained=False)
    model.fc = torch.nn.Sequential(torch.nn.Linear(512, 1), torch.nn.Dropout(0.25))

    checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    idx = np.argmax([int(f.split('.')[0].split('-')[1]) for f in checkpoints])
    weights = torch.load(os.path.join(model_dir, checkpoints[idx]), map_location='cpu')['weights']

    model.load_state_dict(weights, strict=True)

    # Prepare for GradCAM
    model = medcam.inject(model, output_dir=args.out_dir, backend='gcam', save_maps=True, layer='layer4', label=None, cudnn=False)
    model.eval()

    # Prepare data loading
    test_dataset = EchoDataset(data_dir=args.data_dir, split='100122_test_2016-2020', clip_len=None, sampling_rate=1, num_clips=1, kinetics=(args.model_name == 'kinetics'))

    # Reset test dataset to just 10 highest probability studies (and highest probability plax video w/in each study)
    new_label_df = test_dataset.label_df
    new_label_df = new_label_df.loc[new_label_df['acc_num'].isin(acc_nums)]
    new_label_df = new_label_df.loc[new_label_df.groupby('acc_num')['plax_prob'].idxmax()]

    new_label_df = new_label_df.sort_values(by='acc_num', key=lambda x: x.map(({acc_num: order for order, acc_num in enumerate(acc_nums)})))

    test_dataset.label_df = new_label_df
    print(test_dataset.label_df)

    # Prepare data loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch['x'], batch['y']

            x = x.squeeze(2)  # remove "num_clips" dimension

            model.forward(x)  # forward pass automatically saves GradCAM output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/gih5/mounts/nfs_echo_yale/031522_echo_avs_preprocessed')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True, choices=['ssl', 'kinetics', 'random'])

    args = parser.parse_args()

    print(args)

    main(args)