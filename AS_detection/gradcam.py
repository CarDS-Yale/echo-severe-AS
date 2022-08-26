import os

import argparse
import numpy as np
import torch
import torchvision
import tqdm

from medcam import medcam

from dataset import EchoDataset

def main(args):
    if args.model_name == 'ssl':
        model_dir = '/home/gih5/echo_avs/simclr+tshuffle_ft_results/3dresnet18_pretr_ssl-simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_pad-hflip-rot+temporal-correction_aug-heavy_clip-len-16-stride-1_num-clips-4_cw_lr-0.1_30ep_patience-5_bs-88_ls-0.1_drp-0.25'
    elif args.model_name == 'kinetics':
        model_dir = '/home/gih5/echo_avs/kinetics_ft_results/3dresnet18_pretr_aug-heavy_clip-len-16-stride-1_num-clips-4_cw_lr-0.0001_30ep_patience-5_bs-88_ls-0.1_drp-0.25'
    elif args.model_name == 'random':
        model_dir = '/home/gih5/echo_avs/rand_ft_results/3dresnet18_rand_aug-heavy_clip-len-16-stride-1_num-clips-4_cw_lr-0.0001_30ep_patience-5_bs-88_ls-0.1_drp-0.25'
    else:
        sys.exit('args.model_name must be one of ["ssl", "kinetics", "random"]')

    # 5 true positives, 1 true negative, 1 false positive 
    acc_nums = ['E105457874', 'E108086538', 'E103830926', 'E107482161', 'E109469211', 'E104903081', 'E110414652']

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
    test_dataset = EchoDataset(data_dir=args.data_dir, split='test', clip_len=None, sampling_rate=1, num_clips=1, n_TTA=0, kinetics=(args.model_name == 'kinetics'))

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