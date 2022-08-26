import os
import shutil

import argparse
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from scipy.ndimage import zoom

def main(args):
    # 5 most confident true positives, most confident (lowest probability) true negative, + most confident false positive
    files = [
        'E105457874_0.avi',
        'E108086538_2.avi',
        'E103830926_2.avi',
        'E107482161_0.avi',
        'E109469211_0.avi',
        'E104903081_0.avi',
        'E110414652_0.avi'
    ]

    out_dir = f'{get_date()}_{args.model_name}_gradcam_viz'

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    for i in range(len(files)):
        if len(args.gradcam_dir) == 1:
            nib.load(os.path.join(args.gradcam_dir[0], f'attention_map_{i}_0_0.nii.gz')).get_fdata()
        else:
            # Load and stack each model's attention map along zeroth axis
            att_map = [nib.load(os.path.join(gcam_dir, f'attention_map_{i}_0_0.nii.gz')).get_fdata() for gcam_dir in args.gradcam_dir]

            # Fuse attention maps from each model for a given echo video
            att_map = np.array(att_map).mean(0)

        # Read in video
        x = load_video(os.path.join(args.data_dir, 'videos', files[i]))
        x = np.transpose(x, (1, 2, 3, 0))

        # Interpolate heatmap to original video dimensions (112 x 112 x num_frames)
        att_map = zoom(att_map, (x.shape[0]/att_map.shape[0], x.shape[1]/att_map.shape[1], x.shape[3]/att_map.shape[2]))

        if i == 5:
            classification = 'tn'
        elif i == 6:
            classification = 'fp'
        else:
            classification = 'tp'

        # Save raw saliency map video
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        fps = 30
        out = cv2.VideoWriter(os.path.join(out_dir, f'{get_date()}_video_{classification}_{args.model_name}_gradcam_{i}.avi'), fourcc, fps, (112, 112), True)

        # Convert att_map to uint8
        att_map = cv2.normalize(att_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Overlay frame-by-frame spatial heatmap
        for j in range(x.shape[3]):
            frame = cv2.addWeighted(x[:, :, :, j], 1.0, cv2.applyColorMap(att_map[:, :, j], cv2.COLORMAP_INFERNO), 0.5, 0)

            out.write(frame)

        out.release()

        # Max project to 2D and save as saliency map image
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(x[:, :, 0, 0], cmap='gray')
        ax.imshow(att_map.max(axis=-1), cmap='inferno', alpha=0.5)
        ax.set_xticks([]), ax.set_yticks([])
        fig.tight_layout()

        fig.savefig(os.path.join(out_dir, f'{get_date()}_{classification}_{args.model_name}_gradcam_{i}.png'), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/gih5/mounts/nfs_echo_yale/031522_echo_avs_preprocessed')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--gradcam_dir', type=str, nargs='+', required=True, help='path to directory with gradcam output (can be list of paths separated with spaces')

    args = parser.parse_args()

    print(args)

    main(args)
