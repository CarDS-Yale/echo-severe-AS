import os
import shutil

import argparse
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from scipy.ndimage import zoom

from utils import get_date, load_video

def main(args):
    files = [
        'E106571532_2.avi',
        'E108465271_0.avi',
        'E106836599_1.avi',
        'E109967119_4.avi',
        'E104619394_1.avi',
        'E104792228_0.avi',
        'E108960647_1.avi'
    ]

    out_dir = f'{get_date()}_gradcam_video_viz'

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    for i in range(len(files)):
        # Read in video
        x = load_video(os.path.join(args.data_dir, 'videos', files[i]))
        x = np.transpose(x, (1, 2, 3, 0))

        att_maps = []
        for gcam_dir in args.gradcam_dirs:
            att_map = nib.load(os.path.join(gcam_dir, 'layer4', f'attention_map_{i}_0_0.nii.gz')).get_fdata()
            att_map = zoom(att_map, (112/att_map.shape[0], 112/att_map.shape[1], x.shape[3]/att_map.shape[2]))
            att_map = cv2.normalize(att_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            att_maps.append(att_map)
        att_map = np.concatenate(att_maps, axis=1)

        if i == 5:
            classification = 'tn'
        elif i == 6:
            classification = 'fp'
        else:
            classification = 'tp'

        # Save raw video + frame-by-frame saliency map overlaid for multiple models at once (side by side by side)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        out = cv2.VideoWriter(os.path.join(out_dir, f'{get_date()}_video_{classification}_gradcam_{i}.mp4'), fourcc, fps, (112*3, 112), True)

        for j in range(x.shape[3]):
            frame = cv2.addWeighted(np.concatenate([x[:, :, :, j], x[:, :, :, j], x[:, :, :, j]], axis=1), 1.0, cv2.applyColorMap(att_map[:, :, j], cv2.COLORMAP_INFERNO), 0.5, 0)

            out.write(frame)

        out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/gih5/mounts/nfs_echo_yale/031522_echo_avs_preprocessed')
    parser.add_argument('--gradcam_dirs', type=str, nargs='+', required=True, help='path to directories with gradcam output (should be list of paths separated with spaces')

    args = parser.parse_args()

    print(args)

    main(args)
