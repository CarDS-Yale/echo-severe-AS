import datetime
import os
import shutil
import time

import argparse
import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import pydicom as dcm
import tqdm

from utils import ProgressParallel

def crop_echo(echo):
    """
    Deidentifies echo video by masking out periphery and tightly cropping the image content.
    Code adapted from https://bitbucket.org/rahuldeo/echocv/src/master/echoanalysis_tools.py
    """
    # Extract first frame
    frame = echo[0].astype(np.uint8).copy()

    # Binarize image, apply low-pass filter, and threshold
    frame[frame > 0] = 255
    thresh = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY)[1]

    # Find largest contour (central image content)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # Create binary mask for central image content to mask out periphery
    hull = cv2.convexHull(np.array(contour, dtype=np.int32))
    mask = np.zeros(frame.shape, dtype=np.uint8)
    mask = cv2.fillConvexPoly(mask, hull, 1)

    masked_echo = np.array([frame * mask for frame in echo])

    return masked_echo

def load_video(fname):
    capture = cv2.VideoCapture(fname)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count == 0:
        return None, None

    v = np.zeros((frame_count, frame_height, frame_width), np.uint8)

    for count in range(frame_count):        
        ret, frame = capture.read()

        if not ret:
            return None, None

        v[count, :, :] = frame[:, :, 0]

    return v, cv2.CAP_PROP_FPS

def write_video(fname, video, fps):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(fname, fourcc, fps, (video.shape[2], video.shape[1]), False)

    for frame in video:
        out.write(frame)

    out.release()

def main_proc(fpath, acc_num, video_num, output_dir):
    echo, fps = load_video(fpath)

    if echo is None:
        return

    masked_echo = crop_echo(echo)

    masked_echo = np.array([cv2.resize(frame, (112, 112), interpolation=cv2.INTER_AREA) for frame in masked_echo])

    out_file_path = os.path.join(output_dir, f'{acc_num}_{video_num}.avi')
    write_video(out_file_path, masked_echo, fps)

def main(args):
    s = time.perf_counter()

    # Create output directory (first deleting if it already exists)
    if os.path.isdir(args.output_dir):
        print(f'{args.output_dir} already exists. Deleting contents...')
        shutil.rmtree(args.output_dir)
    print(f'Creating output directory {args.output_dir}')
    os.mkdir(args.output_dir)
    os.mkdir(os.path.join(args.output_dir, 'videos'))

    # Read in label csv and view classification csv
    label_df = pd.read_csv(args.label_csv_path)
    plax_df = pd.read_csv(args.plax_csv_path)
    
    plax_df['acc_num'] = plax_df.apply(lambda x: x['fpath'].split('/')[-1].split('_')[0], axis=1)

    # Some processed echos do NOT appear in the original extracted cohort for which we have labels
    processed_acc_nums = plax_df['acc_num'].unique()
    all_acc_nums = label_df['AccessionNumber'].unique()
    acc_nums = np.intersect1d(all_acc_nums, processed_acc_nums)

    plax_df = plax_df[plax_df['acc_num'].isin(acc_nums)]
    label_df = label_df[label_df['AccessionNumber'].isin(acc_nums)]

    # Get AV Stenosis labels and group into 3 levels
    plax_df['av_stenosis'] = plax_df.apply(lambda x: label_df.loc[label_df['AccessionNumber'] == x['acc_num'], 'AVStenosis'].values[0], axis=1)
    plax_df = plax_df[~plax_df['av_stenosis'].isin(['Low gradient AS', 'Paradoxical AS'])]
    plax_df.loc[plax_df['av_stenosis'] == 'Sclerosis without Stenosis', 'av_stenosis'] = 'None'
    plax_df.loc[~plax_df['av_stenosis'].isin(['None', 'Severe']), 'av_stenosis'] = 'Non-Severe'

    # CRUCIAL: need to reset index here for later step
    plax_df = plax_df.sort_values(by=['acc_num', 'fpath'])
    plax_df = plax_df.reset_index(drop=True)

    plax_df['video_num'] = np.concatenate([range(num_videos) for num_videos in plax_df['acc_num'].value_counts().sort_index()])

    pbar = zip(plax_df['fpath'], plax_df['acc_num'], plax_df['video_num'])

    _ = ProgressParallel(use_tqdm=True, total=plax_df.shape[0], n_jobs=args.n_jobs)(delayed(main_proc)(fpath, acc_num, video_num, os.path.join(args.output_dir, 'videos')) for fpath, acc_num, video_num in pbar)

    plax_df['fpath'] = [str(acc_num) + '_' + str(vid_num) + '.avi' for acc_num, vid_num in zip(plax_df['acc_num'], plax_df['video_num'])]

    ext_test_ids = label_df.loc[label_df['ProcedureYear'] == 2021, 'AccessionNumber'].unique()

    np.random.seed(0)
    acc_nums = np.setdiff1d(np.unique(plax_df['acc_num'].values), ext_test_ids)
    train_ratio = 0.75
    val_ratio = 0.1
    test_ratio = 0.15

    train_ids = np.random.choice(acc_nums, size=int(train_ratio*acc_nums.size), replace=False)
    test_ids = np.setdiff1d(acc_nums, train_ids, assume_unique=True)
    val_ids = np.random.choice(test_ids, size=int((val_ratio/(val_ratio+test_ratio))*test_ids.size), replace=False)
    test_ids = np.setdiff1d(test_ids, val_ids, assume_unique=True)

    train_df     = plax_df[plax_df['acc_num'].isin(train_ids)]
    val_df       = plax_df[plax_df['acc_num'].isin(val_ids)]
    test_df      = plax_df[plax_df['acc_num'].isin(test_ids)]
    ext_test_df  = plax_df[plax_df['acc_num'].isin(ext_test_ids)]

    print(train_df.shape, train_ids.shape, train_ids.size/acc_nums.size)
    print(val_df.shape, val_ids.shape, val_ids.size/acc_nums.size)
    print(test_df.shape, test_ids.shape, test_ids.size/acc_nums.size)

    train_df.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(args.output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(args.output_dir, 'test.csv'), index=False)
    ext_test_df.to_csv(os.path.join(args.output_dir, 'ext_test.csv'), index=False)

    e = time.perf_counter()
    print('Time elapsed:', datetime.timedelta(seconds=e-s))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help='path to desired output directory')
    parser.add_argument('--label_csv_path', type=str, default='/home/gih5/echos_111221/echo_train_test_extract.csv')
    parser.add_argument('--plax_csv_path', type=str, default='/home/gih5/s3_echo_avs_intermediate/020922_anon_data_max-plax.csv')
    parser.add_argument('--n_jobs', type=int, default=1, help='number of cores for parallel processing')
    args = parser.parse_args()

    print(args)

    main(args)
