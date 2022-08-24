"""
Author: Greg Holste

Description: Deidentifies echocardiogram videos by masking and cropping out identifying information. Specifically, this script converts DICOM files
that contain short video clips to .avi files and, optionally, extracts patient and scanner metadata from DICOM headers into .csv files.

Usage: python deidentify.py --data_dir <path_to_data> --output_dir <my_data_dir>
"""

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

def crop_echo(echo, method='simple', buffer=10):
    """
    Deidentifies echo video by masking out periphery and tightly cropping the image content.
    Code adapted from https://bitbucket.org/rahuldeo/echocv/src/master/echoanalysis_tools.py

    Parameters
    ----------
        echo : ndarray
            Numpy array of shape (frames, height, width, 3) representing an echocardiogram video clip
        method : str
            One of ["simple", "full"]. If "simple", mask peripheral pixels. If "full", use image processing steps to more throughly mask non-central pixels
        buffer : int
            Number of pixels to pad on all sides when cropping final masked echo clip

    Returns
    -------
        out : ndarray
            Numpy array of shape (frames, cropped_height, cropped_width) with deidentified echo clip

    """

    assert method in ['simple', 'full'], "'method' must be one of ['simple', 'full']"

    if method == 'simple':
        f, h, w, c = echo.shape

        mask = np.zeros_like(echo)
        mask[:, int(0.1*h):int(0.95*h), int(0.1*w):int(0.95*w), :] = 1

        masked_echo = echo * mask

        return masked_echo[:, :, :, 0]
    else:
        # Extract first frame
        frame = echo[0, :, :, 0].astype(np.uint8).copy()

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

        masked_echo = np.array([frame[:, :, 0] * mask for frame in echo])

        x, y, w, h = cv2.boundingRect(contour)
        out = masked_echo[:, y-buffer:y+h+buffer, x-buffer:x+w+buffer]

        return out

def dcm_to_avi(file_path, output_dir, verbose=False):
    """Convert DICOM file to .avi video."""
    echo_id, file_name = file_path.split('/')[-3], file_path.split('/')[-1].split('.')[0]

    # Read in DICOM and extract pixel data
    try:
        dataset = dcm.dcmread(file_path, force=True)
    except:
        if verbose:
            print('Could not load DCM file')
        return

    try:
        img_data = dataset.pixel_array
    except:
        if verbose:
            print('Could not find pixel data')
        return

    # If file does not contain video, ignore
    if len(img_data.shape) != 4:
        if verbose:
            print('DICOM file does not contain video.')
        return

    # Set FPS for cv2 video writing
    f, h, w, c = img_data.shape
    try:
        try:
            fps = dataset[(0x18, 0x40)].value
        except:
            frametime = dataset[0x0018, 0x1063].value
            fps = 1 / (frametime / 1000)
            if verbose:
                print(file_name, f'Frame rate is {fps}.')
    except:
        fps = 30
        if verbose:
            print(file_name, 'Could not find frame rate. Default to 30.')

    # My observation: if FPS ~18, clip is either Doppler or something similar we don't want
    if fps < 25:
        return

    # # Deidentify echo video
    cropped_echo = crop_echo(dataset.pixel_array)
    # cropped_echo = dataset.pixel_array

    if cropped_echo.shape[0] > 100:
        cropped_echo = cropped_echo[:100]

    # Prepare to write to .avi file
    # out_file_path = os.path.join(output_dir, echo_id, 'videos', file_name + '.avi')
    out_file_path = os.path.join(output_dir, f'{echo_id}_{file_name}.avi')

    if os.path.exists(out_file_path):
        out_file_path = os.path.join(output_dir, f'{echo_id}_{file_name}_2.avi') 

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(out_file_path, fourcc, fps, (cropped_echo.shape[2], cropped_echo.shape[1]), False)

    # Write frames to .avi file
    write_failed = False
    for frame in cropped_echo:
        try:
            out.write(frame)
        except:
            write_failed = True
            if verbose:
                print('Frame failed to write:', frame.shape, out_file_path)

    out.release()

    if write_failed:
        print(echo_id, file_name, img_data.shape, fps)
        os.remove(out_file_path)

def main_proc(echo_id, args):
    for sub_dir in os.listdir(os.path.join(args.data_dir, echo_id)):
        if not os.path.isdir(os.path.join(args.data_dir, echo_id, sub_dir)):
            continue

        dcm_files = [f for f in os.listdir(os.path.join(args.data_dir, echo_id, sub_dir)) if f.endswith('.dcm')]

        if len(dcm_files) > 10:
        # if sub_dir.endswith('_1'):
            echo_pbar = os.listdir(os.path.join(args.data_dir, echo_id, sub_dir))

            # Iterate through all DICOM files for a given echo
            for dcm_file in echo_pbar:
                dcm_path = os.path.join(args.data_dir, echo_id, sub_dir, dcm_file)

                # Convert DICOM -> .avi (if DICOM contains a video)
                dcm_to_avi(dcm_path, args.output_dir, args.verbose)
        # # Extract metadata from echo "summary" DICOM file (located in subdirectory that ends in "_2")
        # elif sub_dir.endswith('_2'):
        #     if args.verbose:
        #         print(f'Extracting metadata from {echo_id}...')

        #     dcm_path = os.path.join(args.data_dir, echo_id, sub_dir, '0.dcm')

        #     extract_metadata(dcm_path, args.output_dir, args.verbose)
        else:
            continue

def main(args):
    s = time.perf_counter()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        processed_acc_nums = set()
    else:
        processed_acc_nums = set([f.split('_')[0] for f in os.listdir(args.output_dir)])
    
    all_acc_nums = sorted(os.listdir(args.data_dir))
    
    acc_nums = [f for f in all_acc_nums if f not in processed_acc_nums]

    results = ProgressParallel(use_tqdm=True, total=len(acc_nums), n_jobs=args.n_jobs)(delayed(main_proc)(acc_num, args) for acc_num in acc_nums)
    
    e = time.perf_counter()
    print('Time elapsed:', datetime.timedelta(seconds=e-s))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/gih5/echos_111221/data_copy/', help='path to raw echo data')
    parser.add_argument('--output_dir', type=str, required=True, help='path to desired output directory')
    parser.add_argument('--n_jobs', type=int, default=1, help='number of cores for parallel processing')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    print(args)

    main(args)
