import os
import random

import cv2
import numpy as np
import pandas as pd
import torch

from scipy.ndimage import rotate

from utils import load_video

class EchoDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset that loads echocardiogram videos for supervised finetuning.

    Attributes
    ----------
    split: str
        Data split used to select studies that will be loaded (one of ["train", "val", "test", "ext_test"])
    clip_len : int
        Number of frames to form video clips for training (clip length)
    num_clips : int
        Number of clips to use for each video during inference (will be stacked along zeroth dimension with outputs averaged to form video-level prediction)
    sampling_rate : int
        Temporal "stride" when sampling frames to form clips (e.g., sampling_rate=1 samples consecutive frames)
    augment : bool
        Whether to perform data augmentation (spatially consistent augmentations to individual frames)
    kinetics : bool
        Whether or not to perform channel-wise normalization according to the Kinetics-400 training set mean and standard deviation
    video_dir : str
        Path to directory containing videos in .avi format
    label_df : pandas DataFrame
        Data frame containing file names and labels
    CLASSES : list[str]
        List of target class names
    mean : np.ndarray
        Channel-wise means of Kinetics-400 training set used for normalization
    std : np.ndarray
        Channel-wise standard deviations of Kinetics-400 training set used for normalization
    
    Methods
    -------
    _sample_frames(x)
        Subsample frames of video to form a video "clip" (returns multiple clips stacked along zeroth dimension during inference)
    _augment(x)
        Apply spatial augmentation to frames of video clip
    __len__
        Returns "length"/size of dataset
    __getitem__
        Returns dictionary {'x': [torch.FloatTensor] echo video clip,
                            'y': [torch.FloatTensor] classification label,
                            'acc_num': [str] accession number (study ID),
                            'video_num': [str] video ID to distinguish videos from same study,
                            'plax_prob': [float] predicted probability of video being from the PLAX view, as determined by view classifier}
    """
    def __init__(self, data_dir, split, clip_len=16, sampling_rate=1, num_clips=4, augment=False, frac=1.0, kinetics=False):
        assert split in ['100122_train_2016-2020', '100122_val_2016-2020', '100122_test_2016-2020', '100122_test_2021', '051823_test_2016-2020', '051823_full_test_2016-2020'], "split must be one of ['100122_train_2016-2020', '100122_val_2016-2020', '100122_test_2016-2020', '100122_train_2021', '051823_test_2016-2020', '051823_full_test_2016-2020']"

        self.split = split
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.sampling_rate = sampling_rate
        self.augment = augment
        self.kinetics = kinetics

        self.video_dir = os.path.join(data_dir, 'videos')
        self.label_df = pd.read_csv(os.path.join(data_dir, self.split + '.csv'))

        # Subset fraction of studies
        if frac != 1.0:
            study_ids = np.sort(self.label_df['acc_num'].unique())

            self.label_df = self.label_df[self.label_df['acc_num'].isin(np.random.choice(study_ids, size=int(frac*study_ids.size), replace=False))]

            print('Num studies:', int(frac*study_ids.size))
        print(self.label_df['severe_AS'].value_counts())

        self.CLASSES = ['None', 'Severe']

        # Kinetics-400 mean and std
        self.mean = np.array([0.43216, 0.394666, 0.37645])
        self.std = np.array([0.22803, 0.22145, 0.216989])

    def _sample_frames(self, x):
        if self.clip_len is not None:
            if self.split == '100122_train_2016-2020':
                if x.shape[0] > self.clip_len*self.sampling_rate:
                    start_idx = np.random.choice(x.shape[0]-self.clip_len*self.sampling_rate, size=1)[0]
                    x = x[start_idx:(start_idx+self.clip_len*self.sampling_rate):self.sampling_rate]
                    x = np.transpose(x, (3, 0, 1, 2))
                else:
                    x = x[::self.sampling_rate]
                    x = np.pad(x, ((0, self.clip_len-x.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant')
                    x = np.transpose(x, (3, 0, 1, 2))
            else:
                if x.shape[0] >= self.clip_len*self.sampling_rate + self.num_clips:
                    start_indices = np.arange(0, x.shape[0]-self.clip_len*self.sampling_rate, (x.shape[0]-self.clip_len*self.sampling_rate) // self.num_clips)[:self.num_clips]
                    x = np.stack([x[start_idx:(start_idx+self.clip_len*self.sampling_rate):self.sampling_rate] for start_idx in start_indices], axis=0)
                    x = np.transpose(x, (4, 0, 1, 2, 3))
                elif x.shape[0] > self.clip_len*self.sampling_rate:
                    x = x[::self.sampling_rate]
                    x = x[:self.clip_len]
                    x = np.stack([x] * self.num_clips, axis=0)
                    x = np.transpose(x, (4, 0, 1, 2, 3))
                else:
                    x = x[::self.sampling_rate]
                    x = np.pad(x, ((0, self.clip_len-x.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant')
                    x = np.stack([x] * self.num_clips, axis=0)
                    x = np.transpose(x, (4, 0, 1, 2, 3))
        else:
            x = np.transpose(x, (3, 0, 1, 2))

        return x

    def _augment(self, x):
        # Zero-pad by up to 8 pixels
        pad = 8

        l, h, w, c = x.shape
        temp = np.zeros((l, h + 2 * pad, w + 2 * pad, c), dtype=x.dtype)
        temp[:, pad:-pad, pad:-pad, :] = x
        i, j = np.random.randint(0, 2 * pad, 2)
        x = temp[:, i:(i + h), j:(j + w), :]

        # Random horitzontal flip
        if random.uniform(0, 1) > 0.5:
            x = np.stack([cv2.flip(frame, 1) for frame in x], axis=0)

        # Random rotation between -10 and 10 degrees
        if random.uniform(0, 1) > 0.5:
            angle = np.random.choice(np.arange(-10, 11), size=1)[0]

            x = np.stack([rotate(frame, angle, reshape=False) for frame in x], axis=0)

        return x

    def __len__(self):
        return self.label_df.shape[0]

    def __getitem__(self, idx):
        if self.split == '100122_test_2016-2020':
            plax_prob, fname, acc_num, _, video_num, label, _, site = self.label_df.iloc[idx, :]
        elif self.split in ['051823_test_2016-2020', '051823_full_test_2016-2020']:
            plax_prob, fname, acc_num, _, video_num, label, _, = self.label_df.iloc[idx, :]
        else:
            plax_prob, fname, acc_num, _, video_num, label = self.label_df.iloc[idx, :]

        if self.split in ['051823_test_2016-2020', '051823_full_test_2016-2020']:
            x = load_video(fname)
        else:
            x = load_video(os.path.join(self.video_dir, fname))

        if self.augment:
            x = self._augment(x)

        x = self._sample_frames(x)

        x = (x - x.min()) / (x.max() - x.min())

        if self.kinetics:
            if self.split == '100122_train_2016-2020' or self.clip_len is None:
                x -= self.mean.reshape(3, 1, 1, 1)
                x /= self.std.reshape(3, 1, 1, 1)
            else:
                x -= self.mean.reshape(3, 1, 1, 1, 1)
                x /= self.std.reshape(3, 1, 1, 1, 1)

        y = np.array([label])

        if self.split == '100122_test_2016-2020':
            return {'x': torch.from_numpy(x).float(), 'y': torch.from_numpy(y).float(), 'acc_num': acc_num, 'video_num': video_num, 'plax_prob': plax_prob, 'site': site}
        else:
            return {'x': torch.from_numpy(x).float(), 'y': torch.from_numpy(y).float(), 'acc_num': acc_num, 'video_num': video_num, 'plax_prob': plax_prob}
            