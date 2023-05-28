import itertools
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm

from scipy.ndimage import rotate

from utils import load_video

class EchoDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset that loads echocardiogram videos for self-supervised pretraining. This extends the SimCLR framework by selecting two *different* videos from the
    *same* study (patient) as "positive pairs", rather than artifically generating two views of the same sample via augmentation. For each of the two selected videos,
    a clip is randomly subsampled, that clip is lightly augmented in a spatially consistent manner, then the frames of the clip are randomly shuffled (permuted).
    Frame re-ordering is used as an auxiliary pretraining task, so video clip #1, video clip #2, true frame order #1, and true frame order #1 are returned at each iteration.
    The dataset consists of all unique pairs of different videos from the same study.

    Attributes
    ----------
    split: str
        Data split used to select studies that will be loaded (one of ["train", "val", "test", "ext_test"])
    clip_len : int
        Number of frames to form video clips for training (clip length)
    sampling_rate : int
        Temporal "stride" when sampling frames to form clips (e.g., sampling_rate=1 samples consecutive frames)
    video_dir : str
        Path to directory containing videos in .avi format
    label_df : pandas DataFrame
        Data frame containing file names and labels (only file names used here)
    fnames_i : list[str]
        List of paths to "video #1" to be returned at each iteration
    fnames_j : list[str]
        List of paths to "video #2" to be returned at each iteration
    temporal_orderings : list[tuple]
        List of all permutations of frame indices (target classes for frame re-ordering task)
    
    Methods
    -------
    _sample_frames(x)
        Subsample frames of video to form a video "clip" for training
    _augment(x)
        Apply spatial augmentation to frames of video clip, then randomly shuffle frames
    __len__
        Returns "length"/size of dataset
    __getitem__
        Returns video clip #1, video clip #2, frame order label #1, and frame order label #2 as described above
    """
    def __init__(self, data_dir, split, clip_len=16, sampling_rate=1, n=None):
        self.split = split
        self.clip_len = clip_len
        self.sampling_rate = sampling_rate

        self.video_dir = os.path.join(data_dir, 'videos')
        self.label_df = pd.read_csv(os.path.join(data_dir, self.split + '.csv'))

        if n is not None:
            self.label_df = self.label_df.iloc[:n, :]

        study_ids = np.sort(self.label_df['acc_num'].unique())

        self.fnames_i = []
        self.fnames_j = []
        for study_id in tqdm.tqdm(study_ids):
            fnames = self.label_df[self.label_df['acc_num'] == study_id]['fpath'].values.tolist()

            if len(fnames) == 1:
                self.fnames_i.append(fnames[0])
                self.fnames_j.append(fnames[0])
            else:
                for fname_pair in itertools.combinations(fnames, 2):
                    self.fnames_i.append(fname_pair[0])
                    self.fnames_j.append(fname_pair[1])

        self.temporal_orderings = [_ for _ in itertools.permutations(np.arange(self.clip_len))]

    def _sample_frames(self, x):
        if x.shape[0] > self.clip_len*self.sampling_rate:
            start_idx = np.random.choice(x.shape[0]-self.clip_len*self.sampling_rate, size=1)[0]
            x = x[start_idx:(start_idx+self.clip_len*self.sampling_rate):self.sampling_rate]
        else:
            x = x[::self.sampling_rate]
            x = np.pad(x, ((0, self.clip_len-x.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant')

        return x

    def _augment(self, x):
        # Zero-pad by up to 8 pixels
        pad = 8

        l, h, w, c = x.shape
        temp = np.zeros((l, h + 2 * pad, w + 2 * pad, c), dtype=x.dtype)
        temp[:, pad:-pad, pad:-pad, :] = x
        i, j = np.random.randint(0, 2 * pad, 2)
        x = temp[:, i:(i + h), j:(j + w), :]

        # Random horizontal flip
        if random.uniform(0, 1) > 0.5:
            x = np.stack([cv2.flip(frame, 1) for frame in x], axis=0)

        # Random rotation between -10 and 10 degrees
        if random.uniform(0, 1) > 0.5:
            angle = np.random.choice(np.arange(-10, 11), size=1)[0]

            x = np.stack([rotate(frame, angle, reshape=False) for frame in x], axis=0)

        # Frame re-ordering
        reordering_label = np.random.choice(len(self.temporal_orderings), size=1)[0]
        reordering = self.temporal_orderings[reordering_label]
        x = x[reordering, :, :, :]

        return x, reordering_label

    def __len__(self):
        return len(self.fnames_i)

    def __getitem__(self, idx):
        x_i = load_video(os.path.join(self.video_dir, self.fnames_i[idx]))
        x_j = load_video(os.path.join(self.video_dir, self.fnames_j[idx]))

        # Sample frames to form clip from each "view"
        x_i = self._sample_frames(x_i)
        x_j = self._sample_frames(x_j)

        # Augment each view and obtain frame ordering label
        x_i, reordering_i = self._augment(x_i)
        x_j, reordering_j = self._augment(x_j)
        reordering_i = np.array(reordering_i)
        reordering_j = np.array(reordering_j)

        # Min-max normalize and swap axes for PyTorch
        x_i = (x_i - x_i.min()) / (x_i.max() - x_i.min())
        x_j = (x_j - x_j.min()) / (x_j.max() - x_j.min())

        x_i = np.transpose(x_i, (3, 0, 1, 2))
        x_j = np.transpose(x_j, (3, 0, 1, 2))

        return torch.from_numpy(x_i).float(), torch.from_numpy(x_j).float(), torch.from_numpy(reordering_i).long(), torch.from_numpy(reordering_j).long()