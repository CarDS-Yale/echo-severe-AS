import os
import random
import shutil

import argparse
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
import torch

from skimage.measure import block_reduce

class EchoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, acc_nums=None):
        self.data_dir = data_dir

        self.files = [os.path.join(self.data_dir, f) for f in sorted(os.listdir(self.data_dir))]

        if acc_nums is not None:
            self.files = [f for f in self.files if f.split('/')[-1].split('_')[0] in acc_nums]

        self.skipped_files = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]

        try:
            x = load_video(self.files[idx])
        except OSError as e:
            print(fpath, e)
            self.skipped_files.append(fpath)
            x = None

        return {'fpath': fpath, 'x': x}

def load_video(fname):
    capture = cv2.VideoCapture(fname)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        frame_indices = np.random.choice(frame_count, size=10, replace=False)
    except:
        print(frame_count, frame_width, frame_width, fname)
        return None

    v = np.zeros((10, 224, 224), np.uint8)

    for i, frame_idx in enumerate(frame_indices):
        capture.set(cv2.CAP_PROP_FRAME_COUNT, frame_idx)
        
        ret, frame = capture.read()

        if not ret:
            return None

        frame = cv2.resize(frame[:, :, 0], (224, 224), interpolation=cv2.INTER_AREA)

        v[i, :, :] = frame

    v = v[..., None]

    return v

def collate_fn(data):
    fpath, fps = [], []
    x = []
    for d in data:
        if d['x'] is not None:
            fpath.append(d['fpath'])

            x.append(d['x'])
    x = np.concatenate(x, axis=0)
    fpath = np.array(fpath)

    return {'fpath': fpath, 'x': x}

def main(args):
    np.random.seed(0)
    random.seed(0)

    CLASSES = np.array(pd.read_csv('viewclasses_view_23_e5_class_11-Mar-2018.txt', header=None).iloc[:, 0].values)

    feature_dim = 1
    label_dim = 23
    model_name = 'view_23_e5_class_11-Mar-2018'

    tf.reset_default_graph()
    sess = tf.Session()
    model = Network(0.0, 0.0, feature_dim, label_dim, False)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, model_name)

    out_path = os.path.join(args.output_dir, args.csv_name)

    if os.path.exists(out_path):
        plax_df = pd.read_csv(out_path)

        processed_acc_nums = set([f.split('/')[-1].split('_')[0] for f in plax_df['fpath']])

        all_acc_nums = sorted(list(set([f.split('_')[0] for f in os.listdir(args.data_dir)])))

        acc_nums = [a for a in all_acc_nums if a not in processed_acc_nums]

        print('# studies in source directory:', len(all_acc_nums))
        print('# studies already classified:', len(processed_acc_nums))
    else:
        acc_nums = None

    dataset = EchoDataset(data_dir=args.data_dir, acc_nums=acc_nums)
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=36, num_workers=8, collate_fn=collate_fn, worker_init_fn=val_worker_init_fn)

    plax_probs, plax_fpaths = [], []
    for batch in tqdm.tqdm(loader, total=len(loader)):

        x = batch['x']
        fpath = batch['fpath']

        probs = block_reduce(model.probabilities(sess, x), block_size=(10,1), func=np.mean)  # (n, 23)

        max_idx = np.argmax(probs, axis=1)  # (n, 1)

        plax_idx = np.where(CLASSES[max_idx] == 'plax_plax')[0]

        if plax_idx.size == 0:
            continue

        max_idx = max_idx[plax_idx]

        plax_probs.append(probs[plax_idx, max_idx])
        plax_fpaths.append(fpath[plax_idx])

    plax_probs = np.concatenate(plax_probs)
    plax_fpaths = np.concatenate(plax_fpaths)

    print(plax_probs.shape)
    print(plax_fpaths.shape)

    out_df = pd.DataFrame({'plax_prob': plax_probs, 'fpath': plax_fpaths})
    if os.path.exists(out_path):
       out_df.to_csv(out_path, mode='a', index=False, header=False)
    else:
       out_df.to_csv(out_path, index=False)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--csv_name', type=str)

    args = parser.parse_args()

    print(args)

    main(args)
