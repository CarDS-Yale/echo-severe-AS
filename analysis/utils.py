import os

import cv2
import numpy as np

from datetime import datetime

from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score, precision_recall_fscore_support, confusion_matrix

def bootstrap_threshold(pred_df, metrics, threshold=0.5, alpha=0.95, n_samples=1000):
    boot_metrics_dict = {}
    for metric in metrics:
        boot_metrics_dict[metric] = []

    pos_idx = np.where(pred_df['y_true'] == 1)[0]
    neg_idx = np.where(pred_df['y_true'] == 0)[0]
    for i in range(n_samples):
        ## STRATIFIED BOOTSTRAP SAMPLE (PRESERVING SAME # OF POS AND NEG EXAMPLES) TO ENSURE ENOUGH POSITIVE SAMPLES FOR EVALUATION
        ## (i.e., to avoid the scenario in which a bootstrap sample could contain 0 severe AS cases)
        np.random.seed(i)
        new_pos_idx = np.random.choice(pos_idx, size=pos_idx.size, replace=True)
        new_neg_idx = np.random.choice(neg_idx, size=neg_idx.size, replace=True)
        boot_idx = np.concatenate([new_pos_idx, new_neg_idx])

        boot_df = pred_df.iloc[boot_idx]

        boot_metrics = get_metrics_threshold(boot_df['y_true'], boot_df['y_hat'], threshold)

        for metric in metrics:
            boot_metrics_dict[metric].append(boot_metrics[metric])

    lb_dict, ub_dict = {}, {}
    for metric in metrics:
        sorted_values = np.sort(boot_metrics_dict[metric])

        lb = sorted_values[int((1-alpha) * n_samples)]
        ub = sorted_values[int(alpha * n_samples)]

        lb_dict[metric] = lb
        ub_dict[metric] = ub

    return lb_dict, ub_dict

def get_metrics_threshold(y_true, y_hat, threshold=0.5):
    fprs, tprs, thrs = roc_curve(y_true, y_hat)

    auroc = auc(fprs, tprs)

    idx = np.where(tprs >= 0.9)[0]
    spec_at_90_sens = 1-fprs[idx].min()

    precs, recalls, thrs = precision_recall_curve(y_true, y_hat)
    aupr = auc(recalls, precs)

    idx = np.where(recalls >= 0.9)[0]
    prec_90_sens = precs[idx].max()

    # Get F1, precision/ppv, recall/sensitivity, NPV
    y_pred = (y_hat >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    pr = tp / (tp + fp)
    npv = tn / (tn + fn)
    re = tp / (tp + fn)
    sp = tn / (tn + fp)
    f1 = (2 * tp) / (2 * tp + fp + fn)

    return {'auroc': auroc, 'aupr': aupr, 'f1': f1, 'precision': pr, 'npv': npv, 'recall': re, 'specificity': sp, 'spec@90sens': spec_at_90_sens, 'prec@90sens': prec_90_sens}

def get_date():
    return datetime.today().strftime("%m%d%Y")

def load_video(fname):
    capture = cv2.VideoCapture(fname)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

    for i in range(frame_count):
        ret, frame = capture.read()

        v[i] = frame

    return v