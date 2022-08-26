import os

import cv2
import numpy as np

from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score

def get_metrics(y_true, y_hat):
    fprs, tprs, thrs = roc_curve(y_true, y_hat)

    auroc = auc(fprs, tprs)

    idx = np.where(tprs >= 0.9)[0]
    spec_at_90_sens = 1-fprs[idx].min()

    precs, recalls, thrs = precision_recall_curve(y_true, y_hat)
    aupr = auc(recalls, precs)

    idx = np.where(recalls >= 0.9)[0]
    try:
        partial_aupr_90_sens = auc(recalls[idx], precs[idx])
    except:
        partial_aupr_90_sens = 0

    prec_90_sens = precs[idx].max()

    f1_scores = 2*recalls*precs / (recalls+precs+1e-12)
    idx = np.argmax(f1_scores)

    f1 = f1_scores[idx]
    pr = precs[idx]
    re = recalls[idx]

    return {'auroc': auroc, 'aupr': aupr, 'f1': f1, 'precision': pr, 'recall': re, 'spec@90sens': spec_at_90_sens, 'partial_aupr': partial_aupr_90_sens, 'prec@90sens': prec_90_sens}


def bootstrap(pred_df, metrics, alpha=0.95, n_samples=1000):
    boot_metrics_dict = {}
    for metric in metrics:
        boot_metrics_dict[metric] = []

    for i in range(n_samples):
        boot_df = pred_df.sample(frac=1, random_state=i, replace=True)

        boot_metrics = get_metrics(boot_df['y_true'], boot_df['y_hat'])

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