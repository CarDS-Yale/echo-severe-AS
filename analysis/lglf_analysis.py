import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve

threshold = 0.6075266666666667

x = pd.read_csv('020823_lglf_study_predictions.csv')
x['severe_AS'] = 1
print(x)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax = sns.violinplot(ax=ax, data=x, y='ensemble')
ax.set_xlabel('Ensemble', fontsize=12)
ax.set_ylabel('Predicted Probability', fontsize=12)
fig.tight_layout()
fig.savefig('021523_lglf_pred_violin.pdf', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax = sns.histplot(ax=ax, data=x, x='ensemble', bins=10, kde=True)
ax.set_xlabel('Predicted Probability', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_xlim([0, 1])
fig.tight_layout()
fig.savefig('021523_lglf_pred_hist.pdf', bbox_inches='tight')

y_pred = (x['ensemble'] >= threshold).astype(int)

print(precision_recall_fscore_support(x['severe_AS'], y_pred))
print(np.sum(y_pred) / x.shape[0])
