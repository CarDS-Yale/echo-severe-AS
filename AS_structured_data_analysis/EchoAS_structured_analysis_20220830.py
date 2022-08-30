# %% [markdown]
# # Analysis of Yale structured echo parameters for the paper "Automated detection of severe aortic stenosis using single-view echocardiography: A self-supervised ensemble learning approach" by Holste, Oikonomou, Khera et al.

# %%
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

oldpath = "/Users/evangelos.oikonomou/Box/Evan Rohan shared folder/ECG_echo/echo/"
newpath = "/Users/evangelos.oikonomou/Box/Echo_AS/"

# %% [markdown]
# ### Loading all echo data until 12/2021 and summarizing counts per year

# %%
# Load all echos
echo_all = pd.read_csv(oldpath+"echo_all_Dec_2021.csv")

# Summarize number of echos by year
echo_all['year'] = pd.DatetimeIndex(echo_all['Procedure Date']).year
echo_all['year'].value_counts()

# %% [markdown]
# ### Extract ejection fraction

# %%
echo_all["EF to Report"].value_counts()

# %%
# Get ejection fraction
echo_all["EF Range Minimum"] = echo_all["EF Range"].str.extract('(\d+)').astype(float)
echo_all["EF"] = np.where(pd.notna(echo_all["EF% 3DE"]), echo_all["EF% 3DE"], 
                    np.where(pd.notna(echo_all["EF% BiPlane"]), echo_all["EF% BiPlane"], echo_all["EF Range Minimum"]))

# %% [markdown]
# ### Create dataframes for training, validation, testing

# %%
# Get IDs
ids = pd.read_csv(newpath+'070322_id_by_split.csv')
ids["split"].value_counts()

# %%
# IDs
train_ids = ids[ids["split"]=="train"]["acc_num"].unique()
val_ids = ids[ids["split"]=="val"]["acc_num"].unique()
test_ids = ids[ids["split"]=="test"]["acc_num"].unique()
ext_test_ids = ids[ids["split"]=="ext_test"]["acc_num"].unique()
deriv_ids = set(train_ids) | set(val_ids)


# %%
# Create the derivation set
merged_train = echo_all[echo_all["Accession Number"].isin(deriv_ids)]
len(merged_train)

# %%
# Create the internal testing set
int_test = pd.read_csv(newpath+"061022_ensemble_test_preds.csv")
int_test["Accession Number"] =int_test["acc_num"]
merged_int_test = int_test.merge(echo_all, on='Accession Number', how='left')


# %%
# Create the external testing set
test = pd.read_csv(newpath+"061022_ensemble_ext_test_preds.csv")
test["Accession Number"] = test["acc_num"]
merged_test = test.merge(echo_all, on='Accession Number', how='left')

# %% [markdown]
# ### Get descriptive statistics

# %%
# Get unique number of patients
fordemo_train = merged_train.drop_duplicates(subset=['MRN'], keep='first')
merged_train["MRN"].nunique()

# Get numbers of males/females
merged_train["Gender"].value_counts(normalize=False)

# Get distribution of age
merged_train["Age"].describe()

# Get AS severity
merged_train["AV Stenosis"].value_counts(normalize=False)


# %% [markdown]
# ### Define the testing set

# %%
# Create the testing set
testing = pd.read_csv(newpath + "061022_ensemble_ext_test_preds.csv")
testing["Accession Number"] = testing["acc_num"]

# Merged external predictions with echo data
merged_test = testing.merge(echo_all, on='Accession Number', how='left')


# %%
merged_test['Accession Number'].nunique()

# %%
merged_test['MRN'].nunique()

# %%
merged_test["Gender"].value_counts(normalize=False)

# %%
merged_test["Age"].describe()

# %%
merged_test["error_type"].value_counts()

# %%
merged_test["AV Stenosis"].value_counts()

# %% [markdown]
# ### Plotting the correlation between the model predictions and the continuous metrics of aortic stenosis severity

# %%
merged_test["Predicted probability"] = merged_test["y_hat"]
merged_test["Peak aortic valve velocity (m/sec)"] = merged_test["AV Pk Vel (m/s)"]
merged_test["Mean aortic valve gradient (mm Hg)"] = merged_test["AV Mn Grad (mmHg)"]
merged_test["Aortic valve area (cm2)"] = merged_test["AVA Cont VTI"]

# %%
newdf_pkvel = pd.concat([merged_test["Predicted probability"], merged_test["Peak aortic valve velocity (m/sec)"]], axis=1, sort=False)
newdf_pkvel = newdf_pkvel.dropna()
stats.pearsonr(newdf_pkvel["Predicted probability"], newdf_pkvel["Peak aortic valve velocity (m/sec)"])

# %%
newdf_mngrd = pd.concat([merged_test["Predicted probability"], merged_test["Mean aortic valve gradient (mm Hg)"]], axis=1, sort=False)
newdf_mngrd = newdf_mngrd.dropna()
stats.pearsonr(newdf_mngrd["Predicted probability"], newdf_mngrd["Mean aortic valve gradient (mm Hg)"])

# %%
newdf_ava = pd.concat([merged_test["Predicted probability"], merged_test["Aortic valve area (cm2)"]], axis=1, sort=False)
newdf_ava = newdf_ava.dropna()
stats.pearsonr(newdf_ava["Predicted probability"], newdf_ava["Aortic valve area (cm2)"])

# %% [markdown]
# ### Plot scatterplots

# %%
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(8,6))
sns.regplot(y="Predicted probability", x="Peak aortic valve velocity (m/sec)", data=newdf_pkvel, ax=ax1, scatter_kws={'s':2})
sns.regplot(y="Predicted probability", x="Mean aortic valve gradient (mm Hg)", data=newdf_mngrd, ax=ax2, scatter_kws={'s':2})
#sns.regplot(y="Predicted probability", x="Aortic valve area (cm2)", data=newdf_ava, ax=ax3, scatter_kws={'s':2})
plt.ylim(-0.1, 1.1)
fig.show()


# %% [markdown]
# ### Plot violin plots

# %%
list = ["fp", "tn"]

forplot = merged_test[merged_test["error_type"].isin(list)]

forplot = forplot.replace({'error_type': {"fp": "Severe", "tn": "Non-severe"}})
forplot["Prediction"] = forplot["error_type"]
forplot["LVEF (%)"] = forplot["EF"]
forplot["Peak aortic valve velocity (m/sec)"] = forplot["AV Pk Vel (m/s)"]
forplot["Mean aortic valve gradient (mm Hg)"] = forplot["AV Mn Grad (mmHg)"]
forplot["Aortic valve area (cm2)"] = forplot["AVA Cont VTI"]


fig, ax =plt.subplots(2,2,figsize=(16,12))
sns.violinplot(x=forplot["Prediction"], y=forplot["LVEF (%)"], palette="muted", ax=ax[0,0])
sns.violinplot(x=forplot["Prediction"],  #hue=forplot["LVEF40"], 
                y=forplot["Peak aortic valve velocity (m/sec)"], palette="muted", split=True, ax=ax[0,1])
sns.violinplot(x=forplot["Prediction"], y=forplot["Mean aortic valve gradient (mm Hg)"], palette="muted", ax=ax[1,0])
sns.violinplot(x=forplot["Prediction"], #hue=forplot["LVEF40"], 
                    y=forplot["Aortic valve area (cm2)"], palette="muted", split=True, ax=ax[1,1])
fig.show()


# %% [markdown]
# ### Get p values for pairwise comparisons

# %%
forplot = merged_test[merged_test["error_type"].isin(list)]

# %%
# Get t-test P value for LVEF
from scipy import stats
forplot_ef = forplot.dropna(subset=["EF"])
stats.mannwhitneyu(forplot_ef['EF'][forplot_ef['error_type'] == 'tn'], forplot_ef['EF'][forplot_ef['error_type'] == 'fp'])

# %%
forplot_ef[["EF", "error_type"]].groupby('error_type').describe()

# %%
# Get t-test P value for AV peak velocity
from scipy import stats
forplot_av = forplot.dropna(subset=["AV Pk Vel (m/s)"])
stats.mannwhitneyu(forplot_av['AV Pk Vel (m/s)'][forplot_av['error_type'] == 'tn'], forplot_av['AV Pk Vel (m/s)'][forplot_av['error_type'] == 'fp'])

# %%
forplot_av[["AV Pk Vel (m/s)", "error_type"]].groupby('error_type').describe()

# %%
# Get t-test P value for AV mean gradient
from scipy import stats
forplot_av = forplot.dropna(subset=["AV Mn Grad (mmHg)"])
stats.mannwhitneyu(forplot_av['AV Mn Grad (mmHg)'][forplot_av['error_type'] == 'tn'], forplot_av['AV Mn Grad (mmHg)'][forplot_av['error_type'] == 'fp'])

# %%
forplot_av[["AV Mn Grad (mmHg)", "error_type"]].groupby('error_type').describe()

# %%
# Get t-test P value for AVA by cont VTI
from scipy import stats
forplot_av = forplot.dropna(subset=["AVA Cont VTI"])
stats.mannwhitneyu(forplot_av['AVA Cont VTI'][forplot_av['error_type'] == 'tn'], forplot_av['AVA Cont VTI'][forplot_av['error_type'] == 'fp'])

# %%
forplot_av[["AVA Cont VTI", "error_type"]].groupby('error_type').describe()

# %% [markdown]
# ### Create table of demographics

# %%
merged_train["Group"]="1. Derivation (training & validation)"
merged_int_test["Group"]="2. Internal testing"
merged_test["Group"]="3. External testing"
merged = pd.concat([merged_train, merged_int_test, merged_test])
merged.reset_index(inplace=True)

# %%
merged["Race"] = merged['Race'].replace({'AMERICAN INDIAN/ALASKAN NATIVE':'Other', 
                                        'ARABIC':'Other',
                                        'African American':'Black',
                                        'Asian':'Asian',
                                        'ASIAN':'Asian',
                                        'BLACK/AFRICAN AMERICAN':'Black',
                                        'Black':'Black',
                                        'Caucasian':'White',
                                        'HISPANIC/LATINO':'Hispanic',
                                        'Hispanic':'Hispanic',
                                        'NATIVE HAWAIIAN/PACIFIC ISLANDER':'Other',
                                        'OTHER':'Other',
                                        'Other':'Other',
                                        'UNKNOWN':'Unknown',
                                        'WHITE/CAUCASIAN':'White'
                                        })

merged["AV Stenosis"] = merged['AV Stenosis'].replace({'Mild-Mod':'Mild', 
                                        'Mod-Sev':'Moderate'
                                        })


# %%
tableone = merged[["Group", "year", "Age", "Gender", "Race", "BMI", "BP Systolic", "BP Diastolic", "LVIDd Index", "LA Vol Indexed", "RVSP (mmHg)", "EF", "AV Stenosis", "AVA Cont VTI", "AV Mn Grad (mmHg)", "AV Pk Vel (m/s)"]]
nonnormal = ["Age", "BMI", "BP Systolic", "BP Diastolic", "LVIDd Index", "LA Vol Indexed", "RVSP (mmHg)", "EF", "AVA Cont VTI", "AV Mn Grad (mmHg)", "AV Pk Vel (m/s)"]

from tableone import TableOne
mytable = TableOne(tableone, groupby="Group", pval=True)#, nonnormal=nonnormal
print(mytable.tabulate(tablefmt="simple"))
mytable.to_csv('/Users/evangelos.oikonomou/Box/Echo_AS/070422_table.csv')


