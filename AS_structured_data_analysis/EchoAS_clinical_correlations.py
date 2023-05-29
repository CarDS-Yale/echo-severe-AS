# %%
# Import libraries
import numpy as np
import pandas as pd
from scipy.stats import stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

# Define global variables/DPI for plots
mpl.rcParams['figure.dpi'] = 300

# Define paths
oldpath = "/Users/evangelos.oikonomou/Library/CloudStorage/Box-Box/Evan Rohan shared folder/ECG_echo/echo/"
newpath = "/Users/evangelos.oikonomou/Library/CloudStorage/Box-Box/Echo_AS/"

# %%
# Load all echos up until 12/2021
echo_all = pd.read_csv(oldpath+"echo_all_Dec_2021.csv")

# Get year from Procedure Date
echo_all['year'] = pd.DatetimeIndex(echo_all['Procedure Date']).year

# %%
# Define variable for severe AS
echo_all["SevereAS"] = np.where(echo_all["AV Stenosis"]=="Severe", "Severe", "Other")

# Get ejection fraction
echo_all["EF Range Minimum"] = echo_all["EF Range"].str.extract('(\d+)').astype(float)
echo_all["EF"] = np.where(pd.notna(echo_all["EF% 3DE"]), echo_all["EF% 3DE"], 
                    np.where(pd.notna(echo_all["EF% BiPlane"]), echo_all["EF% BiPlane"], echo_all["EF Range Minimum"]))

# %%
# Get IDs
ids = pd.read_csv("/Users/evangelos.oikonomou/Library/CloudStorage/Box-Box/Echo_AS/052423_yale_echo_as_ids.csv")
train_ids = ids[ids["set"]=="train"]["acc_num"].unique()
val_ids = ids[ids["set"]=="val"]["acc_num"].unique()
deriv_ids = set(train_ids).union(set(val_ids))
test_ids = ids[ids["set"]=="test"]["acc_num"].unique()
ids["set"].value_counts()

# %%
# Define the training set
merged_train = echo_all[echo_all["Accession Number"].isin(deriv_ids)]
print("In total, we have {} scans {} patients in the derivation set for New England.".format(len(merged_train), len(merged_train["MRN"].unique())))

# %%
# get number of MRNs with one unique accession number, two unique accession numbers, etc.
merged_train.groupby("MRN")["Accession Number"].nunique().value_counts()

# %%
# Define the internal testing set
test_2016_20 = pd.read_csv("/Users/evangelos.oikonomou/Library/CloudStorage/Box-Box/Echo_AS/EHJ Revision/R2/052123_051823_full_test_2016-2020_prevalence-0.015_cohort.csv")
int_test_ids = test_2016_20["acc_num"].unique()
merged_int_test = echo_all[echo_all["Accession Number"].isin(int_test_ids)]
len(merged_int_test)

print("In total, we have {} scans {} patients in the New England testing set for 2016-2020.".format(len(merged_int_test), len(merged_int_test["MRN"].unique())))

# %%
# get number of MRNs with one unique accession number, two unique accession numbers, etc.
merged_int_test.groupby("MRN")["Accession Number"].nunique().value_counts()

# %%
# Define the external testing set
test = pd.read_csv(newpath+"101822_ensemble_100122_test_2021_preds.csv")
test["Accession Number"] = test["acc_num"]
merged_test = test.merge(echo_all, on='Accession Number', how='left')

print("In total, we have {} scans {} patients in the New England testing set for 2021.".format(len(merged_test), len(merged_test["MRN"].unique())))

# %%
# get number of MRNs with one unique accession number, two unique accession numbers, etc.
merged_test.groupby("MRN")["Accession Number"].nunique().value_counts()

# %%
# Get the error types in the testing 2021 set
merged_test["error_type"].value_counts()

# %% [markdown]
# Plotting the correlation between the model predictions and the continuous metrics of aortic stenosis severity

# %%
merged_test["Predicted probability"] = merged_test["y_hat"]
merged_test["Peak aortic valve velocity (m/sec)"] = merged_test["AV Pk Vel (m/s)"]
merged_test["Mean aortic valve gradient (mm Hg)"] = merged_test["AV Mn Grad (mmHg)"]
merged_test["Aortic valve area (cm2)"] = merged_test["AVA Cont VTI"]

# %%
merged_test_tocsv = merged_test[["acc_num", "y_hat", "y_true"]]
merged_test_tocsv.to_csv("/Users/evangelos.oikonomou/Library/CloudStorage/Box-Box/Echo_AS/test_2021_preds.csv", index=False)

# %%
# Peak AV velocity
newdf_pkvel = pd.concat([merged_test["Predicted probability"], merged_test["Peak aortic valve velocity (m/sec)"]], axis=1, sort=False)
newdf_pkvel = newdf_pkvel.dropna()
stats.pearsonr(newdf_pkvel["Predicted probability"], newdf_pkvel["Peak aortic valve velocity (m/sec)"])

# %%
# Mean AV gradient
newdf_mngrd = pd.concat([merged_test["Predicted probability"], merged_test["Mean aortic valve gradient (mm Hg)"]], axis=1, sort=False)
newdf_mngrd = newdf_mngrd.dropna()
stats.pearsonr(newdf_mngrd["Predicted probability"], newdf_mngrd["Mean aortic valve gradient (mm Hg)"])

# %%
# Aortic valve area
newdf_ava = pd.concat([merged_test["Predicted probability"], merged_test["Aortic valve area (cm2)"]], axis=1, sort=False)
newdf_ava = newdf_ava.dropna()
stats.pearsonr(newdf_ava["Predicted probability"], newdf_ava["Aortic valve area (cm2)"])

# %%
# Ejection fraction
merged_test["EF Range Minimum"] = merged_test["EF Range"].str.extract('(\d+)').astype(float)
merged_test["EF"] = np.where(pd.notna(merged_test["EF% 3DE"]), merged_test["EF% 3DE"], 
                    np.where(pd.notna(merged_test["EF% BiPlane"]), merged_test["EF% BiPlane"], merged_test["EF Range Minimum"]))

newdf_ef = pd.concat([merged_test["Predicted probability"], merged_test["EF"]], axis=1, sort=False)
newdf_ef = newdf_ef.dropna()
stats.pearsonr(newdf_ef["Predicted probability"], newdf_ef["EF"])

# %%
# plot false positives versus true negatives
list = ["fp", "tn"]

forplot = merged_test[merged_test["error_type"].isin(list)]
forplot = forplot.replace({'error_type': {"fp": "Severe", "tn": "Non-Severe"}})
forplot["Model Prediction"] = forplot["error_type"]
forplot["LVEF (%)"] = forplot["EF"]
forplot["Peak aortic valve velocity (m/sec)"] = forplot["AV Pk Vel (m/s)"]
forplot["Mean aortic valve gradient (mm Hg)"] = forplot["AV Mn Grad (mmHg)"]
forplot["Aortic valve area (cm2)"] = forplot["AVA Cont VTI"]

fig, ax =plt.subplots(2,2,figsize=(12,8))
sns.violinplot(x=forplot["Model Prediction"], y=forplot["LVEF (%)"], palette="muted", ax=ax[0,0])
sns.violinplot(x=forplot["Model Prediction"],  #hue=forplot["LVEF40"], 
                y=forplot["Peak aortic valve velocity (m/sec)"], palette="muted", split=True, ax=ax[0,1])
sns.violinplot(x=forplot["Model Prediction"], y=forplot["Mean aortic valve gradient (mm Hg)"], palette="muted", ax=ax[1,0])
sns.violinplot(x=forplot["Model Prediction"], #hue=forplot["LVEF40"], 
                    y=forplot["Aortic valve area (cm2)"], palette="muted", split=True, ax=ax[1,1])
fig.show()
fig.savefig("/Users/evangelos.oikonomou/Library/CloudStorage/Dropbox/CarDS_lab/Echo DL Project Share/Echo AS Clinical paper/Manuscript/EHJ Revision 2/figures/tn_fp_comparison_20230513.pdf")


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
# # Summarize the cohorts

# %%
merged_train["Group"]="1. Derivation (training & validation)"
merged_int_test["Group"]="2. Internal testing"
merged_test["Group"]="3. External testing"
merged = pd.concat([merged_train, merged_int_test, merged_test])
merged.reset_index(inplace=True)

# %%
merged["Group"].value_counts()

# %%
# Collapse subgroups
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
# Get race from the JDAT file (more consistent)
race = pd.read_excel("/Users/evangelos.oikonomou/Library/CloudStorage/Box-Box/SMARTLV_prelim/df_race_info.xlsx")

# rename race["mrn"] to ["MRN"]
race["MRN"] = race["mrn"]

race["Race_new"] = np.where(race["race_source_value"]=="Native Hawaiian or Other Pacific Islander", "Other",
                                     np.where(race["race_source_value"]=="American Indian or Alaska Native", "Other",  race["race_source_value"]))

race["Ethnicity"] = race["ethnicity_source_value"]

# drop duplicates for "MRN"
race = race.drop_duplicates(subset=["MRN"])

# %%
# left merge tableone with race
merged_fortable = merged.merge(race, on="MRN", how="left")
merged_fortable["Group"].value_counts()

# %%
# Get unique number of patients with severe AS
mrn_severe_all = merged_fortable[(merged_fortable["AV Stenosis"]=="Severe")]["MRN"].nunique()
mrn_severe_deriv = merged_fortable[(merged_fortable["AV Stenosis"]=="Severe") & (merged_fortable["Group"]=="1. Derivation (training & validation)")]["MRN"].nunique()
mrn_severe_int = merged_fortable[(merged_fortable["AV Stenosis"]=="Severe") & (merged_fortable["Group"]=="2. Internal testing")]["MRN"].nunique()
mrn_severe_ext = merged_fortable[(merged_fortable["AV Stenosis"]=="Severe") & (merged_fortable["Group"]=="3. External testing")]["MRN"].nunique()

print("In total, there are", mrn_severe_all, "patients with severe AS.")
print("In the derivation cohort, there are", mrn_severe_deriv, "patients with severe AS.")
print("In the internal testing cohort, there are", mrn_severe_int, "patients with severe AS.")
print("In the external testing cohort, there are", mrn_severe_ext, "patients with severe AS.")

# %%
tableone = merged_fortable[["Group", "year", "Age", "Gender", "Race_new", "Ethnicity", "BMI", "BP Systolic", "BP Diastolic", "LVIDd Index", "LA Vol Indexed", "RVSP (mmHg)", "EF", "AV Stenosis", "AVA Cont VTI", "AV Mn Grad (mmHg)", "AV Pk Vel (m/s)"]]
nonnormal = ["Age", "BMI", "BP Systolic", "BP Diastolic", "LVIDd Index", "LA Vol Indexed", "RVSP (mmHg)", "EF", "AVA Cont VTI", "AV Mn Grad (mmHg)", "AV Pk Vel (m/s)"]

from tableone import TableOne
mytable = TableOne(tableone, groupby="Group", pval=True)
print(mytable.tabulate(tablefmt="simple"))
mytable.to_csv('/Users/evangelos.oikonomou/Library/CloudStorage/Box-Box/Echo_AS/20230524_tableone.csv')

# %% [markdown]
# ## Run additional analysis to correlate diastolic parameters in the 2021 testing set

# %%
echo_pred = merged_test

# remove spaces and special characters from all variable names in echo_pred
echo_pred.columns = echo_pred.columns.str.replace(' ', '_')
echo_pred.columns = echo_pred.columns.str.replace('(', '')
echo_pred.columns = echo_pred.columns.str.replace(')', '')
echo_pred.columns = echo_pred.columns.str.replace('/', '_')
echo_pred.columns = echo_pred.columns.str.replace('-', '_')
echo_pred.columns = echo_pred.columns.str.replace('.', '_')
echo_pred.columns = echo_pred.columns.str.replace('>', '')
echo_pred.columns = echo_pred.columns.str.replace('<', '')
echo_pred.columns = echo_pred.columns.str.replace('=', '')
echo_pred.columns = echo_pred.columns.str.replace('?', '')
echo_pred.columns = echo_pred.columns.str.replace(',', '')
echo_pred.columns = echo_pred.columns.str.replace(';', '')
echo_pred.columns = echo_pred.columns.str.replace(':', '')
echo_pred.columns = echo_pred.columns.str.replace('\'', '')
echo_pred.columns = echo_pred.columns.str.replace('%', '')

# %%
print(echo_pred["LA_Vol_Indexed"].describe())

# %%
# Remove extreme values > 50 for E/e'
echo_pred["E_E_Avg"] = np.where(echo_pred["E_E_Avg"]>50, np.nan, echo_pred["E_E_Avg"])
echo_pred["E_E_Avg"].describe()

# %%
# Get TR Vmax
echo_pred["TR_pk_vel"] = np.sqrt(echo_pred["TV_Pk_Grad"]/4)
echo_pred["TR_pk_vel"].describe()

# %%
# Load predictions
predictions = pd.read_csv(newpath+'012023_echo_AS_preds.csv')

# Left join echo_pred with predictions
echo_pred = echo_pred.merge(predictions, on="acc_num", how="left")

# %%
# Plot interaction between baseline peak AV Vmax and rare of progression
import statsmodels.api as sm
import statsmodels.formula.api as smf

echo_pred["ensemble_perc"] = echo_pred["ensemble"]*100
echo_pred["AV_Pk_Vel_cm_s"] = echo_pred["AV_Pk_Vel_m_s"]*100

formula = 'ensemble_perc ~ AV_Pk_Vel_cm_s + E_E_Avg + LA_Vol_Indexed + TV_Pk_Grad'

model = smf.glm(formula = formula, data=echo_pred)
result = model.fit()
print(result.summary())