import os
import pandas as pd
import numpy as np
from math import ceil

data_dir = '/home/gih5/mounts/nfs_echo_yale/031522_echo_avs_preprocessed'

for cohort in ['cedars', '051823_full_test_2016-2020']:
    np.random.seed(0)

    if cohort == '051823_full_test_2016-2020':
        labels = pd.read_csv(os.path.join(data_dir, cohort + '.csv'))
        study_labels = labels.groupby('acc_num', as_index=False).first()

        prevalences = ['0.015', 'all']

        n_neg = (study_labels['severe_AS'] == 0).sum()
        for prevalence_str in prevalences[:-1]:
            prevalence = float(prevalence_str)
            new_n_pos = ceil(-n_neg * prevalence / (prevalence - 1))  # find num severe AS examples s.t. we have desired prevalence

            neg_acc_nums = study_labels[study_labels['severe_AS'] == 0]['acc_num'].values

            # If possible add more positive samples s.t. prevalence still rounds to desired level
            while (new_n_pos / (new_n_pos + neg_acc_nums.size) - prevalence) < 0.0005 and (new_n_pos / (new_n_pos + neg_acc_nums.size) - prevalence) > 0:
                new_n_pos += 1
            new_n_pos -= 1

            ### 5/11/23: Get 1 random acc num per unique patient, then randomly select from these as below ###
            pos_acc_nums = np.setdiff1d(study_labels.groupby('MRN').sample(n=1, random_state=0)['acc_num'].values, neg_acc_nums)

            new_pos_acc_nums = np.random.choice(pos_acc_nums, size=new_n_pos, replace=False)

            new_acc_nums = np.concatenate([new_pos_acc_nums, neg_acc_nums])

            print(f'Severe AS: {new_n_pos}/{new_acc_nums.size} ({(new_n_pos / new_acc_nums.size)*100:.1f}%)')

            out_df = pd.DataFrame({'acc_num': new_acc_nums})
            out_df.to_csv(f'052123_{cohort}_prevalence-{prevalence_str}_cohort.csv', index=False)
    elif cohort == 'cedars':
        labels = pd.read_csv('/home/gih5/echo-severe-AS/YaleASInference_2-28-23_PseudoID_clean.csv')
        study_labels = labels.groupby(by='study_uid', as_index=False).first()

        # REMOVE LFLG CASES
        study_labels = study_labels[study_labels['paradoxlflg'] == False].reset_index(drop=True)

        acc_num_dict = {'all': study_labels['study_uid'].values}

        print('Origial prevalence:', (study_labels['SevereAS'] == 1).sum() / study_labels.shape[0])

        prevalences = ['0.015', '0.05', '0.1', '0.15', '0.2', 'all']
        n_neg = (study_labels['SevereAS'] == 0).sum()
        for prevalence_str in prevalences[:-1]:
            prevalence = float(prevalence_str)
            new_n_pos = ceil(-n_neg * prevalence / (prevalence - 1))  # find num severe AS examples s.t. we have desired prevalence

            neg_acc_nums = study_labels[study_labels['SevereAS'] == 0]['study_uid'].values

            # If possible add more positive samples s.t. prevalence still rounds to desired level
            while (new_n_pos / (new_n_pos + neg_acc_nums.size) - prevalence) < 0.0005 and (new_n_pos / (new_n_pos + neg_acc_nums.size) - prevalence) > 0:
                new_n_pos += 1
            new_n_pos -= 1

            ### 5/11/23: Get 1 random acc num per unique patient, then randomly select from these as below ###
            pos_acc_nums = np.setdiff1d(study_labels.groupby('patient_id').sample(n=1, random_state=0)['study_uid'].values, neg_acc_nums)

            new_pos_acc_nums = np.random.choice(pos_acc_nums, size=new_n_pos, replace=False)

            new_acc_nums = np.concatenate([new_pos_acc_nums, neg_acc_nums])

            print(f'Severe AS: {new_n_pos}/{new_acc_nums.size} ({(new_n_pos / new_acc_nums.size)*100:.1f}%)')

            out_df = pd.DataFrame({'acc_num': new_acc_nums})
            out_df.to_csv(f'052123_{cohort}_prevalence-{prevalence_str}_cohort.csv', index=False)
