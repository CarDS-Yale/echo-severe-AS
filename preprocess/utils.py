import os
import datetime
import random

import numpy as np
import pandas as pd
import pydicom as dcm
import tqdm

from joblib import Parallel

class ProgressParallel(Parallel):
    """tqdm progress bar for parallel job execution, as per https://stackoverflow.com/a/61900501."""
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm.tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

def val_worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)

def to_datetime(date):
    """Convert yyyymmdd string date representation to datetime object."""
    return datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]))

def calc_age(date, dob):
    """Calculate approximate age in years by computing difference between date and dob."""
    date = to_datetime(date)
    dob = to_datetime(dob)

    diff = (date - dob).total_seconds()

    return diff / (60 * 60 * 24 * 365)

def extract_metadata(file_path, output_dir, verbose=False):
    """Extract patient and scanner data from DICOM metadata."""

    # Dictionary of desired metadata and their standard indices in the DICOM header
    METADATA = {'study_date': (0x0008, 0x0020), 'manufacturer': (0x0008, 0x0070), 'model_name': (0x0008, 0x1090), 'dob': (0x0010, 0x0030),
                'sex': (0x0010, 0x0040), 'size': (0x0010, 0x1020), 'weight': (0x0010, 0x1030)}

    echo_id, file_name = file_path.split('/')[-3], file_path.split('/')[-1].split('.')[0]
    out_file_path = os.path.join(output_dir, echo_id, 'metadata.csv')

    # Read in DICOM
    dataset = dcm.dcmread(file_path, force=True)

    # Extract metadata and place into Pandas data frame
    patient_metadata = [dataset[v].value for _, v in METADATA.items()]
    metadata_df = pd.DataFrame([patient_metadata], columns=METADATA.keys())

    # Calculate age from the study date and patient's date of birth, then remove study date and DOB
    try:
        metadata_df['age'] = calc_age(metadata_df['study_date'].values[0], metadata_df['dob'].values[0])
    except:
        metadata_df['age'] = np.nan
    metadata_df = metadata_df.drop(columns=['study_date', 'dob'])

    metadata_df.to_csv(out_file_path, index=False)
