import glob
import os
from pathlib import Path
import tarfile

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
from scipy import io
import pandas as pd
from get_12ECG_features import get_12ECG_features


# Find unique classes.
def get_classes(input_directory, filenames, static=True):
    if static:
        class_path = os.path.join(Path(input_directory).parents[1], "dx_mapping_scored.csv")
        #class_matrix = pcsv.read_csv(class_path).to_pandas()
        class_matrix = pd.read_csv(class_path)
        classes = class_matrix["SNOMED CT Code"].astype(str)
        return list(set(classes))
    else:
        classes = set()
        for filename in filenames:
            with open(filename, 'r') as f:
                for l in f:
                    if l.startswith('#Dx'):
                        tmp = l.split(': ')[1].split(',')
                        for c in tmp:
                            classes.add(c.strip())
        return sorted(classes)


# Load challenge data.
def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = io.loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header

project_path = Path(__file__).parents[2]
input_directory = project_path / "data/raw"
output_directory = project_path / "data/processed"

folders = ["train", "validation", "test"]

for folder in folders:
    input_directory_path = os.path.join(input_directory, folder)
    output_directory_path = os.path.join(output_directory, folder)
    # make sure output path exist create if not
    Path(output_directory_path).mkdir(exist_ok=True)

    # loop in raw data folders
    for input_folder in glob.glob(f"{input_directory_path}/*"):
        folder_name = input_folder.split(os.path.sep)[-1]
        print(f"processing {folder_name} data..")

        header_files = []
        for f in os.listdir(input_folder):
            g = os.path.join(input_folder, f)
            if not f.lower().startswith('.') and f.lower().endswith('hea') and os.path.isfile(g):
                header_files.append(g)

        classes = get_classes(input_folder, header_files)
        num_classes = len(classes)
        num_files = len(header_files)
        recordings = list()
        headers = list()

        for i in range(num_files):
            recording, header = load_challenge_data(header_files[i])
            recordings.append(recording)
            headers.append(header)

        features = list()
        labels = list()
        label = list()

        for i in range(num_files):
            recording = recordings[i]
            header = headers[i]

            tmp, num_leads = get_12ECG_features(recording, header)
            features.append(tmp)
            
            
            for l in header:
                if l.startswith('#Dx:'):
                    label = list()
                    arrs = l.strip().split(' ')
                    for arr in arrs[1].split(','):
                        # if label not in our labelss
                        if arr.rstrip() not in classes:
                            label = -1
                            continue
                        else:
                            label = classes.index(arr.rstrip())
                            break # Only use first positive index
            labels.append(label)

        features = np.array(features)
        labels = np.array(labels)

        # filter labels which not in our labels
        other_class_mask = labels != -1
        features = features[other_class_mask]
        labels = labels[other_class_mask]

        feature_list = ["age","sex","mean_RR", "mean_Peaks", "median_RR", "median_Peaks", "std_RR", "std_Peaks", "var_RR", "var_Peaks", "skew_RR", "skew_Peaks", "kurt_RR", "kurt_Peaks"]
        # with loop we get fields and values dynamically
        fields = [
            ('label', pa.int64()),
        ]
        table_arrays = [
            pa.array(labels),
        ]

        ix = 0
        for l in range(num_leads):
            for f in feature_list:
                fields += (f'lead{l+1}_{f}', pa.float32()),
                table_arrays += pa.array(features[:, ix]),
                ix += 1

        # create parquet objects
        schema = pa.schema(fields)
        table = pa.Table.from_arrays(
            table_arrays,
            schema=schema,
            )

        print(f"writing {folder_name} data..")
        # write concated data to parquet
        output_path_labels = os.path.join(output_directory_path, f"processed_{folder_name}.parquet")
        pq.write_table(table, where=output_path_labels)
