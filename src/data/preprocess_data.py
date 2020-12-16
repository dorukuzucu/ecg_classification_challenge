import glob
import os
from pathlib import Path
import tarfile

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
from scipy import io

from get_12ECG_features import get_12ECG_features


# Find unique classes.
def get_classes(input_directory, filenames, static=True):
    if static:
        class_path = os.path.join(Path(input_directory).parents[1], "dx_mapping_scored.csv")
        class_matrix = pcsv.read_csv(class_path).to_pandas()
        classes = class_matrix["SNOMED CT Code"]
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


input_directory = "data/raw"
output_directory = "data/processed"

folders = ["training", "validation", "test"]

for folder in folders:
    input_directory_path = os.path.join(input_directory, folder)
    output_directory_path = os.path.join(output_directory, folder)
    # make sure output path exist create if not
    Path(output_directory_path).mkdir(exist_ok=True)

    # loop in raw data folders
    for input_folder in glob.glob(f"{input_directory_path}/*"):
        folder_name = input_folder.split("/")[-1]
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

        for i in range(num_files):
            recording = recordings[i]
            header = headers[i]

            tmp = get_12ECG_features(recording, header)
            features.append(tmp)

            for l in header:
                if l.startswith('#Dx:'):
                    labels_act = np.zeros(num_classes+1)
                    arrs = l.strip().split(' ')
                    for arr in arrs[1].split(','):
                        # if label not in our labels
                        if arr.strip() not in classes:
                            labels_act[-1] = 1
                        else:
                            class_index = classes.index(arr.rstrip()) # Only use first positive index
                            labels_act[class_index] = 1
            labels.append(labels_act)

        # "age", "sex", "mean_RR", "mean_Peaks", "median_RR", "median_Peaks", "std_RR", "std_Peaks", "var_RR", "var_Peaks", "skew_RR", "skew_Peaks", "kurt_RR", "kurt_Peaks"
        features = np.array(features)
        labels = np.array(labels)

        # filter labels which not in our labels
        other_class_mask = labels[:,-1] != 1
        features = features[other_class_mask]
        labels = labels[other_class_mask]


        # features
        # since number of feautes has not been determined we create them statically
        fields_features = [
            ('age', pa.float32()),
            ('sex', pa.float32()),
            ('mean_RR', pa.float32()),
            ('mean_Peaks', pa.float32()),
            ('median_RR', pa.float32()),
            ('median_Peaks', pa.float32()),
            ('std_RR', pa.float32()),
            ('std_Peaks', pa.float32()),
            ('var_RR', pa.float32()),
            ('var_Peaks', pa.float32()),
            ('skew_RR', pa.float32()),
            ('skew_Peaks', pa.float32()),
            ('kurt_RR', pa.float32()),
            ('kurt_Peaks', pa.float32()),
        ]
        table_features_arrays = [
            pa.array(features[:, 0]), 
            pa.array(features[:, 1]), 
            pa.array(features[:, 2]), 
            pa.array(features[:, 3]), 
            pa.array(features[:, 4]), 
            pa.array(features[:, 5]), 
            pa.array(features[:, 6]), 
            pa.array(features[:, 7]), 
            pa.array(features[:, 8]), 
            pa.array(features[:, 9]), 
            pa.array(features[:, 10]), 
            pa.array(features[:, 11]), 
            pa.array(features[:, 12]), 
            pa.array(features[:, 13]),
        ]

        # labels
        # with loop we get label fields and values dynamically
        fields_labels = []
        table_labels_arrays = []
        for l in range(num_classes):
            fields_labels += (f'label_{l+1}', pa.int8()),
            table_labels_arrays += pa.array(labels[:, l]),

        # concat features and labels
        fields = fields_features + fields_labels
        table_arrays = table_features_arrays + table_labels_arrays

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
