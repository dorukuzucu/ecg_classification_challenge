import os
import tarfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import io

from get_12ECG_features import get_12ECG_features


# Find unique classes.
def get_classes(input_directory, filenames):
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


input_directory = "data/raw/Training_WFDB/"
output_directory = "data/processed"
"""
for f in os.listdir(input_directory)[:4]:
    g = os.path.join(input_directory, f)
    if not f.lower().startswith('.') and f.lower().endswith('hea') and os.path.isfile(g):
        recording, header = load_challenge_data(g)
"""


header_files = []
for f in os.listdir(input_directory):
    g = os.path.join(input_directory, f)
    if not f.lower().startswith('.') and f.lower().endswith('hea') and os.path.isfile(g):
        header_files.append(g)

classes = get_classes(input_directory, header_files)
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
            labels_act = np.zeros(num_classes)
            arrs = l.strip().split(' ')
            for arr in arrs[1].split(','):
                class_index = classes.index(arr.rstrip()) # Only use first positive index
                labels_act[class_index] = 1
    labels.append(labels_act)

# "age", "sex", "mean_RR", "mean_Peaks", "median_RR", "median_Peaks", "std_RR", "std_Peaks", "var_RR", "var_Peaks", "skew_RR", "skew_Peaks", "kurt_RR", "kurt_Peaks"
features = np.array(features)
labels = np.array(labels)

# write features
fields = [
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
schema_feature = pa.schema(fields)
table_feature = pa.Table.from_arrays(
    [
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
    ], 
    schema=schema_feature,
    )

# write labels
fields_labels = [
    ('label_1', pa.int8()),
    ('label_2', pa.int8()),
    ('label_3', pa.int8()),
    ('label_4', pa.int8()),
    ('label_5', pa.int8()),
    ('label_6', pa.int8()),

]
schema_labels = pa.schema(fields_labels)
table_labels = pa.Table.from_arrays(
    [
    pa.array(labels[:, 0]), 
    pa.array(labels[:, 1]), 
    pa.array(labels[:, 2]), 
    pa.array(labels[:, 3]), 
    pa.array(labels[:, 4]), 
    pa.array(labels[:, 5]), 
    ], 
    schema=schema_labels,
    )


output_path_feature = os.path.join(output_directory, "preprocess.parquet")
pq.write_table(table_feature, where=output_path_feature)

output_path_labels = os.path.join(output_directory, "labels.parquet")
pq.write_table(table_labels, where=output_path_labels)