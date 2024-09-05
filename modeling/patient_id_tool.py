import os
import csv

import pandas as pd
from datetime import datetime

DIR_ROOT = os.path.join("D:\\", "LAA_STROKE", "data", "UW")
DIR_WRITE = os.path.join(DIR_ROOT, "labels", "chart_review")

DIR_WRITE_COUNTS = r"D:\LAA_STROKE\data\UW\labels"

path_chart_review = os.path.join(
    DIR_ROOT, "raw", "LAAAndStroke_DATA_LABELS_2024-01-29_1514.csv"
)
path_mapped = os.path.join(DIR_ROOT, "raw", "uw_mapped.csv")

df_redcap = pd.read_csv(path_chart_review)
df_mapped = pd.read_csv(path_mapped)


def return_acc(record_id: str) -> str:
    return str(
        int(
            df_mapped.loc[df_mapped["record_id"] == record_id, "rand_accession"].values[
                0
            ]
        )
    )


def return_record_id(accession_num: str) -> str:
    return str(
        int(
            df_mapped.loc[
                df_mapped["rand_accession"] == float(accession_num), "record_id"
            ].values[0]
        )
    )


DIR_seg_training = r"D:\LAA_STROKE\data\UW\labels\segmentation_masks\FULL"
DIR_seg_holdout = r"D:\LAA_STROKE\data\UW\holdout_test_set\labels\final"

list_training = [case.split("_")[0] for case in os.listdir(DIR_seg_training)]
list_training = [[return_record_id(case)] for case in list_training]
list_holdout = [case.split("_")[0] for case in os.listdir(DIR_seg_holdout)]
list_holdout = [[return_record_id(case)] for case in list_holdout]

with open(os.path.join(DIR_WRITE_COUNTS, "seg_training.csv"), "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(list_training)

with open(os.path.join(DIR_WRITE_COUNTS, "seg_holdout.csv"), "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(list_holdout)

print(list_training)
