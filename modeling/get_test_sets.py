import os
import csv

import pandas as pd
from datetime import datetime

DIR_ROOT = os.path.join("D:\\", "LAA_STROKE", "data", "UW")
DIR_WRITE = os.path.join(DIR_ROOT, "labels", "chart_review")

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


DIR_seg_training = r"D:\LAA_STROKE\data\UW\labels\segmentation_masks\FULL"
DIR_seg_holdout = r"D:\LAA_STROKE\data\UW\holdout_test_set\labels\final"

list_training = [case.split("_")[0] for case in os.listdir(DIR_seg_training)]
list_holdout = [case.split("_")[0] for case in os.listdir(DIR_seg_holdout)]
