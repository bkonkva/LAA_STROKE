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


def return_redcap(
    patient: int,
    chads_key: str,
    positive_output: str,
    chads_score: int,
    df: pd.DataFrame = df_redcap,
) -> int:
    if df.loc[df["Record ID"] == patient, chads_key].iloc[0] == positive_output:
        return chads_score
    else:
        return 0


def return_age(patient: int, df: pd.DataFrame = df_mapped) -> int:
    if pd.isna(df.loc[df["record_id"] == patient, "age"].iloc[0]):
        return 0
    age = int(df.loc[df["record_id"] == patient, "age"].iloc[0].strip("Y"))
    if age >= 75:
        return 2
    elif age > 65 and age <= 74:
        return 1
    return 0


def return_imaging_date(patient, df: pd.DataFrame = df_mapped):
    return df.loc[df["record_id"] == patient, "scan_date"].iloc[0]


def stroke_before_imaging(
    stroke_timing: str, patient: int, df: pd.DataFrame = df_redcap
) -> bool:
    if stroke_timing == "Prior to imaging but exact date is unknown":
        return True
    elif stroke_timing == "After imaging but exact date is unknown":
        return False
    elif stroke_timing == "Unknown":
        return False  # not sure what to do with these
    elif stroke_timing == "Exact/approximate date is known":
        stroke_date = datetime.strptime(
            df.loc[df["Record ID"] == patient, "Date of stroke"].iloc[0], r"%Y-%m-%d"
        )
        imaging_date = datetime.strptime(return_imaging_date(patient), r"%m/%d/%Y")
        if imaging_date > stroke_date:
            return True
        else:
            return False


def return_prior_stroke(patient: int, df: pd.DataFrame = df_redcap) -> int:
    if df.loc[df["Record ID"] == patient, "History of stroke/TIA/TE"].iloc[0] == "Yes":
        stroke_timing = df.loc[df["Record ID"] == patient, "Timing of stroke"].iloc[0]
        if stroke_before_imaging(stroke_timing, patient):
            return 2
    return 0


def return_stroke(patient: int, df: pd.DataFrame = df_redcap) -> bool:
    if df.loc[df["Record ID"] == patient, "History of stroke/TIA/TE"].iloc[0] == "Yes":
        stroke_timing = df.loc[df["Record ID"] == patient, "Timing of stroke"].iloc[0]
        if stroke_before_imaging(stroke_timing, patient):
            return False
        else:
            return True
    return False


def return_stroke_any(patient: int, df: pd.DataFrame = df_redcap) -> bool:
    if df.loc[df["Record ID"] == patient, "History of stroke/TIA/TE"].iloc[0] == "Yes":
        return True
    return False


def return_chads(patient):
    dict_patient = {}
    dict_patient["sex"] = return_redcap(patient, "Sex", "Female", 1)
    dict_patient["chf"] = return_redcap(patient, "CHF", "Yes", 1)
    dict_patient["hypertension"] = return_redcap(patient, "Hypertension", "Yes", 1)
    dict_patient["diabetes"] = return_redcap(patient, "Diabetes", "Yes", 1)
    dict_patient["vascular disease"] = return_redcap(
        patient, "Vascular disease", "Yes", 1
    )
    dict_patient["stroke history"] = return_prior_stroke(patient)
    dict_patient["age"] = return_age(patient)
    return dict_patient


def dict_to_csv(dictionary: dict, headers: list, write_path: str) -> None:
    with open(write_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for key, value in dictionary.items():
            writer.writerow([key, value])


def return_afib(patient_id, df: pd.DataFrame = df_redcap) -> bool:
    afib = df.loc[
        df["Record ID"] == patient_id, "Atrial Fibrillation or Atrial Flutter"
    ].iloc[0]
    if afib == "Yes":
        return True
    elif afib == "No":
        return False
    else:
        return ""


def return_demo(
    patient_id, df: pd.DataFrame = df_redcap, df2: pd.DataFrame = df_mapped
) -> dict:
    demographics = {}
    demographics["sex "] = df.loc[df["Record ID"] == patient_id, "Sex"].iloc[0]
    demographics["race"] = df.loc[
        df["Record ID"] == patient_id, "Racial and Ethnic Category"
    ].iloc[0]
    age = df2.loc[df2["record_id"] == patient_id, "age"].iloc[0]
    if type(age) == str:
        age = age.lower().split("y")[0]
    demographics["age "] = age
    return demographics


def main():
    DICT_ACC = {}
    DICT_STROKE = {}
    DICT_CHADS = {}
    DICT_SCORE = {}
    DICT_STROKE_NEW = {}
    DICT_SCORE_NOSTROKE = {}
    DICT_AFIB = {}
    DICT_DEMO = {}
    for index, row in df_redcap.iterrows():
        record_id = row["Record ID"]
        try:
            rand_acc = return_acc(record_id)
        except:
            print(f"ERROR: {record_id}")
        if not (df_redcap["Record ID"] == record_id).any():
            print(f"{record_id} missing from redcap data")
        elif not (df_mapped["record_id"] == record_id).any():
            print(f"{record_id} missing from mapped data")
        else:
            DICT_ACC[rand_acc] = record_id
            DICT_AFIB[rand_acc] = return_afib(record_id)
            DICT_DEMO[rand_acc] = return_demo(record_id)
            DICT_CHADS[rand_acc] = return_chads(record_id)
            DICT_SCORE[rand_acc] = sum(DICT_CHADS[rand_acc].values())
            DICT_SCORE_NOSTROKE[rand_acc] = (
                sum(DICT_CHADS[rand_acc].values())
                - DICT_CHADS[rand_acc]["stroke history"]
            )
            DICT_STROKE[rand_acc] = return_stroke(record_id)
            DICT_STROKE_NEW[rand_acc] = return_stroke_any(record_id)
    dict_to_csv(
        DICT_ACC,
        headers=["accession", "patiend id"],
        write_path=os.path.join(DIR_WRITE, "accession_key.csv"),
    )
    dict_to_csv(
        DICT_STROKE,
        headers=["patient", "stroke"],
        write_path=os.path.join(DIR_WRITE, "stroke.csv"),
    )
    dict_to_csv(
        DICT_STROKE_NEW,
        headers=["patient", "stroke"],
        write_path=os.path.join(DIR_WRITE, "stroke_any.csv"),
    )
    dict_to_csv(
        DICT_SCORE,
        headers=["patient", "cha2ds2-vasc score"],
        write_path=os.path.join(DIR_WRITE, "chads_score.csv"),
    )
    dict_to_csv(
        DICT_SCORE_NOSTROKE,
        headers=["patient", "cha2ds2-vasc score"],
        write_path=os.path.join(DIR_WRITE, "chads_score_nostroke.csv"),
    )
    dict_to_csv(
        DICT_CHADS,
        headers=["patient", "cha2ds2-vasc features"],
        write_path=os.path.join(DIR_WRITE, "chads_features.csv"),
    )
    dict_to_csv(
        DICT_AFIB,
        headers=["patient", "afib"],
        write_path=os.path.join(DIR_WRITE, "atrial_fibrillation.csv"),
    )
    dict_to_csv(
        DICT_DEMO,
        headers=["patient", "sex", "race"],
        write_path=os.path.join(DIR_WRITE, "demographics.csv"),
    )


if __name__ == "__main__":
    main()

# save CHADS2
# import csv

# # Your simple key:value dictionary

# # The name of the CSV file where you want to save the dictionary
# filename = "D:\LAA_STROKE\data\LAA_TEST\mode_stuff2.csv"

# Open the file in write mode

# headers = ["patient", "stroke"]
# write_path = os.path.join(
#     DIR_ROOT,
# )


# # Specify the filename
# filename = 'D:\LAA_STROKE\data\LAA_TEST\chads_scoring.csv'

# # Open the file in write mode
# with open(filename, 'w', newline='') as csvfile:
#     # Create a csv writer object
#     csvwriter = csv.writer(csvfile)

#     # Write the header
#     csvwriter.writerow(['patient', 'chads', 'stroke'])

#     # Iterate through the keys of the first dictionary
#     for key in DICT_SCORE:
#         # Write the key and the corresponding values from both dictionaries
#         csvwriter.writerow([key, DICT_SCORE[key], DICT_STROKE[key]])
