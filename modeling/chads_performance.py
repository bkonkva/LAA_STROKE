import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
)
from matplotlib import pyplot as plt

DIR_ROOT = os.path.join("D:\\", "LAA_STROKE", "data", "UW")
DIR_WRITE = os.path.join(DIR_ROOT, "labels", "chart_review")

path_chads_score = os.path.join(DIR_WRITE, "chads_score_nostroke.csv")
path_stroke_history = os.path.join(DIR_WRITE, "stroke_any.csv")

df_chads = pd.read_csv(path_chads_score)
df_stroke = pd.read_csv(path_stroke_history)
chads = list(df_chads["cha2ds2-vasc score"])
stroke = list(df_stroke["stroke"])


def return_tpr(chads_threshold):
    y_pred = [1 if risk > chads_threshold else 0 for risk in chads]
    return recall_score(stroke, y_pred)


def return_fpr(chads_threshold):
    y_pred = [1 if risk > chads_threshold else 0 for risk in chads]
    return 1 - recall_score(stroke, y_pred, pos_label=0)


def main():

    TPR = [return_tpr(thresh) for thresh in range(0, np.max(chads))]
    FPR = [return_fpr(thresh) for thresh in range(0, np.max(chads))]

    # accuracy = accuracy_score(y, y_pred)
    # sensitivity = recall_score(y, y_pred)  # recall is the same as sensitivity
    # specificity = recall_score(y, y_pred, pos_label=0)
    # confusion = confusion_matrix(y, y_pred)

    # print(f"Accuracy: {accuracy}")
    # print(f"Sensitivity: {sensitivity}")
    # print(f"Specificity: {specificity}")
    # print(f"Confusion Matrix:\n{confusion}")

    # TPR_chads = recall_score(y, y_pred)
    # FPR_chads = 1- recall_score(y, y_pred, pos_label=0)

    plt.plot(
        FPR, TPR, marker="o", markersize=10, label="CHADS Threshold", linestyle="none"
    )

    print("done")


if __name__ == "__main__":
    main()
