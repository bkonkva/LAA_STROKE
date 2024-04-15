# TASKS
"""
1. CHECK SHAPEWORKS EXPORT NAMING - Amy (Nicole)
2. Shape feature data exploration - Nicole
# reference UTAH SSM paper - for details on logistic regression. 
3. EXPLORING MODEL PERFORMANCE
    -are we overfitting? 
    -do we need some additional regularization?
    -different model types?
    -feature importance...
    -normalizing etc.
"""

from sklearn.metrics import roc_curve, auc, accuracy_score
import numpy as np
import pandas as pd
import os
import csv
import json
import ast
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.feature_selection import RFE, VarianceThreshold, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb

# Constants
DIR_ROOT = "D:\\LAA_STROKE\\data\\UW"
DIR_WRITE = os.path.join(DIR_ROOT, "labels", "chart_review")
DIR_LABELS = os.path.join(DIR_ROOT, "labels")
DIR_METRICS = "D:\\LAA_STROKE\\models\\metrics"
MODE_LIMIT = 10  # =95% variance # total number of shape features we want to include
RANDOM_STATE = 12
EPOCHS = 1000
RFE_FEATURES = 11
FILTER_AFIB = False
FILTER_NOAFIB = False

# TEST SET
DICT_RACE_COUNT = {
    # race: (stroke, nostroke)
    "White": [20, 20],
    "Black or African American": [4, 4],
    "Hispanic or Latino": [4, 4],
    "Asian": [4, 4],
    "American Indian or Alaska Native": [4, 4],
}


def check_dict(df: pd.DataFrame, patient_header: str, patient: str, header: str):
    return df.loc[df[patient_header] == patient, header].iloc[0]


def check_race(patient, df_demo, df_stroke):
    for race, count in DICT_RACE_COUNT.items():
        demo = ast.literal_eval(
            check_dict(df_demo, "patient", patient, "sex").replace("nan", "None")
        )
        if not race == "White":
            print("step")
        if check_dict(df_stroke, "patient", patient, "stroke") == True:
            if demo["race"] == race and count[0] > 0:
                DICT_RACE_COUNT[race][0] -= 1
                return demo["race"]
        elif check_dict(df_stroke, "patient", patient, "stroke") == False:
            if demo["race"] == race and count[1] > 0:
                DICT_RACE_COUNT[race][1] -= 1
                return demo["race"]
    return False


def build_test_set(
    df_demo: pd.DataFrame, df_stroke: pd.DataFrame, list_patients: list
) -> dict:
    test_set = {}
    for patient in list_patients:
        race = check_race(patient, df_demo, df_stroke)
        if race:
            test_set[patient] = race
    return test_set


# Load and prepare data
def load_data():
    component_scores_path = os.path.join(
        DIR_LABELS, "statistical_shape_modeling", "component_scores", "02-24.csv"
    )
    project_json_path = os.path.join(
        DIR_LABELS, "statistical_shape_modeling", "project_jsons", "02-24.swproj"
    )
    demographics_path = os.path.join(DIR_LABELS, "chart_review", "demographics.csv")
    stroke_path = os.path.join(DIR_LABELS, "chart_review", "stroke_any.csv")
    chads_score_path = os.path.join(DIR_WRITE, "chads_score_nostroke.csv")
    chads_features_path = os.path.join(DIR_WRITE, "chads_features.csv")
    afib_path = os.path.join(DIR_WRITE, "atrial_fibrillation.csv")
    acc_path = os.path.join(DIR_WRITE, "accession_key.csv")

    component_scores = (
        pd.read_csv(component_scores_path).drop("Group", axis=1).iloc[:, :MODE_LIMIT]
    )
    with open(project_json_path) as file:
        file_names_ssm = json.load(file)
    IDs = [
        int(os.path.split(item["shape_1"])[-1].split(".")[0])
        for item in file_names_ssm["data"]
    ]
    df_demo = pd.read_csv(demographics_path)
    stroke = pd.read_csv(stroke_path)
    df_chads = pd.read_csv(chads_score_path)
    df_chads_features = pd.read_csv(chads_features_path)
    df_afib = pd.read_csv(afib_path)
    df_acc = pd.read_csv(acc_path)

    component_scores["patient"] = IDs
    df = pd.merge(component_scores, stroke, on="patient", how="inner")
    df_chads_features_new = prepare_chads_features(df_chads_features)
    return df, df_chads, df_chads_features_new, df_afib, df_demo, stroke, df_acc


def prepare_chads_features(df_chads_features):
    df_new = pd.DataFrame(
        columns=[
            "patient",
            "sex",
            "chf",
            "hypertension",
            "diabetes",
            "vascular_disease",
            "age",
        ]
    )
    for _, row in df_chads_features.iterrows():
        dict_chads = ast.literal_eval(row["cha2ds2-vasc features"])
        dict_chads.pop("stroke history", None)
        df_new.loc[len(df_new)] = [row["patient"]] + list(dict_chads.values())
    return df_new


# Model training and evaluation
def evaluate_model(
    model,
    X,
    y,
    cv,
    oversample=True,
    plot_label="",
    color="",
    linestyle="-",
    test_set=False,
):
    if oversample:
        ros = RandomOverSampler(random_state=RANDOM_STATE)
        pipeline = make_pipeline_imb(ros, model)
    else:
        pipeline = make_pipeline(model)
    tprs, aucs, mean_fpr = [], [], np.linspace(0, 1, 100)

    for train, test in cv.split(X, y):
        pipeline.fit(X.iloc[train], y.iloc[train])
        if not test_set:
            probas_ = pipeline.predict_proba(X.iloc[test])
            fpr, tpr, _ = roc_curve(y.iloc[test], probas_[:, 1])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
        else:
            probas_ = pipeline.predict_proba(test_set[0])
            fpr, tpr, _ = roc_curve(test_set[1], probas_[:, 1])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

    plot_roc_curve(mean_fpr, tprs, aucs, plot_label, color, linestyle)


# Function to evaluate CHADS score as thresholds
def evaluate_chads_thresholds(chads_scores, y_true, plot_label="", color=""):
    tprs, fprs = [], []
    thresholds = range(0, 8)  # CHADS scores range from 0 to 7

    for threshold in thresholds:
        # Predict positive if CHADS score is above the threshold, negative otherwise
        y_pred = [1 if score > threshold else 0 for score in chads_scores]
        y_pred = pd.core.series.Series(y_pred)

        # Calculate TPR and FPR
        TP = sum(
            [a and b for a, b in zip(y_pred, y_true)]
        )  # sum((y_pred == 1) & (y_true == 1))
        FP = sum(
            [a and not b for a, b in zip(y_pred, y_true)]
        )  # sum((y_pred == 1) & (y_true == 0))
        FN = sum(
            [not a and b for a, b in zip(y_pred, y_true)]
        )  # sum((y_pred == 0) & (y_true == 1))
        TN = sum(
            [not a and not b for a, b in zip(y_pred, y_true)]
        )  # sum((y_pred == 0) & (y_true == 0))

        TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0

        tprs.append(TPR)
        fprs.append(FPR)

    # Plot the ROC curve based on CHADS thresholds
    auc_chads = auc(fprs, tprs)
    plt.plot(
        fprs,
        tprs,
        linestyle="-",
        color=color,
        label=r"%s (AUC = %0.2f)" % (plot_label, auc_chads),
    )


def plot_roc_curve(mean_fpr, tprs, aucs, label, color, linestyle="-"):
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(
        mean_fpr,
        mean_tpr,
        color=color,
        label=r"%s (AUC = %0.2f $\pm$ %0.2f)" % (label, mean_auc, std_auc),
        lw=2,
        alpha=0.8,
        linestyle=linestyle,
    )


# def scale_set(set):
#     scaler = StandardScaler()
#     scaler.fit(set)
#     return pd.core.frame.DataFrame(scaler.transform(set))


def scale_set(set):
    return set


def preprocess_group_data(df, group_column, group_value):
    group_df = df[df[group_column] == group_value]
    X_group = group_df.drop(columns=["patient", "stroke", group_column])
    y_group = group_df["stroke"]
    scaler = StandardScaler()
    X_group_scaled = scaler.fit_transform(X_group)
    return X_group_scaled, y_group


def run_inference(model, X, y, label, color, linestyle):
    tprs, aucs, mean_fpr = [], [], np.linspace(0, 1, 100)
    probas_ = model.predict_proba(X)
    fpr, tpr, _ = roc_curve(y, probas_[:, 1])
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plot_roc_curve(mean_fpr, tprs, aucs, label, color, linestyle)
    return probas_


def dict_to_csv(dictionary: dict, headers: list, write_path: str) -> None:
    with open(write_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for key, value in dictionary.items():
            writer.writerow([key, value])


def save_probas(df_acc, df_test_set, probas, write_name):
    dict_probas = {}
    count = 0
    for idx, patient_row in df_test_set.iterrows():
        id = df_acc.loc[
            df_acc["accession"] == patient_row["patient"], "patiend id"
        ].values[0]
        dict_probas[id] = probas[count][0]
        count += 1
    dict_to_csv(
        dict_probas,
        headers=["patient id", "prediction"],
        write_path=os.path.join(DIR_WRITE, write_name),
    )


# Main function with extended models
def main():

    df, df_chads, df_chads_features_new, df_afib, df_demo, stroke, df_acc = load_data()

    df_save_features = pd.merge(
        df, df_acc, left_on="patient", right_on="accession", how="inner"
    )
    df_save_features = df_save_features.drop(["patient", "stroke", "accession"], axis=1)
    df_save_features.to_csv(
        os.path.join(DIR_WRITE, "patient_features.csv"), index=False
    )

    if FILTER_AFIB:
        afib_group = df_afib[df_afib["afib"] == True]["patient"]
        df = df[~df["patient"].isin(afib_group)]
    if FILTER_NOAFIB:
        noafib_group = df_afib[df_afib["afib"] == False]["patient"]
        df = df[~df["patient"].isin(noafib_group)]

    holdout_test_set = build_test_set(df_demo, stroke, list(df["patient"]))

    df_train = df[~df["patient"].isin(list(holdout_test_set.keys()))]

    list_white = [key for key, value in holdout_test_set.items() if value == "White"]
    list_black = [
        key
        for key, value in holdout_test_set.items()
        if value == "Black or African American"
    ]
    list_hisp = [
        key for key, value in holdout_test_set.items() if value == "Hispanic or Latino"
    ]
    list_asian = [key for key, value in holdout_test_set.items() if value == "Asian"]
    list_amind = [
        key
        for key, value in holdout_test_set.items()
        if value == "American Indian or Alaska Native"
    ]

    X = df_train.drop(["patient", "stroke"], axis=1)
    y = df_train["stroke"]

    df_test_white = df[df["patient"].isin(list_white)]
    X_white = scale_set(df_test_white.drop(["patient", "stroke"], axis=1))
    y_white = df_test_white["stroke"]

    list_nonwhite = list_black + list_hisp + list_asian + list_amind
    df_test_nonwhite = df[df["patient"].isin(list_nonwhite)]
    X_nonwhite = scale_set(df_test_nonwhite.drop(["patient", "stroke"], axis=1))
    y_nonwhite = df_test_nonwhite["stroke"]

    df_test_black = df[df["patient"].isin(list_black)]
    X_black = scale_set(df_test_black.drop(["patient", "stroke"], axis=1))
    y_black = df_test_black["stroke"]
    df_test_hisp = df[df["patient"].isin(list_hisp)]
    X_hisp = scale_set(df_test_hisp.drop(["patient", "stroke"], axis=1))
    y_hisp = df_test_hisp["stroke"]
    df_test_asian = df[df["patient"].isin(list_asian)]
    X_asian = scale_set(df_test_asian.drop(["patient", "stroke"], axis=1))
    y_asian = df_test_asian["stroke"]
    df_test_amind = df[df["patient"].isin(list_amind)]
    X_amind = scale_set(df_test_amind.drop(["patient", "stroke"], axis=1))
    y_amind = df_test_amind["stroke"]

    X = scale_set(X)

    # Logistic Regression with SSM features only
    log_reg = LogisticRegression(
        max_iter=EPOCHS,
        random_state=RANDOM_STATE,
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0,
    )
    rfe = RFE(estimator=log_reg, n_features_to_select=RFE_FEATURES)
    # rfe = SequentialFeatureSelector(
    #     estimator=log_reg, n_features_to_select=RFE_FEATURES, direction="forward"
    # )
    cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)

    # # train full
    # ros = RandomOverSampler(random_state=RANDOM_STATE)
    # pipeline = make_pipeline_imb(ros, rfe)
    # pipeline.fit(X, y)

    # run_inference(
    #     pipeline, X_white, y_white, label="log reg: white", color="blue", linestyle=":"
    # )
    # run_inference(
    #     pipeline,
    #     X_nonwhite,
    #     y_nonwhite,
    #     label="log reg: non-white",
    #     color="green",
    #     linestyle=":",
    # )
    # run_inference(pipeline, X_black, y_black, label="black", color="red", linestyle="-")
    # run_inference(pipeline, X_hisp, y_hisp, label="hisp", color="black", linestyle="-")
    # run_inference(
    #     pipeline, X_asian, y_asian, label="asian", color="orange", linestyle="-"
    # )
    # run_inference(
    #     pipeline, X_amind, y_amind, label="amind", color="green", linestyle="-"
    # )

    # evaluate_model(rfe, X, y, cv, plot_label="LAA Shape Alone", color="lightgreen")
    # evaluate_model(
    #     rfe,
    #     X,
    #     y,
    #     cv,
    #     plot_label="white",
    #     color="red",
    #     linestyle=":",
    #     test_set=[X_white, y_white],
    # )
    # evaluate_model(
    #     rfe,
    #     X,
    #     y,
    #     cv,
    #     plot_label="hispanic",
    #     color="green",
    #     linestyle=":",
    #     test_set=[X_hisp, y_hisp],
    # )
    # evaluate_model(
    #     rfe,
    #     X,
    #     y,
    #     cv,
    #     plot_label="black",
    #     color="blue",
    #     linestyle=":",
    #     test_set=[X_black, y_black],
    # )
    # evaluate_model(
    #     rfe,
    #     X,
    #     y,
    #     cv,
    #     plot_label="asian",
    #     color="cyan",
    #     linestyle=":",
    #     test_set=[X_asian, y_asian],
    # )
    # evaluate_model(
    #     rfe,
    #     X,
    #     y,
    #     cv,
    #     plot_label="amind",
    #     color="yellow",
    #     linestyle=":",
    #     test_set=[X_amind, y_amind],
    # )

    # MLP with SSM features only
    # mlp = MLPClassifier(
    #     hidden_layer_sizes=(100,),
    #     activation="relu",
    #     solver="adam",
    #     max_iter=EPOCHS,
    #     random_state=RANDOM_STATE,
    # )
    # cv_mlp = StratifiedKFold(n_splits=5)
    # evaluate_model(
    #     mlp, X, y, cv_mlp, oversample=False, plot_label="MLP (SSM)", color="c"
    # )

    # Preparing extended feature set with CHADS
    df_chads_score = pd.DataFrame(
        {
            "patient": df_chads_features_new["patient"],
            "score": df_chads_features_new.iloc[:, 1:].sum(axis=1),
        }
    )
    df_extended = pd.merge(df_train, df_chads_score, on="patient", how="inner").drop(
        "patient", axis=1
    )
    X_extended = df_extended.drop("stroke", axis=1)
    scaler = StandardScaler()
    scaler.fit(X_extended)
    X_extended = pd.core.frame.DataFrame(scaler.transform(X_extended))

    # X_chads = df_extended.iloc[:, -6:]

    # # Logistic Regression with CHADS features only
    log_reg = LogisticRegression(
        max_iter=EPOCHS, random_state=RANDOM_STATE, penalty="l2"
    )
    rfe = RFE(estimator=log_reg, n_features_to_select=RFE_FEATURES)
    # cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    # evaluate_model(
    #     rfe, X_chads, y, cv, plot_label="CHADS-VASc Score alone", color="k"
    # )

    # Logistic Regression with SSM + CHADS features
    # evaluate_model(
    #     rfe,
    #     X_extended,
    #     y,
    #     cv,
    #     plot_label="CHADS-VASc Score plus LAA shape",
    #     color="b",
    # )
    # train full
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    pipeline = make_pipeline_imb(ros, rfe)
    pipeline.fit(X, y)

    probas_white = run_inference(
        pipeline, X_white, y_white, label="log reg: white", color="blue", linestyle=":"
    )
    save_probas(df_acc, df_test_white, probas_white, "probas_white.csv")

    probas_nonwhite = run_inference(
        pipeline,
        X_nonwhite,
        y_nonwhite,
        label="log reg: non-white",
        color="green",
        linestyle=":",
    )

    save_probas(df_acc, df_test_nonwhite, probas_nonwhite, "probas_nonwhite.csv")

    # MLP with SSM + CHADS features
    # evaluate_model(
    #     mlp,
    #     X_extended,
    #     y,
    #     cv_mlp,
    #     oversample=False,
    #     plot_label="MLP (SSM + CHADS)",
    #     color="m",
    # )

    # Evaluate CHADS score thresholds
    chads_scores = df_chads["cha2ds2-vasc score"].values
    chads_white = df_chads[df_chads["patient"].isin(list_white)][
        "cha2ds2-vasc score"
    ].values
    chads_nonwhite = df_chads[df_chads["patient"].isin(list_nonwhite)][
        "cha2ds2-vasc score"
    ].values
    chads_black = df_chads[df_chads["patient"].isin(list_black)][
        "cha2ds2-vasc score"
    ].values
    chads_asian = df_chads[df_chads["patient"].isin(list_asian)][
        "cha2ds2-vasc score"
    ].values
    chads_hisp = df_chads[df_chads["patient"].isin(list_hisp)][
        "cha2ds2-vasc score"
    ].values
    chads_amind = df_chads[df_chads["patient"].isin(list_amind)][
        "cha2ds2-vasc score"
    ].values
    # evaluate_chads_thresholds(
    #     chads_scores, y, plot_label="CHADS Score Thresholds", color="r"
    # )

    evaluate_chads_thresholds(
        chads_white, y_white, plot_label="CHADS: white", color="b"
    )
    evaluate_chads_thresholds(
        chads_nonwhite, y_nonwhite, plot_label="CHADS: non-white", color="g"
    )
    # evaluate_chads_thresholds(
    #     chads_hisp, y_hisp, plot_label="CHADS - hispanic", color="green"
    # )
    # evaluate_chads_thresholds(
    #     chads_black, y_black, plot_label="CHADS - black", color="blue"
    # )
    # evaluate_chads_thresholds(
    #     chads_asian, y_asian, plot_label="CHADS - asian", color="cyan"
    # )
    # evaluate_chads_thresholds(
    #     chads_amind, y_amind, plot_label="CHADS - amind", color="yellow"
    # )

    plt.plot([0, 1], [0, 1], lw=1, color="grey", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()
