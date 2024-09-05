from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
import os
import json
import ast


from matplotlib import pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_predict,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb


DIR_ROOT = os.path.join("D:\\", "LAA_STROKE", "data", "UW")
DIR_WRITE = os.path.join(DIR_ROOT, "labels", "chart_review")
DIR_LABELS = os.path.join(DIR_ROOT, "labels")
DIR_METRICS = os.path.join("D:\\", "LAA_STROKE", "models", "metrics")

MODE_LIMIT = 10
RANDOM_STATE = 12

COMPONENT_SCORES = pd.read_csv(
    os.path.join(
        DIR_LABELS, "statistical_shape_modeling", "component_scores", "02-23.csv"
    )
)
COMPONENT_SCORES = COMPONENT_SCORES.drop("Group", axis=1)
COMPONENT_SCORES = COMPONENT_SCORES.iloc[:, :MODE_LIMIT]
with open(
    os.path.join(
        DIR_LABELS, "statistical_shape_modeling", "project_jsons", "02-23.swproj"
    )
) as file:
    FILE_NAMES_SSM = json.load(file)
IDs = [
    int(os.path.split(FILE_NAMES_SSM["data"][x]["shape_1"])[-1].split(".")[0])
    for x in range(len(FILE_NAMES_SSM["data"]))
]

STROKE = pd.read_csv(os.path.join(DIR_LABELS, "chart_review", "stroke_any.csv"))
COMPONENT_SCORES["patient"] = IDs


path_chads_score = os.path.join(DIR_WRITE, "chads_score_nostroke.csv")
path_stroke_history = os.path.join(DIR_WRITE, "stroke_any.csv")
path_chads_features = os.path.join(DIR_WRITE, "chads_features.csv")

df_chads = pd.read_csv(path_chads_score)
df_stroke = pd.read_csv(path_stroke_history)
df_chads_features = pd.read_csv(path_chads_features)
chads = list(df_chads["cha2ds2-vasc score"])
stroke = list(df_stroke["stroke"])


df_chads_features_new = pd.DataFrame(
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
for index, row in df_chads_features.iterrows():
    dict_chads = ast.literal_eval(row["cha2ds2-vasc features"])
    dict_chads.pop("stroke history")
    add_values = list(dict_chads.values())
    add_values.insert(0, row["patient"])
    df_chads_features_new.loc[index] = add_values


def return_tpr(chads_threshold):
    y_pred = [1 if risk > chads_threshold else 0 for risk in chads]
    return recall_score(stroke, y_pred)


def return_fpr(chads_threshold):
    y_pred = [1 if risk > chads_threshold else 0 for risk in chads]
    return 1 - recall_score(stroke, y_pred, pos_label=0)


def return_weights(log_reg, rfe, X):
    coefficients = log_reg.coef_[0]
    selected_features = rfe.support_
    selected_feature_names = X.columns[selected_features]
    feature_importance = pd.DataFrame(
        coefficients, index=selected_feature_names, columns=["Coefficient"]
    )
    feature_importance["Absolute"] = feature_importance["Coefficient"].abs()
    feature_importance = feature_importance.sort_values(by="Absolute", ascending=False)
    return feature_importance


def logistic_regression():
    return


def multilayer_perceptron():
    return


def return_features():
    df = pd.merge(COMPONENT_SCORES, STROKE, on="patient", how="inner")
    df = df.drop("patient", axis=1)

    X = df.drop("stroke", axis=1)
    y = df["stroke"]
    return df, X, y


def return_features_with_chads():
    return


def main():

    # Define the oversampler
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    # Create a logistic regression model
    log_reg = LogisticRegression(max_iter=100000)
    # Define RFE
    rfe = RFE(estimator=log_reg, n_features_to_select=20)
    # Create a pipeline with oversampling and the RFE wrapped model
    pipeline = make_pipeline_imb(ros, rfe, log_reg)
    # Define the k-fold cross-validation procedure
    cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(10, 8))

    # Manually iterate over the folds
    for train, test in cv.split(X, y):
        pipeline.fit(X.iloc[train], y.iloc[train])
        probas_ = pipeline.predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (len(aucs), roc_auc))
    ssm_weights = return_weights(log_reg, rfe, X)
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="y",
        label=r"SSM (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    # SSM + CHADS

    df = pd.merge(COMPONENT_SCORES, STROKE, on="patient", how="inner")
    df = pd.merge(df, df_chads_features_new, on="patient", how="inner")
    df = df.drop("patient", axis=1)

    X = df.drop("stroke", axis=1)
    y = df["stroke"]

    # Define the oversampler
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    # Create a logistic regression model
    log_reg = LogisticRegression(max_iter=100000)
    # Define RFE
    rfe = RFE(estimator=log_reg, n_features_to_select=20)
    # Create a pipeline with oversampling and the RFE wrapped model
    pipeline = make_pipeline_imb(ros, rfe, log_reg)
    # Define the k-fold cross-validation procedure
    cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # plt.figure(figsize=(10, 8))

    # Manually iterate over the folds
    for train, test in cv.split(X, y):
        pipeline.fit(X.iloc[train], y.iloc[train])
        probas_ = pipeline.predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (len(aucs), roc_auc))

    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    ssm_chads_weights = return_weights(log_reg, rfe, X)
    # print(f'ssm weights: {ssm_weights}')
    # print(f'ssm chads weights: {ssm_chads_weights}')
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"SSM + CHADS (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    TPR_chads = [return_tpr(thresh) for thresh in range(0, np.max(chads))]
    FPR_chads = [return_fpr(thresh) for thresh in range(0, np.max(chads))]
    AUC_chads = auc(FPR_chads, TPR_chads)
    plt.plot(
        FPR_chads,
        TPR_chads,
        color="g",
        label=r"CHADS (AUC = %0.2f)" % (AUC_chads),
        lw=2,
        alpha=0.8,
    )

    # MLP

    df = pd.merge(COMPONENT_SCORES, STROKE, on="patient", how="inner")
    df = df.drop("patient", axis=1)

    X = df.drop("stroke", axis=1)
    y = df["stroke"]

    smote = SMOTE(random_state=RANDOM_STATE)

    mlp_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(1000,),
                    activation="relu",
                    solver="adam",
                    max_iter=100000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=5)
    TPR_MLP = []
    FPR_MLP = []

    thresholds = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train, y_train = smote.fit_resample(X_train, y_train)
        mlp_pipeline.fit(X_train, y_train)
        y_scores = mlp_pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, threshold = roc_curve(y_test, y_scores)
        TPR_MLP.append(tpr)
        FPR_MLP.append(fpr)
        thresholds.append(threshold)
    # TPR = [item for array in TPR_MLP for item in array]
    # FPR = [item for array in FPR_MLP for item in array]
    AUC_mlp = auc(fpr, tpr)
    plt.plot(
        fpr,
        tpr,
        color="c",
        label=r"MLP (AUC = %0.2f)" % (AUC_mlp),
        lw=2,
        alpha=0.8,
    )

    # MLP + CHADS

    df = pd.merge(COMPONENT_SCORES, STROKE, on="patient", how="inner")
    df = pd.merge(df, df_chads_features_new, on="patient", how="inner")
    df = df.drop("patient", axis=1)

    X = df.drop("stroke", axis=1)
    y = df["stroke"]

    smote = SMOTE(random_state=RANDOM_STATE)

    mlp_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(1000,),
                    activation="relu",
                    solver="adam",
                    max_iter=100000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=5)
    TPR_MLP = []
    FPR_MLP = []

    thresholds = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train, y_train = smote.fit_resample(X_train, y_train)
        mlp_pipeline.fit(X_train, y_train)
        y_scores = mlp_pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, threshold = roc_curve(y_test, y_scores)
        TPR_MLP.append(tpr)
        FPR_MLP.append(fpr)
        thresholds.append(threshold)
    # TPR = [item for array in TPR_MLP for item in array]
    # FPR = [item for array in FPR_MLP for item in array]
    AUC_mlp = auc(fpr, tpr)
    plt.plot(
        fpr,
        tpr,
        color="m",
        label=r"MLP + CHADS (AUC = %0.2f)" % (AUC_mlp),
        lw=2,
        alpha=0.8,
    )

    cv_scores = cross_val_score(mlp_pipeline, X, y, cv=5, scoring="accuracy")
    print()

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(DIR_METRICS, "roc-auc-ssm.jpg"), format="jpg", dpi=300)
    ssm_weights.to_csv(os.path.join(DIR_METRICS, "ssm_weights.csv"), index=True)
    ssm_chads_weights.to_csv(
        os.path.join(DIR_METRICS, "ssm_chads_weights.csv"), index=True
    )
    plt.show()


if __name__ == "__main__":
    main()
