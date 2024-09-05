import os 
import json
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline as make_pipeline_imb


MODE_LIMIT = 81
RANDOM_STATE = 15

def main():
    DIR_ROOT = os.path.join("D:\\", "LAA_STROKE", "data", "UW")
    DIR_LABELS = os.path.join(DIR_ROOT, 'labels')

    COMPONENT_SCORES = pd.read_csv(os.path.join(DIR_LABELS, 'statistical_shape_modeling', 'component_scores', '02-23.csv'))
    COMPONENT_SCORES = COMPONENT_SCORES.drop('Group', axis=1)

    with open(os.path.join(DIR_LABELS, 'statistical_shape_modeling', 'project_jsons', '02-23.swproj')) as file:
        FILE_NAMES_SSM = json.load(file)

    IDs = [int(os.path.split(FILE_NAMES_SSM['data'][x]['shape_1'])[-1].split(".")[0]) for x in range(len(FILE_NAMES_SSM['data']))]
    COMPONENT_SCORES['patient'] = IDs
    print(COMPONENT_SCORES.shape)
    # print(IDs)
    STROKE = pd.read_csv(os.path.join(DIR_LABELS, 'chart_review', 'stroke_any.csv'))
    print(STROKE.shape)
    df = pd.merge(COMPONENT_SCORES, STROKE, on='patient', how='inner')
    print(df.shape)
    df = df.drop('patient', axis=1)

    X = df.drop('stroke', axis=1)
    X = X.iloc[:, :MODE_LIMIT]
    y = df['stroke']

    # Define the oversampler
    ros = RandomOverSampler(random_state=RANDOM_STATE)

    # Create a logistic regression model
    log_reg = LogisticRegression(max_iter=1000)

    # Define RFE
    rfe = RFE(estimator=log_reg, n_features_to_select=20)

    # Create a pipeline with oversampling and the RFE wrapped model
    pipeline = make_pipeline_imb(ros, rfe, log_reg)

    # Define the k-fold cross-validation procedure
    cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    print(X.shape)
    print(y.shape)
    print(cv)
    # Execute the cross-validation and prediction
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    # y_probs = model.predict_proba()

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    sensitivity = recall_score(y, y_pred)  # recall is the same as sensitivity
    specificity = recall_score(y, y_pred, pos_label=0)
    confusion = confusion_matrix(y, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Confusion Matrix:\n{confusion}")

    # metrics for CHA2DS2-VASc 

    print('done')



if __name__ == "__main__":
    main()

