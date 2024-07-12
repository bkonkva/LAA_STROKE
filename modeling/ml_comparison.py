import os,sys
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline as make_pipeline_sk
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_recall_fscore_support, roc_auc_score
# from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline as make_pipeline_imb


MODE_LIMIT = 81
RANDOM_STATE = 15

def load_data():
    print('Loading data ...')
    DIR_ROOT = os.path.join("D:\\", "LAA_STROKE", "data", "UW")
    DIR_LABELS = os.path.join(DIR_ROOT, 'labels')

    COMPONENT_SCORES = pd.read_csv(os.path.join(DIR_LABELS, 'statistical_shape_modeling', 'component_scores', '02-23.csv'))
    COMPONENT_SCORES = COMPONENT_SCORES.drop('Group', axis=1)

    with open(os.path.join(DIR_LABELS, 'statistical_shape_modeling', 'project_jsons', '02-23.swproj')) as file:
        FILE_NAMES_SSM = json.load(file)

    IDs = [int(os.path.split(FILE_NAMES_SSM['data'][x]['shape_1'])[-1].split(".")[0]) for x in range(len(FILE_NAMES_SSM['data']))]
    COMPONENT_SCORES['patient'] = IDs
    # print(COMPONENT_SCORES.shape)
    # print(IDs)
    STROKE = pd.read_csv(os.path.join(DIR_LABELS, 'chart_review', 'stroke_any.csv'))
    # print(STROKE.shape)
    df = pd.merge(COMPONENT_SCORES, STROKE, on='patient', how='inner')
    # print(df.head())
    df = df.drop('patient', axis=1)
    return df


def log_reg(X=[],y=[]):
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
    # Execute the cross-validation and prediction
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    # y_probs = model.predict_proba()
    return y_pred


def run_svm(X=[],y=[]):
    # make pipeline with LinearSVC model
    # pipeline = make_pipeline_sk(StandardScaler(), LinearSVC(random_state=RANDOM_STATE,tol=1e-5))
    # pipeline = make_pipeline_sk(StandardScaler(), SVC(C=5,random_state=RANDOM_STATE,kernel='poly',degree=10,coef0=1))
    pipeline = make_pipeline_sk(StandardScaler(),SGDClassifier(loss="hinge",max_iter=1000,tol=1e-3))
    # Define the k-fold cross-validation procedure
    cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    # Execute the cross-validation and prediction
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    # y_probs = model.predict_proba()
    return y_pred


def run_DT(X=[],y=[]):
    # make pipeline with Decision Tree model
    pipeline = make_pipeline_sk(DecisionTreeClassifier(random_state=RANDOM_STATE,max_depth=2))
    # pipeline = make_pipeline_sk(StandardScaler(), SVC(random_state=RANDOM_STATE,gamma='auto'))
    # Define the k-fold cross-validation procedure
    cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    # Execute the cross-validation and prediction
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    # y_probs = model.predict_proba()
    return y_pred


def run_RF(X=[],y=[]):
    # make pipeline with Random Forest Classifier model
    pipeline = make_pipeline_sk(RandomForestClassifier(n_estimators=500,n_jobs=-1,random_state=RANDOM_STATE))
    # pipeline = make_pipeline_sk(StandardScaler(), SVC(random_state=RANDOM_STATE,gamma='auto'))
    # Define the k-fold cross-validation procedure
    cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    # Execute the cross-validation and prediction
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    # y_probs = model.predict_proba()
    return y_pred


def run_KNN(X=[],y=[]):
    # make pipeline with K-Nearest Neighbor model
    pipeline = make_pipeline_imb(KNeighborsClassifier(n_neighbors=3))
    # pipeline = make_pipeline_sk(StandardScaler(), SVC(random_state=RANDOM_STATE,gamma='auto'))
    # Define the k-fold cross-validation procedure
    cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    # Execute the cross-validation and prediction
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    # y_probs = model.predict_proba()
    return y_pred


def run_GNB(X=[],y=[]):
    # make pipeline with Random Forest Classified model
    pipeline = make_pipeline_sk(GaussianNB())
    # pipeline = make_pipeline_sk(BayesianGaussianMixture(n_components=10,n_init=10))
    # pipeline = make_pipeline_sk(StandardScaler(), SVC(random_state=RANDOM_STATE,gamma='auto'))
    # Define the k-fold cross-validation procedure
    cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    # Execute the cross-validation and prediction
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    # y_probs = model.predict_proba()
    return y_pred


def run_MLP(X=[],y=[]):
    from sklearn.neural_network import MLPClassifier
    # make pipeline with Random Forest Classified model
    pipeline = make_pipeline_sk(MLPClassifier(random_state=RANDOM_STATE, alpha=0.001,max_iter=30000))
    # pipeline = make_pipeline_sk(BayesianGaussianMixture(n_components=10,n_init=10))
    # pipeline = make_pipeline_sk(StandardScaler(), SVC(random_state=RANDOM_STATE,gamma='auto'))
    # Define the k-fold cross-validation procedure
    cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    # Execute the cross-validation and prediction
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    # y_probs = model.predict_proba()
    return y_pred


def add_stats(df_stats,ml_name='',y=[],y_pred=[]):
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    sensitivity = recall_score(y, y_pred)  # recall is the same as sensitivity
    specificity = recall_score(y, y_pred, pos_label=0)

    precision, recall, fscore = precision_recall_fscore_support(y,y_pred,average='binary')[:3]

    AUC = roc_auc_score(y,y_pred)
    confusion = confusion_matrix(y, y_pred)

    if sensitivity-recall > 0.1:
        print('Whoops, check the following discrepancy: (sensitivity,recall) = ({0:0.5f},{1:0.5f})'.format(sensitivity,recall))

    # print(f"\n==ML model name: {ml_name}==\n")
    # print(f"Confusion Matrix:\n{confusion}")

    tn,fp,fn,tp = confusion.flatten().tolist()

    df_curr = pd.DataFrame([[ml_name,accuracy,AUC,precision,recall,fscore,sensitivity,specificity,tn,fp,fn,tp]],columns=list(df_stats))
    df_stats = pd.concat([df_stats,df_curr],ignore_index=True)

    return df_stats

def reduce_X(X=[]):
    from sklearn.decomposition import KernelPCA
    rbf_pca = KernelPCA(n_components=100,kernel='rbf',gamma=0.04)
    return rbf_pca.fit_transform(X)

def compare_ml_models(X=[],y=[]):

    print('Original components size is {}'.format(X.shape))

    # p < 0.05, top 8 components
    # components = ['P48', 'P68', 'P76', 'P83', 'P131', 'P177', 'P185', 'P216']
    # p < 0.1, top 21 components
    components = ['P6', 'P7', 'P13', 'P19', 'P46', 'P48', 'P60', 'P68', 'P76', 'P77', 'P83', 'P131', 'P145', 'P146', 'P177', 'P182', 'P185', 'P195', 'P208', 'P216', 'P219']
    # p < 0.2, top 44 components
    # components = ['P1', 'P5', 'P6', 'P7', 'P13', 'P19', 'P27', 'P34', 'P40', 'P46', 'P48', 'P57', 'P58', 'P60', 'P62', 'P63', 'P64', 'P66', 'P68', 'P71', 'P72', 'P76', 'P77', 'P83', 'P107', 'P113', 'P117', 'P122', 'P131', 'P133', 'P145', 'P146', 'P147', 'P177', 'P182', 'P183', 'P185', 'P188', 'P193', 'P195', 'P208', 'P209', 'P216', 'P219']
    X = X[components]

    print('Using the top {} most discriminant components'.format(len(components)))
    print('Updated components size is {}'.format(X.shape))

    # write to file, to make accessible to PowerBI
    # patient id, prediction score for each model
    # make ROC for patient level?
    columns = ['ML model','accuracy','AUC', 'precision', 'recall', 'F1','sensitivity','specificity','tn','fp','fn','tp']
    df_stats = pd.DataFrame([],columns=columns)
    
    # continue developing other ML: class imbalance, 
    # ensure shape feature extraction, reproduce within software
    # chad2vasc for all 
    ml_name='Logistic Regression'
    print('Running {} ...'.format(ml_name))
    y_pred = log_reg(X=X,y=y)
    df_stats = add_stats(df_stats,ml_name=ml_name,y=y,y_pred=y_pred)
    
    ml_name='Linear SVC'
    print('Running {} ...'.format(ml_name))
    y_pred = run_svm(X=X,y=y)
    df_stats = add_stats(df_stats,ml_name=ml_name,y=y,y_pred=y_pred)

    ml_name='Decision Tree'
    print('Running {} ...'.format(ml_name))
    y_pred = run_DT(X=X,y=y)
    df_stats = add_stats(df_stats,ml_name=ml_name,y=y,y_pred=y_pred)

    ml_name='Random Forest Classifier'
    print('Running {} ...'.format(ml_name))
    y_pred = run_RF(X=X,y=y)
    df_stats = add_stats(df_stats,ml_name=ml_name,y=y,y_pred=y_pred)

    # ml_name='K-Nearest Neighbors'
    # print('Running {} ...'.format(ml_name))
    # y_pred = run_KNN(X=X,y=y)
    # df_stats = add_stats(df_stats,ml_name=ml_name,y=y,y_pred=y_pred)

    ml_name='Gaussian Naive Bayes'
    print('Running {} ...'.format(ml_name))
    y_pred = run_GNB(X=X,y=y)
    df_stats = add_stats(df_stats,ml_name=ml_name,y=y,y_pred=y_pred)

    ml_name='Multi-layer Perceptron'
    print('Running {} ...'.format(ml_name))
    y_pred = run_MLP(X=X,y=y)
    df_stats = add_stats(df_stats,ml_name=ml_name,y=y,y_pred=y_pred)

    print(df_stats)

def get_pvals(df):
    comps = list(df.drop('stroke', axis=1))    
    stroke_counts = df['stroke'].value_counts()
    group1 = df[df['stroke']==stroke_counts.index[0]]
    group2 = df.loc[df['stroke']==stroke_counts.index[1]]
    pval_all = []
    for c in comps:
        ttest = ttest_ind(group1[c], group2[c])
        pval_all = pval_all + [ttest.pvalue]
    dfp = pd.DataFrame(list(zip(comps,pval_all)),columns=['components','pval'])
    # dfp.sort_values(by=['pval'],inplace=True)
    # dfp = dfp.reset_index(drop=True)
    return dfp

def svm_by_comp(X=[],y=[],dfp=[]):
    ml_name='Linear SVC'
    print('Running {} ...'.format(ml_name))
    components = []
    AUC = []
    for c in dfp['components']:
        components=components+[c]
        Xcur = X[components]
        y_pred = run_svm(X=Xcur,y=y)
        AUC = AUC + [roc_auc_score(y,y_pred)]
    dfp['AUC'] = AUC
    print(dfp)

    # plot results
    fig,ax = plt.subplots()
    plt.plot(dfp['components'],dfp['AUC'])
    plt.xlabel('cumulative components (by discriminant order)')
    ax.set_ylabel('AUC for Linear SVC')
    xticks = [1] + [i for i in range(50,len(components)+1,50)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    plt.tight_layout()
    plt.show()

def original_comparison():
    # this loads the original SSM features from 02-23
    # then runs comparison across ML models
    # option to rank the components by p-value or not

    # load the SSM features and stroke outcomes
    df = load_data()
    dfp = get_pvals(df)
    # full X is (214,225), not sure we need all 225 columns
    X = df.drop('stroke', axis=1)
    # X is restricted to MODE_LIMIT=81 (214,81)
    # X = X.iloc[:, :MODE_LIMIT]
    # filter by Afib only
    # print(X.shape)
    # X = reduce_X(X=X)
    y = df['stroke']

    compare_ml_models(X=X,y=y)
    # svm_by_comp(X=X,y=y,dfp=dfp)

def main():

    original_comparison()


if __name__ == "__main__":
    main()

