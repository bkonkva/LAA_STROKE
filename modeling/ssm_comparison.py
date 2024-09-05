import os,sys
import json
import numpy as np
import math
import pandas as pd
from scipy.stats import ttest_ind,f

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
DIR_ROOT = os.path.join("D:\\", "LAA_STROKE", "data", "UW")
DIR_LABELS = os.path.join(DIR_ROOT, 'labels')

# full dataset : D:\LAA_STROKE\data\UW\labels\segmentation_masks\LAA_ISOLATED\ARRAY_from_nnunet


def load_data(nb_corr=256):
    print('Loading data ...')

    COMPONENT_SCORES = pd.read_csv(os.path.join(DIR_LABELS, 'statistical_shape_modeling', 'component_scores', '02-23_xp','02-23_corr{}_comp.csv'.format(nb_corr)))
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


def log_reg_rfe(X=[],y=[]):
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


def log_reg(X=[],y=[]):
    # Create a logistic regression model
    log_reg = LogisticRegression(random_state=RANDOM_STATE,max_iter=1000)
    # Create a pipeline with oversampling and the RFE wrapped model
    pipeline = make_pipeline_sk(StandardScaler(),log_reg)
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

def TwoSampleT2Test(X, Y):
    nx, p = X.shape
    ny, _ = Y.shape
    delta = np.mean(X, axis=0) - np.mean(Y, axis=0)
    Sx = np.cov(X, rowvar=False)
    Sy = np.cov(Y, rowvar=False)
    S_pooled = ((nx-1)*Sx + (ny-1)*Sy)/(nx+ny-2)
    t_squared = (nx*ny)/(nx+ny) * np.matmul(np.matmul(delta.transpose(), np.linalg.inv(S_pooled)), delta)
    statistic = t_squared * (nx+ny-p-1)/(p*(nx+ny-2))
    F = f(p, nx+ny-p-1)
    p_value = 1 - F.cdf(statistic)
    # print(f"Test statistic: {statistic}\nDegrees of freedom: {p} and {nx+ny-p-1}\np-value: {p_value}")
    return statistic, p_value

def get_pvals(df):
    comps = list(df.drop('stroke', axis=1))    
    stroke_counts = df['stroke'].value_counts()
    group1 = df[df['stroke']==stroke_counts.index[0]]
    group2 = df.loc[df['stroke']==stroke_counts.index[1]]
    pval_all = []
    t2p_all = []
    for ic,c in enumerate(comps):
        # versicolor = iris.data[iris.target==1, :2]
        # virginica = iris.data[iris.target==2, :2]
        # print(group1.loc[:,:c])
        t2p = np.nan
        if ic:
            t2,t2p = TwoSampleT2Test(group1.loc[:,:c], group2.loc[:,:c])
        t2p_all = t2p_all + [t2p]
        ## Test statistic: 15.82660099191812
        ## Degrees of freedom: 2 and 97
        ## p-value: 1.1259783253558808e-06

        ttest = ttest_ind(group1[c], group2[c])
        pval_all = pval_all + [ttest.pvalue]
    dfp = pd.DataFrame(list(zip(comps,pval_all,t2p_all)),columns=['components','pval','T2pval'])
    # dfp.sort_values(by=['pval'],inplace=True)
    # dfp = dfp.reset_index(drop=True)
    return dfp

def MLP_by_comp(X=[],y=[],dfp=[]):
    ml_name='Multilayer Perceptron'
    print('Running {} ...'.format(ml_name))
    components = []
    AUC = [0]
    for ic,c in enumerate(dfp['components']):
        components=components+[c]
        if ic==0:
            continue
        Xcur = X[components]
        y_pred = run_MLP(X=Xcur,y=y)
        AUC = AUC + [roc_auc_score(y,y_pred)]
    dfp['AUC'] = AUC
    # print(dfp)
    return dfp

def logreg_by_comp(X=[],y=[],dfp=[]):
    ml_name='Logistic Regression'
    print('Running {} ...'.format(ml_name))
    components = []
    AUC = [0]
    for ic,c in enumerate(dfp['components']):
        components=components+[c]
        if ic==0:
            continue
        Xcur = X[components]
        y_pred = log_reg(X=Xcur,y=y)
        AUC = AUC + [roc_auc_score(y,y_pred)]
    dfp['AUC'] = AUC
    # print(dfp)
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
    # print(dfp)
    return dfp

def plot_by_comp(subplot_item = 321,ylabel='AUC for Linear SVC',dfp=[],nb_comps = 100):
    # plot results
    ax = plt.subplot(subplot_item)
    plt.plot(dfp['components'],dfp['AUC'])
    # plt.plot(dfp['components'],dfp['explained_var'],'r-')
    if math.ceil((subplot_item%10) / (subplot_item//10%10)) == subplot_item//100 :
        plt.xlabel('Cumul. components (by order)')
    if subplot_item%10% (subplot_item//10%10) == 1:
        ax.set_ylabel(ylabel)
    step = nb_comps//4
    xticks = [1] + [i for i in range(step,nb_comps+1,step)]
    ax.set_xticks(xticks)
    ax.set_ylim([0.4,0.8])
    ax.set_xticklabels(xticks)
    return ax
    # plt.show()

def cut_var_sort(dfp=[],var_cutoff=0.95):
    print(dfp)
    dfp = dfp.loc[dfp['explained_var']<var_cutoff]
    dfp.sort_values(by=['pval'],inplace=True)
    dfp = dfp.reset_index(drop=True)    
    print(dfp)
    return dfp

def prep_by_comp(nb_corr = 256,var_cutoff=0.95):
    # this loads the original SSM features from 02-23
    # then runs comparison across ML models
    # option to rank the components by p-value or not

    # load the SSM features and stroke outcomes
    # nb_corr = 256
    df = load_data(nb_corr=nb_corr)
    print(df)
    dfp = get_pvals(df)
    df_var = pd.read_csv(os.path.join(DIR_LABELS, 'statistical_shape_modeling', 'component_scores', '02-23_xp','02-23_corr{}_expvar.csv'.format(nb_corr)))
    print(df_var)
    plot_color = {64:'r',256:'k',1024:'g'}
    if var_cutoff == 0.95:
        plt.figure(1001)
        plt.plot(df_var['#  Number of Modes'],df_var[' Explained Variance'],'{}--'.format(plot_color[nb_corr]),label='# correspondence = {}'.format(nb_corr))
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance')
        if nb_corr==1024:
            plt.legend()
    dfp['explained_var'] = df_var[' Explained Variance']
    dfp=dfp.fillna(1.0)
    print(dfp)
    dfp = cut_var_sort(dfp=dfp,var_cutoff=var_cutoff)

    # dfp = append_var(dfp,nb_corr)
    
    # full X is (214,225), not sure we need all 225 columns
    X = df.drop('stroke', axis=1)
    # X is restricted to MODE_LIMIT=81 (214,81)
    # X = X.iloc[:, :MODE_LIMIT]
    # filter by Afib only
    # print(X.shape)
    # X = reduce_X(X=X)
    y = df['stroke']

    return X,y,dfp

def correspondence_comparison(subplot_item = 321, nb_corr = 256,var_cutoff=0.95):
    
    X,y,dfp0 = prep_by_comp(nb_corr = nb_corr,var_cutoff=var_cutoff)

    # compare_ml_models(X=X,y=y)
    # dfp = svm_by_comp(X=X,y=y,dfp=dfp0)
    # plt.figure(num=1002, figsize=(8,6))
    # ax = plot_by_comp(subplot_item=subplot_item,ylabel='AUC for Linear SVC',dfp=dfp,nb_comps=len(dfp['components']))
    # ax.set_title('(var,#corr) :: ({0:0.2f},{1})'.format(var_cutoff,nb_corr))
    # plt.tight_layout()

    dfp = logreg_by_comp(X=X,y=y,dfp=dfp0)
    plt.figure(num=1003, figsize=(8,6))
    ax = plot_by_comp(subplot_item=subplot_item,ylabel='AUC for logistic regression',dfp=dfp,nb_comps=len(dfp['components']))
    ax.set_title('(var,#corr) :: ({0:0.2f},{1})'.format(var_cutoff,nb_corr))
    plt.tight_layout()    

    # dfp = MLP_by_comp(X=X,y=y,dfp=dfp0)
    # plt.figure(num=1004, figsize=(8,6))
    # ax = plot_by_comp(subplot_item=subplot_item,ylabel='AUC for Multilayer Perc',dfp=dfp,nb_comps=len(dfp['components']))
    # ax.set_title('(var,#corr) :: ({0:0.2f},{1})'.format(var_cutoff,nb_corr))
    # plt.tight_layout()    
    

def main():
    # var_cutoff = 0.95
    # nb_corr = 256
    nb_corrs = [64,256,1024]
    var_cuttoffs = [0.95,0.98,0.99]
    plot_item = 0
    for nb_corr in nb_corrs:
        for var_cutoff in var_cuttoffs:
            plot_item = plot_item+1
            print('Number of correspondence points : >>{}<<'.format(nb_corr))
            print('Variance explained cutoff : {}'.format(var_cutoff))
            correspondence_comparison(subplot_item = 100*len(nb_corrs)+10*len(var_cuttoffs)+plot_item, nb_corr = nb_corr,var_cutoff=var_cutoff)
    plt.show()


if __name__ == "__main__":
    main()

