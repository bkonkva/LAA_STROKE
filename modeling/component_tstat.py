import os,sys
import json
import numpy as np
import pandas as pd
import seaborn as sns

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


def reduce_X(X=[]):
    from sklearn.decomposition import KernelPCA
    rbf_pca = KernelPCA(n_components=100,kernel='rbf',gamma=0.04)
    return rbf_pca.fit_transform(X)

def main():
    # load the SSM features and stroke outcomes
    df = load_data()

    # full X is (214,225), not sure we need all 225 columns
    X = df.drop('stroke', axis=1)
    # X is restricted to MODE_LIMIT=81 (214,81)
    # X = X.iloc[:, :MODE_LIMIT]
    # filter by Afib only
    # X = reduce_X(X=X)
    y = df['stroke']

    print(df.head(3))

    # number of components
    nb_comps = df.shape[1]-1
    print('We are working with {} components'.format(nb_comps))

    sns.violinplot(data=df,x='P0',y='stroke')


if __name__ == "__main__":
    main()

