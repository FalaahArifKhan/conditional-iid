import pandas as pd
import numpy as np
import scipy as sp
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

def predict_ensemble(models, train_groups, test_samples, pred_threshold=0.5):
    ensemble_preds = np.array([models[train_group].predict_proba(test_samples)[:,1] for train_group in train_groups if train_group != 'overall']).mean(axis=0)
    y_preds = [int(x>=pred_threshold) for x in ensemble_preds]
    return y_preds

def predict_CIID(models, train_groups, test_samples_groups, features_lst, target_name):
    #test_samples_groups = set_protected_groups(test_samples, ['sex','race'], [0, 'Caucasian'])
    y_true = []
    y_preds = []
    for test_group_name in test_samples_groups.keys():
        if (len(test_samples_groups[test_group_name]) > 0) & (test_group_name in train_groups) :
            y_true+= list(test_samples_groups[test_group_name][target_name].values)
            y_preds+=list(models[test_group_name].predict(test_samples_groups[test_group_name][features_lst]))
    
    return y_true, y_preds


def intialize_base_classifer(experiment, name, SEED):
    if experiment == 'folktables':
        if name == 'knn':
            return KNeighborsClassifier(n_neighbors=15, weights='uniform', metric='minkowski')
        if name == 'dtc':
            return DecisionTreeClassifier(criterion='entropy', max_depth=10, max_features=0.6, min_samples_split=0.02, random_state=SEED)
        if name == 'mlp':
            return MLPClassifier(hidden_layer_sizes=(100, 50, 100), learning_rate='constant', solver='sgd', activation='relu', random_state=SEED)
        if name == 'lr':
            return LogisticRegression(C=1, max_iter=200, solver='liblinear', random_state=SEED)
        if name == 'rf':
            return RandomForestClassifier(max_depth = 10, max_features = 0.6, min_samples_leaf= 2, n_estimators=50, random_state=SEED)

    if experiment == 'compas':
        if name == 'knn':
            return KNeighborsClassifier(n_neighbors=15, weights='uniform', metric='minkowski')
        if name == 'dtc':
            return DecisionTreeClassifier(criterion='gini', max_depth=5, max_features=0.6, min_samples_split=0.1, random_state=SEED)
        if name == 'mlp':
            return MLPClassifier(hidden_layer_sizes=(100,), learning_rate='constant', solver='adam', activation='tanh', random_state=SEED)
        if name == 'lr':
            return LogisticRegression(C=1, max_iter=200, solver='liblinear', random_state=SEED)
        if name == 'rf':
            return RandomForestClassifier(max_depth = 4, max_features = 0.6, min_samples_leaf= 1, n_estimators=100, random_state=SEED)
    

def initialize_base_model(experiment_name, model_name, SEED, categorical_columns, numerical_columns,category_lst=None):
    if category_lst == None:
        category_lst = 'auto'
    encoder = ColumnTransformer(transformers=[
                    ('categorical_features', OneHotEncoder(categories=category_lst, sparse=False, handle_unknown='infrequent_if_exist'), categorical_columns),
                    ('numerical_features', StandardScaler(), numerical_columns)])
    model = Pipeline([
                                ('features', encoder),
                                ('learner', intialize_base_classifer(experiment=experiment_name, name=model_name, SEED=SEED))
                    ])
    return model    

def print_demographics(train_groups, full_size):
    group_stats = {}
    for g in train_groups.keys():
        if g != 'overall':
            group_stats[g] = train_groups[g].shape[0]/full_size
    
    return pd.DataFrame([group_stats])

def partition_by_group_intersectional(df, column_names, priv_values):
    priv_priv = df[(df[column_names[0]] == priv_values[0]) & (df[column_names[1]] == priv_values[1])]
    priv_dis = df[(df[column_names[0]] == priv_values[0]) & (df[column_names[1]] != priv_values[1])]
    dis_priv = df[(df[column_names[0]] != priv_values[0]) & (df[column_names[1]] == priv_values[1])]
    dis_dis = df[(df[column_names[0]] != priv_values[0]) & (df[column_names[1]] != priv_values[1])]
    return priv_priv, priv_dis, dis_priv, dis_dis


def partition_by_group_binary(df, column_name, priv_value):
    priv = df[df[column_name] == priv_value]
    dis = df[df[column_name] != priv_value]
    if len(priv)+len(dis) != len(df):
        raise ValueError("Error! Not a partition")
    return priv, dis


def set_protected_groups(X_test, column_names, priv_values):
    groups={}
    #groups[column_names[0]+'_'+column_names[1]+'_priv'], groups[column_names[0]+'_'+column_names[1]+'_dis'] = partition_by_group_intersectional(X_test, column_names, priv_values)
    groups[column_names[0]+'_'+column_names[1]+'_priv_priv'], groups[column_names[0]+'_'+column_names[1]+'_priv_dis'], groups[column_names[0]+'_'+column_names[1]+'_dis_priv'], groups[column_names[0]+'_'+column_names[1]+'_dis_dis'] = partition_by_group_intersectional(X_test, column_names, priv_values)
    groups[column_names[0]+'_priv'], groups[column_names[0]+'_dis'] = partition_by_group_binary(X_test, column_names[0], priv_values[0])
    groups[column_names[1]+'_priv'], groups[column_names[1]+'_dis'] = partition_by_group_binary(X_test, column_names[1], priv_values[1])
    return groups 

