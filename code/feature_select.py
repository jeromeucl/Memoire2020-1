from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import feature_selection, tree
import pandas as pd
from Machine_learning import tbl, worktbl, matching
from sklearn import preprocessing

import numpy as np


def features_selection(features_df, target, nb_of_features):
    scaled_worktbl = features_df.drop(['patientnumber', 'date', 'surgery_date', 'patient_id'], axis=1)
    # normalize the table to
    min_max_scaler = preprocessing.MinMaxScaler()
    a_scaled = pd.DataFrame(min_max_scaler.fit_transform(scaled_worktbl))
    a_scaled.columns = scaled_worktbl.columns
    # https://stackoverflow.com/questions/39839112/the-easiest-way-for-getting-feature-names-after-running-selectkbest-in-scikit-le
    # Create and fit selector with MI
    selector = SelectKBest(feature_selection.mutual_info_classif, k=nb_of_features)
    selector.fit_transform(a_scaled, target.notnull().astype(int).to_frame())
    # Get columns to keep
    cols = a_scaled.columns[selector.get_support(indices=True)]

    # Select variables with tree
    clf = tree.DecisionTreeClassifier(max_depth=5, class_weight='balanced')
    clf = clf.fit(scaled_worktbl, target.fillna(method='bfill').notnull().astype(int).to_frame())
    # Get the most important feature
    importances = clf.feature_importances_

    best_feature = list(scaled_worktbl.columns[np.flip(np.argsort(importances)[-nb_of_features:])])

    features = list(set(list(cols)) | set(list(best_feature)))
    # Create new dataframe with only desired columns, or overwrite existing
    a_scaled = a_scaled[features]

    return a_scaled

a = features_selection(worktbl, tbl['1001_frequency'], 20)

# features_selection(worktbl,label,6)
scaled_worktbl = worktbl.drop(['patientnumber', 'date', 'surgery_date', 'patient_id'], axis=1)

nb_of_best_features = 20

min_max_scaler = preprocessing.MinMaxScaler()
a_scaled = pd.DataFrame(min_max_scaler.fit_transform(scaled_worktbl))
a_scaled.columns = scaled_worktbl.columns
selector = SelectKBest(feature_selection.mutual_info_classif, k=nb_of_best_features)
selector.fit_transform(a_scaled, tbl['1001_frequency'].notnull().astype(int).to_frame())
cols = selector.get_support(indices=True)
colname = a_scaled.columns[cols]



clf = tree.DecisionTreeClassifier(max_depth=5, class_weight='balanced')
clf = clf.fit(scaled_worktbl, tbl['1001_frequency'].notnull().astype(int).to_frame())
# Get the most important feature
importances = clf.feature_importances_

best_feature = scaled_worktbl.columns[np.flip(np.argsort(importances)[-nb_of_best_features:])]

