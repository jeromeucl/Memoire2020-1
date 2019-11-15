from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
from Machine_learning import tbl, worktbl, matching
from sklearn import preprocessing

def features_selection(features_df, target,nb_of_features):
    # normalize the table to

    #https://stackoverflow.com/questions/39839112/the-easiest-way-for-getting-feature-names-after-running-selectkbest-in-scikit-le
    # Create and fit selector
    selector = SelectKBest(chi2, k=nb_of_features)
    selector.fit_transform(features_df, target)
    # Get columns to keep
    cols = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
    df = features_df.iloc[cols]

    return df
#features_selection(worktbl,label,6)
scaled_worktbl = worktbl.drop(['patientnumber', 'date', 'surgery_date', 'patient_id'], axis=1)
a = pd.DataFrame(data = {'col1': [1000, 200,300], 'col2': [10, 4,1],'col3': [1000, 200,300]})

min_max_scaler = preprocessing.MinMaxScaler()
a_scaled = pd.DataFrame(min_max_scaler.fit_transform(scaled_worktbl))
a_scaled.columns = scaled_worktbl.columns
selector = SelectKBest(chi2, k=3)
selector.fit_transform(a_scaled, tbl['1001_frequency'].fillna(method='bfill').fillna(method='ffill'))
cols = selector.get_support(indices=True)
a_scaled = a_scaled.iloc[cols]