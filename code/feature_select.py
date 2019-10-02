from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


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

a = pd.DataFrame(data = {'col1': [1000, 200,300], 'col2': [10, 4,1]})
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
a_scaled = min_max_scaler.fit_transform(a)