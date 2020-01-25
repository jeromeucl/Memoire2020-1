from Machine_learning import frequency_of_exercise_output,intensity_of_exercise_output,matching_intensity,matching, Working_Directory,inputML_knee,inputML_hip
from sklearn.metrics import mean_squared_error, make_scorer
from math import sqrt
from sklearn.model_selection import train_test_split
def custom_metric(y_test, y_pred):
    return sqrt(mean_squared_error(y_test, y_pred))
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt



'''This part is aimed to predict the frequency'''
from sklearn.model_selection import cross_val_score

exo = '1005_frequency'
worktbl = inputML_knee.drop(['patientnumber', 'date', 'surgery_date', 'patient_id'], axis=1)
learn_index = frequency_of_exercise_output[exo].notna()

input = worktbl.loc[learn_index]
target = frequency_of_exercise_output[exo].loc[learn_index]

M_squared_error = make_scorer(custom_metric, greater_is_better=False)


'''Scale data'''
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(input), columns=input.columns)




'''Train-test split'''
#X_train_valid, X_test, y_train_valid, y_test = train_test_split(input, target, test_size=0.2, random_state=42)
X_train_valid, X_test, y_train_valid, y_test = train_test_split(df_scaled, target, test_size=0.2, random_state=42)

#Parameters

param_grid = p_grid = {"max_depth": [2,3,5,10,15],"min_samples_split" : [5,10,20],"min_impurity_decrease" : [0,0.01,0.02,0.1],'criterion':['mse','friedman_mse','mae']}
param_grid = p_grid = {"max_depth": [2,3],"min_samples_split" : [20],"min_impurity_decrease" : [0,0.01],'criterion':['mse']}
model = DecisionTreeRegressor()

param_grid = {'kernel': ('linear', 'rbf'), 'C': [1, 10], 'gamma':['scale']}
param_grid = {'kernel': ['rbf'], 'C': [1], 'gamma':['scale']}
model = svm.SVR()

param_grid = dict(fit_intercept = [True])
model = LinearRegression()


param_grid = dict(n_neighbors = [10,20,30], weights = ["uniform", "distance"])
model = KNeighborsRegressor()

''''''
grid = GridSearchCV(model, param_grid, cv = 5, scoring = M_squared_error)
grid.fit(X_train_valid,y_train_valid)


print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)


print('Results on the test set')
clf = grid.best_estimator_
clf.fit(X_train_valid,y_train_valid)
pred = np.around(np.transpose(clf.predict(X_test)))
print(custom_metric(y_test, pred))


'''This part is aimed to predict the intensity'''