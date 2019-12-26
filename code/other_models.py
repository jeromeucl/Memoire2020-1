#https://gist.github.com/sawansaurabh/3748a083ffdec38aacb8f43660a5f654
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from feature_select import features_selection
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression
BCR = make_scorer(balanced_accuracy_score)


tbl_db = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\online_tbl2019-12-09.csv")
worktbl_db = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\online_worktbl2019-12-09.csv")
matching = [s for s in tbl_db if "frequency" in s]
matching.remove("9999_frequency")
matching = [x for x in matching if not x.startswith('3')]
matching = [x for x in matching if not x.startswith('4')]

i = '1007_frequency'
scaled = features_selection(worktbl_db, tbl_db[i], 10)
tbl_db[i].notnull().astype(int).to_frame().sum() # 7699
X_train, X_test, y_train, y_test = train_test_split(scaled, tbl_db[i].notnull().astype(int).to_frame(), test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=3)
clf = clf.fit(X_train, y_train)


pred = clf.predict(X_test)
#y_test.sum() #1588
pred.sum() #(max_depth=5,class_weight ='balanced'): 1671,DecisionTreeClassifier(max_depth=5):1605,clf = DecisionTreeClassifier(max_depth=5,class_weight ={0:1,1:100}), 2622
print(balanced_accuracy_score(y_test, pred))

def modeltest(model,scaleds,label,parameters):
    x_train, x_test, Y_train, Y_test = train_test_split(scaleds,
                                                        label.notnull().astype(int).to_frame(),
                                                        test_size=0.2, random_state=42)
    grids = GridSearchCV(model, parameters, cv=5, scoring=BCR)
    grids.fit(x_train, Y_train)

    print(grids.best_score_)
    print(grids.best_params_)
    print(grids.best_estimator_)

    print('Results on the test set')

    clfs = grids.best_estimator_
    clfs.fit(x_train, Y_train)

    predi = clfs.predict(x_test)
    bcr_test=balanced_accuracy_score(Y_test, predi)


    return bcr_test,grids.best_estimator_

k_range = list([5,10,15,20,30,50,100,200])
weight_options = ["uniform", "distance"]

param_grid = dict(n_neighbors = k_range, weights = weight_options)
results = pd.DataFrame(columns=["Exercise", "Type_of_algorithm", "Bcr_test"])
for i in matching[5:44]:
    print(i)
    scaled = features_selection(worktbl_db, tbl_db[i], 10)
    result,best_algo = modeltest(KNeighborsClassifier(), scaled, tbl_db[i], param_grid)
    print(result)
    results = results.append({"Exercise": i, "Type_of_algorithm": best_algo,  "Bcr_test": result},
        ignore_index=True)

import sys
from datetime import date
sys.path.append('C:\\Users\cocol\Desktop\memoire\Jéjé_work')
results.to_csv('C:\\Users\cocol\Desktop\memoire\Jéjé_work\KNN\knee-hip'+str(date.today())+".csv")
results.rename(columns={'Bcr_test':'Bcr_test_KNN'},inplace=True)
tabletree = pd.read_csv('C:\\Users\cocol\Desktop\memoire\Jéjé_work\metaparams\metaparam1\FINALTBL2019-11-08.csv')
tabletree.rename(columns={'Bcr_test':'Bcr_test_tree','exercise_number':'Exercise'},inplace=True)

tbl = pd.merge(results, tabletree, on=['Exercise'], how='left')
tbl = tbl[['Exercise','Bcr_test_tree','Bcr_test_KNN']]
tbl['IS_knn_better?'] = (tbl['Bcr_test_KNN'] > tbl['Bcr_test_tree'])
tbl['value_diff?'] = abs(tbl['Bcr_test_KNN'] - tbl['Bcr_test_tree'])
tbl.to_csv('C:\\Users\cocol\Desktop\memoire\Jéjé_work\KNN\knn-tree-diff'+str(date.today())+".csv")