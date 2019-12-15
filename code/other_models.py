#https://gist.github.com/sawansaurabh/3748a083ffdec38aacb8f43660a5f654
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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



'''X_train, X_test, y_train, y_test = train_test_split(scaled, tbl_db['1001_frequency'].notnull().astype(int).to_frame(), test_size=0.2, random_state=42)

k_range = list([5,10,15,20,30,50,100,200])
weight_options = ["uniform", "distance"]

param_grid = dict(n_neighbors = k_range, weights = weight_options)
#print (param_grid)
knn = KNeighborsClassifier()

grid = GridSearchCV(knn, param_grid, cv = 5,scoring=BCR)
grid.fit(X_train,y_train)


print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)


print('Results on the test set')

clf = grid.best_estimator_
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print(balanced_accuracy_score(y_test, pred))'''

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