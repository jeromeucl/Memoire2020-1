'''This file aim to export all models selected with metatreatment'''
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from Machine_learning import worktbl,tbl,matching
import pickle

worktbl = worktbl.drop(['patientnumber', 'date', 'surgery_date', 'patient_id'], axis=1)


model_param = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\metaparam\FINALTBL2019-12-30.csv")

for exo in matching:

    label = tbl[exo].notnull().astype(int).to_frame()
    # Split the data and the label into test and train set
    train, test, label_train, label_test = train_test_split(worktbl, label, test_size=0.2)
    # If this exercise was never used by the physio, don't run the algorithm
    
    if sum(label_train.values) != 0:
        # Train prediction
        param = model_param[model_param['exercise_number'] == exo]
        clf = tree.DecisionTreeClassifier(max_depth=param['max_depth'].values[0],criterion=param['criterion'].values[0],min_samples_split=int(param['min_samples_split'].values[0]),min_impurity_decrease=param['min_impurity_decrease'].values[0],class_weight ='balanced')
        clf = clf.fit(train, label_train)
        # Predict the label for train set
        pickle.dump(clf, open("modeltoexport\\modelfor_"+str(exo)+".sav",'wb'))
    else:
        print('Issue with label_train.values')


