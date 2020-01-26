'''This files is aimed to explore the metaparameters for the tree'''
__author__      = "Jérôme Dewandre"

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
#from Machine_learning import worktbl,tbl,matching
def tree_metaparam(worktbl,tbl,matching,workdir):
    worktbl = worktbl.drop(['patientnumber', 'date', 'surgery_date', 'patient_id'], axis=1)

    '''This funciton is aimed to build a table of metaparameters to be explored
    Input: table: the table we need to updtate to add all the combination with the new parameter
        lisT: a list with the value of the new parameter to be explored
        nameoflist: a string name of the new parameter to be explored
    Output: The table merged with list'''
    def mgtbl(table,lisT,nameoflist):
        inTH = len(lisT)
        newtbl = pd.DataFrame({'Nb_run': np.tile(Nb_run, inTH), nameoflist: lisT})
        return table.merge(newtbl, on='Nb_run')



    '''Metaparameters to be explored'''
    max_depth = [1,2,3,5,10,20]

    min_samples_split = [10,20]
    min_impurity_decrease = [0,0.01]

    Nb_run = list(range(1))
    t3 = pd.DataFrame({'Nb_run':0,'criterion':['entropy','gini']})
    t5 = mgtbl(t3,max_depth,'max_depth')
    t6 = mgtbl(t5,min_samples_split,'min_samples_split')
    t8 = mgtbl(t6,min_impurity_decrease,'min_impurity_decrease')
    t8['Bcr_test'] = 0
    t8['Bcr_train'] = 0
    t8['parameters'] = '0'
    t8['iter'] = list(range(len(t8)))

    '''This function is aimed to fill the 'table' exploring all the metaparameters. It uses a StratifiedKFold for more stability 
    Input: table:the table with de metaparameters we need to fill with bcr test and train
        Tbl: the table containing the label
        Worktbl: the talbe we need to train with
        current: a string with the name of the current exercise (example 1001_frequency)
    Output: the table updated'''
    def fillinetree(table, Tbl, Worktbl,current):
        label = Tbl[current].notnull().astype(int).to_frame()
        # Split the data and the label into test and train set with StratifiedKFold
        used_fold = 5
        skf = StratifiedKFold(n_splits=used_fold)

        mean_Bcr_train = 0
        mean_Bcr_test = 0
        for train_index, test_index in skf.split(Worktbl, label):

            train, test = Worktbl.loc[ train_index , : ], Worktbl.loc[test_index,:]
            label_train, label_test = label.loc[train_index,:], label.loc[test_index,:]
            # If this exercise was never used by the physio, don't run the algorithm
            if sum(label_train.values) != 0:
                # Train prediction
                clf = tree.DecisionTreeClassifier(max_depth=table['max_depth'], criterion=table['criterion'],
                                                  min_samples_split=table['min_samples_split'],
                                                  min_impurity_decrease=table['min_impurity_decrease'],
                                                  class_weight='balanced')
                clf = clf.fit(train, label_train)

                # Predict the label for train set
                train_pred = clf.predict(train)

                bcr_train = balanced_accuracy_score(label_train, train_pred)
                mean_Bcr_train = mean_Bcr_train + bcr_train

                # Test prediction with the model build on the train set
                test_pred = clf.predict(test)

                bcr_test = balanced_accuracy_score(label_test, test_pred)
                mean_Bcr_test = mean_Bcr_test + bcr_test
            else:
                mean_Bcr_train = mean_Bcr_train + 0.5

                mean_Bcr_test = mean_Bcr_test + 0.5

            # Bcr encoding
            table['Bcr_train'] = mean_Bcr_train/used_fold

            table['Bcr_test'] = mean_Bcr_test/used_fold
            table['parameters'] = clf.get_params()
        print(str(table['iter']) +' :'+str(current) + " :" + str(table['Bcr_test']))

        return table

    '''loop accross all esxercises'''
    for meta in range(0,len(matching)):
        curr = matching[meta]

        t9 = t8.copy()

        t9 = t9.apply(fillinetree,args = (tbl,worktbl,curr), axis=1)
        t9.to_csv(workdir+"metaparam\met"+str(curr)+".csv")
