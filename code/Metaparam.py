import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold  # import KFold

def mgtbl(table,lisT,nameoflist):
    inTH = len(lisT)
    newtbl = pd.DataFrame({'Nb_run': np.tile(Nb_run, inTH), nameoflist: np.repeat(lisT, 10, axis=0)})
    return table.merge(newtbl, on='Nb_run')


Nb_run = list(range(10))
max_depth = [1,2,3,4,5,6,7,8,9,10,15,20]
min_samples_split = [1,2,3,4,5,6,7,8,9,10,15,20]
min_impurity_decrease = [0,0.01,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.4]
t1 = pd.DataFrame({'Nb_run':np.repeat(Nb_run,1,axis=0),'criterion':np.repeat(['entropy'],10, axis=0)})
t2 = pd.DataFrame({'Nb_run':np.repeat(Nb_run,1,axis=0),'criterion':np.repeat(['gini'],10, axis=0)})
t3 = t1.append(t2)
inT = len(max_depth)
t4 = pd.DataFrame({'Nb_run':np.tile(Nb_run,inT),'max_depth':np.repeat(max_depth,10,axis=0)})
t5 = t3.merge(t4, on ='Nb_run' )
inT = len(min_samples_split)
t6 = mgtbl(t5,min_samples_split,'min_samples_split')
t7 = mgtbl(t6,min_samples_split,'min_samples_split')
t8 = mgtbl(t7,min_impurity_decrease,'min_impurity_decrease')

t8['Bcr_test'] = 0
t8['Bcr_train'] = 0



train, test, label_train, label_test = train_test_split(Worktbl, label, random_state=42, test_size=0.2)
def fillinetree(table):


    return table
