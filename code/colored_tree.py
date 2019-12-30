import pickle
from sklearn import tree
from Machine_learning import *
import pydotplus



#https://gist.github.com/sawansaurabh/3748a083ffdec38aacb8f43660a5f654
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

from sklearn.metrics import balanced_accuracy_score, make_scorer

BCR = make_scorer(balanced_accuracy_score)


from Machine_learning import worktbl, tbl, matching
worktbl = worktbl.drop(['patientnumber', 'date', 'surgery_date', 'patient_id'], axis=1)

results= pd.DataFrame(columns=["Exercise", "Type_of_algorithm", "Bcr_train", "Bcr_test",'best_feature'])
images = 'from_metaparam'
for exo in matching:
    if images == 'from_metaparam':
        clf2 = pickle.load(open("modeltoexport\\modelfor_"+str(exo)+".sav", 'rb'))
    else :
        clf2 = tree.DecisionTreeClassifier(max_depth=1, class_weight='balanced')

        x_train, x_test, Y_train, Y_test = train_test_split(worktbl,
                                                            tbl[exo].notnull().astype(int).to_frame(),
                                                            test_size=0.2, random_state=42)
        clf2 = clf2.fit(x_train, Y_train)
        predi = clf2.predict(x_test)
        bcr_test = balanced_accuracy_score(Y_test, predi)
        predi = clf2.predict(x_train)
        bcr_train = balanced_accuracy_score(Y_train, predi)

        # Get the most important feature
        importances = clf2.feature_importances_
        # ([-3:] because you need to take the last 20 elements of the array since argsort sorts in ascending order)

        best_feature = list(worktbl.columns[np.flip(np.argsort(importances)[-5:])])

        results = results.append({"Exercise": exo.split('_')[0],
                                  "Type_of_algorithm": tree.DecisionTreeClassifier(max_depth=3,
                                                                                   class_weight='balanced'),
                                  "Bcr_test": bcr_test, "Bcr_train": bcr_train, 'best_feature': best_feature},
                                 ignore_index=True)

    # Create DOT data
    dot_data = tree.export_graphviz(clf2,impurity=False, out_file=None,
                                    feature_names=list(worktbl.columns),
                                    class_names=matching[1],filled=True,
                                    rounded=True,
                                    special_characters=True)

    # Draw graph
    graph = pydotplus.graph_from_dot_data(dot_data)

    # Show graph
    nodes = graph.get_node_list()
    #nodes = graph.get_edge_list()
    for node in nodes:
        if node.get_label():
            if node.get_label().split("samples = ")[0]=='<':
                if node.get_label().split('class = ')[1].split('>')[0] =='1':
                    node.set_fillcolor('red')
                else:
                    node.set_fillcolor('green')
            else:
                node.set_fillcolor('white')
    graph.write_png('C:\\Users\cocol\Desktop\memoire\Jéjé_work\\tree_per_exo_dislays\\tree_for'+str(exo)+'.png')
#results.to_csv('C:\\Users\cocol\Desktop\memoire\Jéjé_work\\tree_per_exo_dislays\\len1\\results.csv')