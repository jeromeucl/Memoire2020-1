import pandas as pd
from Machine_learning import matching
import numpy as np
from sklearn import tree
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from Machine_learning import worktbl,tbl,matching
import pickle
import pydotplus
import graphviz
from IPython.display import Image
import collections

matching.remove('4011_frequency')
matching = [x for x in matching if not x.startswith('3')]
#model_param = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\metaparam\FINALTBL2019-10-14.csv")
model_param = pd.read_csv("FINALTBL2019-10-14.csv")
#matching = ["1001_frequency"]
for exo in matching:

    label = tbl[exo].notnull().astype(int).to_frame()
    # Split the data and the label into test and train set
    train, test, label_train, label_test = train_test_split(worktbl, label, test_size=0.2)
    # If this exercise was never used by the physio, don't run the algorithm
    if sum(label_train.values) != 0:
        # Train prediction
        param = model_param[model_param['exercise_number'] == exo]
        clf = tree.DecisionTreeClassifier(max_depth=param['max_depth'].values[0],criterion=param['criterion'].values[0],min_samples_split=int(param['min_samples_split'].values[0]),min_impurity_decrease=param['min_impurity_decrease'].values[0])
        clf = clf.fit(train, label_train)
        # Predict the label for train set
        pickle.dump(clf, open("modeltoexport\\modelfor_"+str(exo)+".sav",'wb'))
    else:
        print('Issue with label_train.values')

clf2 = pickle.load(open("modeltoexport\\modelfor_1001_frequency.sav", 'rb'))
#test_pred = clf2.predict(test)



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
                node.set_fillcolor('green')
            else:
                node.set_fillcolor('red')
        else:
            node.set_fillcolor('white')
graph.write_png('colored_tree.png')
