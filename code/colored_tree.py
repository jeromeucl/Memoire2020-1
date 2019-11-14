import pickle
from sklearn import tree
from Machine_learning import *
import pydotplus
matching.remove('4011_frequency')
matching = [x for x in matching if not x.startswith('3')]
worktbl = worktbl.drop(['patientnumber', 'date', 'surgery_date', 'patient_id'], axis=1)
for exo in matching:
    clf2 = pickle.load(open("modeltoexport\\modelfor_"+str(exo)+".sav", 'rb'))



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
    graph.write_png('C:\\Users\cocol\Desktop\memoire\Jéjé_work\\tree_per_exo_dislay\\tree_for'+str(exo)+'.png')