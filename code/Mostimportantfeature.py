import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split


# Python code to merge dict using a single
# expression
def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

'''Find in a string a dollar, return true if there is one and fasle otherwise'''


def find_dolar(String):
    return String.find('$') > -1


'''This function find the index of the first occurrence (and only the first if there are more)
of a string in a certain column of a dataframe
Input : String: the string we are searching for
        DataFrame: the dataframe with multiple columns
        name_of_column: the column in the dataframe where we will search for the String in argument
Output : The index of the fist occurrence of the String in the DataFarme at column name_of_column if there is one,
        otherwise return -1
'''
def find_index(String, Dataframe, name_of_column):
    index = Dataframe.index[Dataframe[name_of_column] == String].tolist()
    if len(index) > 0:
        return index[0]
    return -1


'''This function return all the indexes of a string in a certain column of a dataframe
Input : String: the string we are searching for
        DataFrame: the dataframe with multiple columns
        name_of_column: the column in the dataframe where we will search for the String in argument
Output : All the indexes of the occurrences of the String in the DataFarme at column name_of_column
'''
def return_index(String, Dataframe, name_of_column):
    index = Dataframe.index[Dataframe[name_of_column] == String].tolist()
    return index

def importfeature(Matching,Mapping_exercises,Big_tbl,Worktbl,Mapping_questionnaires, Mapping_answers):
    # nb_of_best_features printed in results
    nb_of_best_features = 10
    Important_features_list = ["Important_feature_" + str(s + 1) for s in range(nb_of_best_features)]
    # Build a dataframe for the results
    results = pd.DataFrame(columns=["Exercise","Number_of_exo", "Type_of_algorithm", "Bcr_train", "Bcr_test"] + Important_features_list)

    '''Main loop'''
    # Pich one column corresponding to an exercise at a time and make it the label
    for exercise_number in Matching:
        # Extract the number of the exercise (example: 1001)
        number_of_exercise = exercise_number.replace("_frequency", "")
        # Extract the full name of the exercise (example: Exercise 1 (K): Circulation )
        name_of_exercise = Mapping_exercises['name'][
            Mapping_exercises.index[Mapping_exercises['number'] == int(number_of_exercise)].tolist()].values[0]
        # Create the label for the machine learning algorithm
        label = Big_tbl[exercise_number].notnull().astype(int).to_frame()
        # Split the data and the label into test and train set
        train, test, label_train, label_test = train_test_split(Worktbl, label, random_state=42, test_size=0.2)

        # If this exercise was never used by the physio, don't run the algorithm
        if sum(label_train.values) == 0:
            results = results.append(
                Merge({"Exercise": name_of_exercise,"Number_of_exo":number_of_exercise, "Type_of_algorithm": "Not_attempted", "Bcr_train": 0, "Bcr_test": 0},
                      {"Important_feature_" + str(s + 1): np.nan for s in range(nb_of_best_features)}),
                ignore_index=True)
        else:
            # Train prediction
            clf = tree.DecisionTreeClassifier(max_depth=3,class_weight ='balanced')
            clf = clf.fit(train, label_train)
            # Get the most important feature
            importances = clf.feature_importances_
            # ([-3:] because you need to take the last 20 elements of the array since argsort sorts in ascending order)

            best_feature = list(Worktbl.columns[np.flip(np.argsort(importances)[-nb_of_best_features:])])
            # Create a message with the most important feature
            message_filled = {}
            i = 0
            for ft in best_feature:
                i = i + 1
                message = ''
                if find_dolar(ft):
                    feature_code, answer = ft.split("$")
                    index1 = find_index(feature_code, Mapping_questionnaires, "question_code")
                    if index1 > -1:
                        message = message + " " + feature_code + ": " + Mapping_questionnaires['question'][index1] + " "
                        index2 = find_index(feature_code, Mapping_answers, "question_code")
                        if index2 > -1:
                            positions = return_index(feature_code, Mapping_answers, "question_code")
                            ans = Mapping_answers[['value_text', 'value_code']].iloc[positions]
                            message = message + " ANSWER: " + ans[ans.value_code == int(answer)]['value_text'].values[
                                0] + " "
                    else:
                        message = message + ft + " "
                else:
                    index3 = find_index(ft, Mapping_questionnaires, "question_code")
                    if index3 > -1:
                        message = message + " " + ft + ": " + Mapping_questionnaires['question'][index3] + " "
                    else:
                        message = message + ft + " "
                message_filled = Merge(message_filled,{"Important_feature_" + str(i):message})

            # Predict the label for train set
            train_pred = clf.predict(train)

            #Bcr calculationn
            bcr_train = balanced_accuracy_score(label_train, train_pred)


            # Test prediction with the model build on the train set
            test_pred = clf.predict(test)

            bcr_test = balanced_accuracy_score(label_test, test_pred)

            # Add everinthing to the Result table
            results = results.append(
                Merge({"Exercise": name_of_exercise,"Number_of_exo":number_of_exercise, "Type_of_algorithm": "Tree", "Bcr_train": bcr_train,
                       "Bcr_test": bcr_test}, message_filled), ignore_index=True)


    return results
