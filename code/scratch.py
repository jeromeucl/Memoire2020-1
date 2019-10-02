import os

import numpy as np
import pandas as pd
import pymysql.cursors
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

Working_Directory = "C:\\Users\cocol\Desktop\memoire\Jéjé_work"

# Python code to merge dict using a single
# expression
def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res
# BCR formula (Balanced classificatoin ratio) to handle class imbalance
def bcr(Tn, Fp, Fn, Tp):
    return 0.5 * ((Tp / (Tp + Fn)) + (Tn / (Fp + Tn)))


''' This fuction handle the '1A2A3' fromat to usefull features for the machine learning algorithm and split the cell
    into a certain number of column equals to the number_of_diffrerent_responses to the question asked and fill the column
    with one if the patient answered yes to a given question and with 0 otherwise. For example if there is 6 possibilities of
    answers (0, 1, 2, 3, 4, 5) and the cells show '1A2A3' you will have a [ 0 1 1 1 0 0] for this row.
Input: Table: A table with a column containing the 1A2A3 format
       number_of_different_responses: the number of different possibility of answers to the question asked to the patient (ex: 8)
       STring: the string reffered to the column in 'Table' (ex: 'AcHw1')
Output : A dataframe where each row contains 0 or 1 corresponding to the respective '1A2A3' format of the imput'''


def AAunwrap(Table, number_of_diffrerent_responses, STring):
    cname = [STring + '$' + str(i) for i in list(range(0, number_of_diffrerent_responses + 1))]
    nbrow = len(Table[Table.columns[0]])
    Temporarytbl = pd.DataFrame(np.zeros(shape=(nbrow, len(cname)), dtype=int), columns=cname)

    Temporarytbl['String'] = Table[STring]
    '''This fuction is used for speed up everinthing and work with the apply function below'''

    def fastfill(TAble):
        if TAble['String'] is not None:
            # split the 1A2A3 fromat into [1 2 3]
            Lst = list(map(int, list(filter(None, TAble['String'].split('A')))))
            for t in range(len(Lst)):
                TAble[Lst[t]] = 1
        return TAble

    # Fill the table composed with only 0 with the corresponding ones with respect to the answers of the patient
    Temporarytbl = Temporarytbl.apply(fastfill, axis=1).drop(['String'], axis=1)

    return Temporarytbl


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


# mysql connection to the cloud
# connection = pymysql.connect(host='173.31.240.35.bc.googleusercontent.com', user='', password='',
#                             db='moveup_dwh', charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)

# mysql connection to the local database
connection = pymysql.connect(host='127.0.0.1', user='root', password='root',
                             db='moveup_dwh', charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
# Select the exercise_scheme table and patient_daily_data table to build one big table useful for the machine learning
# Each row is a patient at a given day
'''This function uses a sql statement and a connection to a database to return a dataframe according to the sql_statement'''


def read_from_sql_server(Connection, sql_statement):
    return pd.read_sql(sql_statement, Connection, index_col=None, coerce_float=True, params=None, parse_dates=None,
                       columns=None,
                       chunksize=None)


# Load all the useful dataframe
sql = "select * from exercise_scheme;"
exercise_scheme = read_from_sql_server(connection, sql)

sql = "select * from patient_daily_data;"
patient_daily_data = read_from_sql_server(connection, sql)

sql = "SELECT * FROM moveup_dwh.mapping_exercises;"
mapping_exercises = read_from_sql_server(connection, sql)

sql = "SELECT * FROM moveup_dwh.mapping_questionnaires;"
mapping_questionnaires = read_from_sql_server(connection, sql)
sql = "SELECT * FROM moveup_dwh.mapping_answers;"
mapping_answers = read_from_sql_server(connection, sql)

sql = "SELECT * FROM moveup_dwh.patients;"
patient_data = read_from_sql_server(connection, sql)

patient_data.rename(columns={'id': 'patient_id'}, inplace=True)
patient_dt = patient_data[['patient_id', 'age', 'gender', 'limb']]

# Merge all teh dataframe to get one big table

patient_daily_data.rename(columns={'diff': 'day'}, inplace=True)
# Handle the fact that the programation of physio is made from data from the day before
exercise_scheme['day'] += 1
tbl = pd.merge(exercise_scheme, patient_daily_data, on=['patient_id', 'day'], how='inner')
tbl = pd.merge(patient_dt, tbl, on=['patient_id'], how='inner')

# Get the different columns name for each exercises: frequency, intensity and actual
exsh_column = list(exercise_scheme.columns)

# Get the number of row of the final frame
nrow = len(tbl[tbl.columns[0]])

# Select from the big talbe the data usefull for the machine learning and drop the labels
worktbl = tbl.drop(exsh_column, axis=1)

# Drop unuseful column for machine learning
worktbl = worktbl.drop(['patientnumber', 'date', 'surgery_date'], axis=1)

# That part i remove it now but i will add it later because there is an issue with the 1A2A3 format and i will treat ti
worktbl = worktbl.drop(['AcWh1', 'InDo1', 'MeAr1_other', 'MeAr2_other', 'ExWh3', 'WeWh2'], axis=1)

# Transform some columns to a useful format (from string to number)
worktbl["gender"].replace({'Female': 0, 'Male': 1}, inplace=True)
worktbl['limb'].replace(np.nan, -1)
worktbl['limb'], _ = pd.factorize(worktbl['limb'])

# Fill the null values
worktbl = worktbl.fillna(method='bfill')
worktbl = worktbl.fillna(method='ffill')

'''Preprocessing string with 1A2A3A fromat'''

'''This function add to the workTbl a feature form the Bigtbl under the 1A2A3 fromat under the shape of multiple
columns filled with 0 or 1
Input : String: The name of the column in the Bigtbl
        worktbl: The table in which the features are added
        Bigtbl: A talbe with 1A2A3 format columns
        number_of_diffrerent_responses: for the question String, several possible answers exist,
        number_of_diffrerent_responses is the number of possible answers
Output : the above worktbl modified
'''
def add_to_work(String, workTbl, Bigtbl, number_of_diffrerent_responses):
    if not os.path.isfile(Working_Directory+"\\filled_" + String + ".csv"):
        Newtbl = AAunwrap(Bigtbl, number_of_diffrerent_responses, String)
        Newtbl.to_csv(Working_Directory+"\\filled_" + String + ".csv")
    df1 = pd.read_csv(Working_Directory+"\\filled_" + String + ".csv")
    workTbl = pd.concat([workTbl, df1], axis=1, sort=False)
    return workTbl.drop(['Unnamed: 0'], axis=1)


# AcWh1 (what's activity did you do today)?
worktbl = add_to_work('AcWh1', worktbl, tbl, 14)

# InDo1 (Do you experience swelling in other places than the index joint?  )
worktbl = add_to_work('InDo1', worktbl, tbl, 6)

#  'ExWh3'Why didn't you do your exercises
worktbl = add_to_work('ExWh3', worktbl, tbl, 4)

# 'WeWh2' Why didn't you wear your band all day??
worktbl = add_to_work('WeWh2', worktbl, tbl, 3)

# Select all the columns containing frequency in the table with the different exercise as columns for the label
matching = [s for s in exsh_column if "frequency" in s]

#------------------------
# nb_of_best_features printed in results
nb_of_best_features = 4
Important_features_list = ["Important_feature_" + str(s + 1) for s in range(nb_of_best_features)]
# Build a dataframe for the results
Results = pd.DataFrame(columns=["Exercise", "Type_of_algorithm", "Bcr_train", "Bcr_test"] + Important_features_list)

'''Main loop'''
# Pich one column corresponding to an exercise at a time and make it the label
for exercise_number in matching:
    # Extract the number of the exercise (example: 1001)
    name_of_exercise = exercise_number.replace("_frequency", "")
    # Extract the full name of the exercise (example: Exercise 1 (K): Circulation )
    name_of_exercise = mapping_exercises['name'][
        mapping_exercises.index[mapping_exercises['number'] == int(name_of_exercise)].tolist()].values[0]
    # Create the label for the machine learning algorithm
    label = tbl[exercise_number].notnull().astype(int).to_frame()
    # Split the data and the label into test and train set
    train, test, label_train, label_test = train_test_split(worktbl, label, random_state=42, test_size=0.2)

    # If this exercise was never used by the physio, don't run the algorithm
    if sum(label.values) == 0:
        Results = Results.append(
            Merge({"Exercise": name_of_exercise, "Type_of_algorithm": "Not_attempted", "Bcr_train": 0, "Bcr_test": 0},
                  {"Important_feature_" + str(s + 1): np.nan for s in range(nb_of_best_features)}),
            ignore_index=True)
    else:
        # Train prediction
        clf = tree.DecisionTreeClassifier(max_depth=5)
        clf = clf.fit(train, label_train)
        # Get the most important feature
        importances = clf.feature_importances_
        # ([-3:] because you need to take the last 20 elements of the array since argsort sorts in ascending order)

        best_feature = list(worktbl.columns[np.flip(np.argsort(importances)[-nb_of_best_features:])])
        # Create a message with the most important feature
        message_filled = {}
        i = 0
        for ft in best_feature:
            i = i + 1
            message = ''
            if find_dolar(ft):
                feature_code, answer = ft.split("$")
                index1 = find_index(feature_code, mapping_questionnaires, "question_code")
                if index1 > -1:
                    message = message + " " + feature_code + ": " + mapping_questionnaires['question'][index1] + " "
                    index2 = find_index(feature_code, mapping_answers, "question_code")
                    if index2 > -1:
                        positions = return_index(feature_code, mapping_answers, "question_code")
                        ans = mapping_answers[['value_text', 'value_code']].iloc[positions]
                        message = message + " ANSWER: " + ans[ans.value_code == int(answer)]['value_text'].values[
                            0] + " "
                else:
                    message = message + ft + " "
            else:
                index3 = find_index(ft, mapping_questionnaires, "question_code")
                if index3 > -1:
                    message = message + " " + ft + ": " + mapping_questionnaires['question'][index3] + " "
                else:
                    message = message + ft + " "
            message_filled = Merge(message_filled,{"Important_feature_" + str(i):message})

        # Predict the label for train set
        train_pred = clf.predict(train)
        # confusion_matrix(y_true, y_pred)
        tn1, fp1, fn1, tp1 = confusion_matrix(label_train, train_pred, labels=[0, 1]).ravel()
        bcr_train = bcr(tn1, fp1, fn1, tp1)

        # Test prediction with the model build on the train set
        test_pred = clf.predict(test)
        # confusion_matrix(y_true, y_pred)
        tn2, fp2, fn2, tp2 = confusion_matrix(label_test, test_pred, labels=[0, 1]).ravel()
        bcr_test = bcr(tn2, fp2, fn2, tp2)
        # Add everinthing to the Result table
        Results = Results.append(
            Merge({"Exercise": name_of_exercise, "Type_of_algorithm": "Tree", "Bcr_train": bcr_train,
                   "Bcr_test": bcr_test}, message_filled), ignore_index=True)



# save the Results
#
#Results.to_csv(Working_Directory+"\Results_"+str(date.today())+".csv")
