'''This file is the main file to execute, You have to run on you laptop the local mysql database
first and change the variable working directory (witch will be the directory where all the output at csv format will be registered)
second change the 'sytem.path.append(YOURFOLDER)' with the location of the python files'''
from datetime import date
import os
import sys

import pymysql.cursors

sys.path.append('C:\\Users\cocol\Desktop\memoire\Jéjé_work\code')
from Mostimportantfeature import *

Working_Directory = "C:\\Users\cocol\Desktop\memoire\Jéjé_work"





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
        if TAble['String'] is not None and  pd.isna(TAble['String'])==False:
            # split the 1A2A3 fromat into [1 2 3]
            Lst = list(map(int, list(filter(None, TAble['String'].split('A')))))
            for t in range(len(Lst)):
                TAble[Lst[t]] = 1
        return TAble

    # Fill the table composed with only 0 with the corresponding ones with respect to the answers of the patient
    Temporarytbl = Temporarytbl.apply(fastfill, axis=1).drop(['String'], axis=1)

    return Temporarytbl





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

# Get the different columns name for each exercises: frequency, intensity and actual
exsh_column = list(exercise_scheme.columns)

# Merge all teh dataframe to get one big table

patient_daily_data.rename(columns={'diff': 'day'}, inplace=True)


# Store a table with the exercises of the day before
exercise_scheme_of_the_day_before = exercise_scheme.copy()

cols = exercise_scheme_of_the_day_before.columns.drop(
    list(exercise_scheme_of_the_day_before.filter(regex='intensity')) + list(
        exercise_scheme_of_the_day_before.filter(regex='actual')))
exercise_scheme_of_the_day_before = exercise_scheme_of_the_day_before[cols]
exercise_scheme_of_the_day_before = exercise_scheme_of_the_day_before.apply(
    lambda x: x.notnull().astype(int) if "frequency" in x.name else x)

#I would like to go through all the columns in a dataframe and rename (or map) columns if they contain certain strings.
#https://stackoverflow.com/questions/32621677/pandas-rename-column-if-contains-string
exercise_scheme_of_the_day_before.columns = exercise_scheme_of_the_day_before.columns.str.replace('frequency', 'Activated_yesterday')

# Handle the fact that the programation of physio is made from data from the day before
patient_daily_data_of_the_day_before = patient_daily_data.copy()
patient_daily_data_of_the_day_before['day'] += 1
#Add the exercices mades the day before
exercise_scheme_of_the_day_before['day'] += 1
# merge a big table with every data we need
tbl = pd.merge(exercise_scheme, patient_daily_data_of_the_day_before, on=['patient_id', 'day'], how='left')
tbl = pd.merge(patient_dt, tbl, on=['patient_id'], how='right')
tbl = pd.merge(exercise_scheme_of_the_day_before, tbl, on=['patient_id', 'day'], how='right')

# removing patient operated before 1 november 2017, the 3 in the lines of code are because the tbl doesnt contain the day of the operation.
#tbl['date'] = pd.to_datetime(tbl['date'])

#names = tbl['patient_id'][(tbl['date'] <= '2017-11-3')]
#tbl = tbl.loc[~tbl['patient_id'].isin(names.values)]

# Get the different columns name for each exercises: frequency, intensity and actual
exsh_column = list(exercise_scheme.columns)
exsh_column.remove('day')
# Get the number of row of the final frame
nrow = len(tbl[tbl.columns[0]])

# Select from the big talbe the data usefull for the machine learning and drop the labels and patient id
worktbl = tbl.drop(exsh_column, axis=1)


# That part i remove it now but i will add it later because there is an issue with the 1A2A3 format and i will treat ti
worktbl = worktbl.drop(['AcWh1', 'InDo1', 'MeAr1_other', 'MeAr2_other', 'ExWh3', 'WeWh2'], axis=1)

# Transform some columns to a useful format (from string to number)
worktbl["gender"].replace({'Female': 0, 'Male': 1}, inplace=True)
worktbl["MeAr2"].replace(-1.0, np.nan, inplace=True)

worktbl = pd.concat([worktbl.drop(['limb'], axis=1), pd.get_dummies(worktbl['limb'])], axis=1)



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
    if not os.path.isfile(Working_Directory + "\\filled_" + String + ".csv"):
        Newtbl = AAunwrap(Bigtbl, number_of_diffrerent_responses, String)
        Newtbl.to_csv(Working_Directory + "\\filled_" + String + ".csv")
    df1 = pd.read_csv(Working_Directory + "\\filled_" + String + ".csv")
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
matching.remove("9999_frequency")
# Fill the null values
worktbl = worktbl.fillna(method='bfill')
worktbl = worktbl.fillna(method='ffill')

# Select from the big talbe the data usefull for the machine learning and drop the labels and patient id
worktbl = tbl.drop(exsh_column, axis=1)




# That part i remove it now but i will add it later because there is an issue with the 1A2A3 format and i will treat ti
worktbl = worktbl.drop(['AcWh1', 'InDo1', 'MeAr1_other', 'MeAr2_other', 'ExWh3', 'WeWh2'], axis=1)

# Transform some columns to a useful format (from string to number)
worktbl["gender"].replace({'Female': 0, 'Male': 1}, inplace=True)
worktbl["MeAr2"].replace(-1.0, np.nan, inplace=True)

worktbl = pd.concat([worktbl.drop(['limb'], axis=1), pd.get_dummies(worktbl['limb'])], axis=1)



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
    if not os.path.isfile(Working_Directory + "\\filled_" + String + ".csv"):
        Newtbl = AAunwrap(Bigtbl, number_of_diffrerent_responses, String)
        Newtbl.to_csv(Working_Directory + "\\filled_" + String + ".csv")
    df1 = pd.read_csv(Working_Directory + "\\filled_" + String + ".csv")
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
matching.remove("9999_frequency")
# Fill the null values
worktbl = worktbl.fillna(method='bfill')
worktbl = worktbl.fillna(method='ffill')


# Drop unuseful column for machine learning
worktbl = worktbl.drop(['patientnumber', 'date', 'surgery_date'], axis=1)
# ------------------------
# ------------------------
#Results_cv = crossval(matching, mapping_exercises, tbl, worktbl)
# save the Results
#
# Results_cv.to_csv(Working_Directory+"\Results_cv_"+str(date.today())+".csv")

#Results = importfeature(matching,mapping_exercises,tbl,worktbl,mapping_questionnaires, mapping_answers)
# save the Results
#
# Results.to_csv(Working_Directory+"\Results_with_previousdexo"+str(date.today())+".csv")
