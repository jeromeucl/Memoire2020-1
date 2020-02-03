'''This file is the main file to execute, You have to run on you laptop the local mysql database
first and change the variable working directory (witch will be the directory where all the output at csv format will be registered)
second change the 'sytem.path.append(YOURFOLDER)' with the location of the python files'''
__author__ = "Jérôme Dewandre"

'''TO FILL'''
# Are you using the local db?
localdb = True
Working_Directory = "C:\\Users\cocol\Desktop\memoire\Jéjé_work\\test2\\"

from datetime import date
import os
import sys
import pymysql.cursors
# Check if those directorys exist and create them if they don't
from pathlib import Path

Path(Working_Directory).mkdir(parents=True, exist_ok=True)

sys.path.append(Working_Directory)
from Mostimportantfeature import *

from Usefull_function import *

# mysql connection to the cloud
# connection = pymysql.connect(host='173.31.240.35.bc.googleusercontent.com', user='', password='',
#                             db='moveup_dwh', charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
if localdb:
    # mysql connection to the local database
    connection = pymysql.connect(host='127.0.0.1', user='root', password='root', db='moveup_dwh', charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
else:
    # mysql connection to the moveUp database
    connection = pymysql.connect(host='35.240.31.173', user='root', password='aKkidLJ45hdturo3', db='moveup_dwh',
                                 charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)

# Load all the useful dataframes
sql = "select * from exercise_scheme;"
exercise_scheme = read_from_sql_server(connection, sql)

sql = "select * from patient_daily_data;"
patient_daily_data = read_from_sql_server(connection, sql)
# Harmonise the names of the columns
patient_daily_data.rename(columns={'diff': 'day'}, inplace=True)

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

'''This part of the code is aimed to merge the columns of exercises that are exaclty the same between hip and knee'''

merge_exo_list = [[1001, 2001], [1019, 2008], [1012, 2010], [1011, 2009], [1002, 1003]]

for i in merge_exo_list:
    exercise_scheme = merge_exo(i[0], i[1], exercise_scheme)

# Because exercise those exercises are a new and a old versoin of the same exercise, after merging them we can remove one of them
exercise_scheme = exercise_scheme.loc[:, ~exercise_scheme.columns.str.startswith('1003')]

# Those exercises will be removed because not used by the pain_threshold
exercise_scheme = exercise_scheme.loc[:, ~exercise_scheme.columns.str.startswith('1021')]
exercise_scheme = exercise_scheme.loc[:, ~exercise_scheme.columns.str.startswith('2002')]
exercise_scheme = exercise_scheme.loc[:, ~exercise_scheme.columns.str.startswith('2019')]
# Because series 3000,4000 and 9000 are test series of exercises: remove them
exercise_scheme = exercise_scheme.loc[:, ~exercise_scheme.columns.str.startswith('3')]
exercise_scheme = exercise_scheme.loc[:, ~exercise_scheme.columns.str.startswith('4')]
exercise_scheme = exercise_scheme.loc[:, ~exercise_scheme.columns.str.startswith('9')]

# Get the different columns name for each exercises: frequency, intensity and actual
exsh_column = list(exercise_scheme.columns)

'''This part of the code handle the exercises done the day before'''

# Store a table with the exercises of the day before
exercise_scheme_of_the_day_before = exercise_scheme.copy()

# This information might have some interest so i commented the code
'''#Drop columns that contains intensity or actual for each exercise and only keep frequency
cols = exercise_scheme_of_the_day_before.columns.drop(
    list(exercise_scheme_of_the_day_before.filter(regex='intensity')) + list(
        exercise_scheme_of_the_day_before.filter(regex='actual')))
exercise_scheme_of_the_day_before = exercise_scheme_of_the_day_before[cols]'''

# Copy the frequency columns into a new feature because this
for colname in [k for k in exercise_scheme_of_the_day_before.columns if 'frequency' in k]:
    exercise_scheme_of_the_day_before[str(colname.replace("frequency", "FREQ")) + '_yesterday'] = \
    exercise_scheme_of_the_day_before[colname]

# Convert the frequency columns into 0 or 1 (activated or not)
exercise_scheme_of_the_day_before = exercise_scheme_of_the_day_before.apply(
    lambda x: x.notnull().astype(int) if "frequency" in x.name else x)

# I would like to go through all the columns in a dataframe and rename (or map) columns if they contain certain strings.
# https://stackoverflow.com/questions/32621677/pandas-rename-column-if-contains-string


# Replace the in the columns of the exercises activated the day before 'frequency' by 'activated yesterday'
exercise_scheme_of_the_day_before.columns = exercise_scheme_of_the_day_before.columns.str.replace('frequency',
                                                                                                  'Activated_yesterday')
exercise_scheme_of_the_day_before.columns = exercise_scheme_of_the_day_before.columns.str.replace('actual',
                                                                                                  'actual_frequency_yesterday')
exercise_scheme_of_the_day_before.columns = exercise_scheme_of_the_day_before.columns.str.replace('intensity',
                                                                                                  'intensity_yesterday')
# Adapt the day of the exercise of the day before tbl so it fitt with the real exercise sheme table
exercise_scheme_of_the_day_before['day'] += 1

'''This part of the code handle the fact that the programation 
of physio is made from data from the day before and create a tbl with all the usefull information 'tbl' '''
patient_daily_data_of_the_day_before = patient_daily_data.copy()
patient_daily_data_of_the_day_before['day'] += 1

# Merge all the dataframe to get one big table with all data we need
tbl = pd.merge(exercise_scheme, patient_daily_data_of_the_day_before, on=['patient_id', 'day'], how='left')
tbl = pd.merge(patient_dt, tbl, on=['patient_id'], how='right')

# convert the date format to datetime
tbl['date'] = pd.to_datetime(tbl['date'],format='%Y-%m-%d')

'''Some days, exercises are proposed while the patient doesn't fill the daily questionnary'''


tbl = fillmissingDate(tbl)


'''The part below was aimed to seek for differences between the exercises did the day before but was removed'''

'''tbl = pd.merge(exercise_scheme_of_the_day_before, tbl, on=['patient_id', 'day'], how='right')
'''

# Get only the different columns name for each exercises: frequency, intensity and actual inside exsh_column

exsh_column.remove('day')
exsh_column.remove('patient_id')
# Get the number of row of the final frame
nrow = len(tbl[tbl.columns[0]])

# Select from the big talbe the data usefull for the machine learning and drop the labels (from exercise_scheme) and patient id
worktbl = tbl.drop(exsh_column, axis=1)

'''That part drop meaningless variables'''
worktbl = worktbl.drop(['is_basic_strength', 'is_functional_strength', 'is_mobility', 'is_symp_reduction', 'calories'],
                       axis=1)
''' That part of the code is aimed to deal with the 1A2A3 format'''
worktbl = worktbl.drop(['AcWh1', 'InDo1', 'MeAr1_other', 'MeAr2_other', 'ExWh3', 'WeWh2' ], axis=1)
# I also drop 'calories' here


# Transform some columns to a useful format (from string to number)
worktbl["gender"].replace({'Female': 0, 'Male': 1}, inplace=True)
worktbl["MeAr2"].replace(-1.0, np.nan, inplace=True)

#Is it hip or a knee surgery?

#Is it hip (1) or knee (0) surgery?
worktbl = pd.concat([worktbl, pd.DataFrame({'Hip or Knee?':worktbl ['limb'].str.contains("Hip").astype(int)})], axis=1)
# This part build one columns for each possible type of surgery and fill them with 1 (that type of surgery was used) or 0 (that type of surgery wasn't used)
worktbl = pd.concat([worktbl.drop(['limb'], axis=1), pd.get_dummies(worktbl['limb'])], axis=1)

'''Preprocessing string with 1A2A3A fromat'''

# AcWh1 (what's activity did you do today)?
worktbl = add_to_work('AcWh1', worktbl, tbl, 14, Working_Directory, localdb)

# InDo1 (Do you experience swelling in other places than the index joint?)
worktbl = add_to_work('InDo1', worktbl, tbl, 6, Working_Directory, localdb)

#  'ExWh3'Why didn't you do your exercises
worktbl = add_to_work('ExWh3', worktbl, tbl, 4, Working_Directory, localdb)

# 'WeWh2' Why didn't you wear your band all day??
worktbl = add_to_work('WeWh2', worktbl, tbl, 3, Working_Directory, localdb)

# Select all the columns containing frequency in the table with the different exercise as columns for the label of the classification
matching = [s for s in exsh_column if "frequency" in s]

# Fill the null values
worktbl = worktbl.fillna(method='bfill')
worktbl = worktbl.fillna(method='ffill')

# convert the date format to datetime
worktbl['date'] = pd.to_datetime(worktbl['date'])
# Add the trend of the pain

'''This part of the code aim to handle continuous variables and their variation in time'''

thresh = 0
var = 'PaIn1'
trend_tbl = add_trend_to_worktbl(var, thresh, 3, 7, worktbl)
worktbl = pd.concat([worktbl, trend_tbl], axis=1)
worktbl = worktbl.loc[:, ~worktbl.columns.duplicated()]

trend_tbl = add_trend_to_worktbl(var, thresh, 2, 5, worktbl)
worktbl = pd.concat([worktbl, trend_tbl], axis=1)
worktbl = worktbl.loc[:, ~worktbl.columns.duplicated()]

trend_tbl = add_trend_to_worktbl(var, thresh, 3, 10, worktbl)
worktbl = pd.concat([worktbl, trend_tbl], axis=1)
worktbl = worktbl.loc[:, ~worktbl.columns.duplicated()]

var = 'PaIn2'
trend_tbl = add_trend_to_worktbl(var, thresh, 3, 7, worktbl)
worktbl = pd.concat([worktbl, trend_tbl], axis=1)
worktbl = worktbl.loc[:, ~worktbl.columns.duplicated()]

trend_tbl = add_trend_to_worktbl(var, thresh, 2, 5, worktbl)
worktbl = pd.concat([worktbl, trend_tbl], axis=1)
worktbl = worktbl.loc[:, ~worktbl.columns.duplicated()]

trend_tbl = add_trend_to_worktbl(var, thresh, 3, 10, worktbl)
worktbl = pd.concat([worktbl, trend_tbl], axis=1)
worktbl = worktbl.loc[:, ~worktbl.columns.duplicated()]

#Using regression
vartoplot = 'PaIn2'
regtbl = worktbl[['patient_id','day',vartoplot]]
regtbl['linearRegression_'+vartoplot] =0
set(worktbl['patient_id'])

patient = regtbl[regtbl['patient_id']=='zwmXheCSRJkyAjf6J##GttezLQ4XRXtBMLNk'][['day',vartoplot]].reset_index(drop=True)


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression


Y = np.array(patient['PaIn2'].values)
Yhat= savgol_filter(Y, 11,3)
X = np.array(patient['day'].values.reshape(-1,1))
plt.figure(1)
plt.plot(X, Y,  color='gray')
for start in list(range(0,60,5)):
    stop = start + 5
    x = X[start:stop]
    y = Y.reshape(-1,1)[start:stop]
    yhat = Yhat.reshape(-1,1)[start:stop]
    regressor = LinearRegression()
    regressor.fit(x, y) #training the algorithm
    pred = regressor.predict(x)
    regressor = LinearRegression()
    regressor.fit(x, yhat) #training the algorithm
    pred_hat = regressor.predict(x)
    plt.plot(x, pred_hat, color='red', linewidth=2)

plt.show()

plt.figure(1)
plt.scatter(x, y,  color='gray')
plt.plot(x, pred, color='red', linewidth=2)
plt.show()
plt.figure(2)
plt.scatter(x, yhat,  color='gray')
plt.plot(x, pred_hat, color='red', linewidth=2)
plt.show()


'''This part of the code is made for interpretability purpose and replace the names of the variabes by their full names'''
# Replace all names of columns in the worktbl by their full names:
code_names = list(worktbl.columns)

for ft in code_names:
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
                message = message + " ANSWER: " + ans[ans.value_code == int(answer)]['value_text'].values[0] + " "
                worktbl.rename(columns={ft: message}, inplace=True)
            else:
                worktbl.rename(columns={ft: message}, inplace=True)
        else:
            message = message + ft + " "
            worktbl.rename(columns={ft: message}, inplace=True)

    else:
        index3 = find_index(ft, mapping_questionnaires, "question_code")
        if index3 > -1:
            message = message + " " + ft + ": " + mapping_questionnaires['question'][index3] + " "
            worktbl.rename(columns={ft: message}, inplace=True)

# Build a worktbl (able to enter the machine learning algorithm) of patient
worktbl['day'] = worktbl['day'].astype(int)

'''The part below was aimed to seek for differences between the protocol and the decision of the pt but was removed
because the purpose of the thesis changed a bit and we don't aime for a tool to help the pt but we aim for a fully 
automated tool'''

'''#load protocol and merge it with exercise shceme
protocoltbl_hip = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\comparativetbl\protocol\protocol_hip.csv")
protocoltbl_hip.rename(columns={"Days": "day"}, inplace=True)
protocoltbl_knee = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\comparativetbl\protocol\protocol_knee.csv")
protocoltbl_knee.rename(columns={"Days": "day"}, inplace=True)
prot_pt_tbl_hip = pd.merge(exercise_scheme, protocoltbl_hip, on=['day'], how='right')
prot_pt_tbl_knee = pd.merge(exercise_scheme, protocoltbl_knee, on=['day'], how='right')


def compare_protocol_PT(prot_pt_tbl,hip_or_knee):
    Returntbl = prot_pt_tbl[['patient_id','day']].copy()
    Returntbl['day'] = prot_pt_tbl['day']

    exexcise_list = [s for s in prot_pt_tbl.columns if s.isdigit()]
    for ex_number in exexcise_list:
        Returntbl['PT_Protocol_difference_'+hip_or_knee+'_' + ex_number] = (prot_pt_tbl[ex_number + "_frequency"].notnull().astype(int).to_frame()[ex_number + "_frequency"] - prot_pt_tbl[ex_number]).abs()

    return Returntbl

compare_protocol_PT_hip =compare_protocol_PT(prot_pt_tbl_hip,'Hip')
compare_protocol_PT_knee =compare_protocol_PT(prot_pt_tbl_knee,'Knee')

compare_protocol_PT_hip['day'] = compare_protocol_PT_hip['day']+1
compare_protocol_PT_knee['day'] = compare_protocol_PT_knee['day']+1

worktbl = pd.merge(worktbl, compare_protocol_PT_hip, on=['day','patient_id'], how='left')
worktbl = pd.merge(worktbl, compare_protocol_PT_knee, on=['day','patient_id'], how='left')'''

# Fill the null values
worktbl = worktbl.fillna(method='bfill')
worktbl = worktbl.fillna(method='ffill')

'''This part of the code is aimed to make the distinction between hip and knee'''
#Split the exercise of the hip and the exercises of the hip
matchi_knee = [x for x in matching if x.startswith('1')]
matchi_hip = [x for x in matching if x.startswith('2')]

list_of_exercises_knee = [x.replace("_frequency", "") for x in matching if x.startswith('1')]
list_of_exercises_hip = [x.replace("_frequency", "") for x in matching if x.startswith('2')]

# This part of the code is aimed to select all the column specific to the knee/hip in the worktbl
col_specific_knee = [k for k in worktbl.columns if ('Knee' in k) & ('ANSWER: Knee ' not in k) | (
            str([int(s) for s in k.split('_') if s.isdigit()]).replace('[', '').replace(']',
                                                                                        '') in list_of_exercises_knee)]
col_specific_hip = [k for k in worktbl.columns if ('Hip' in k) & ('ANSWER: Hip ' not in k) | (
            str([int(s) for s in k.split('_') if s.isdigit()]).replace('[', '').replace(']',
                                                                                        '') in list_of_exercises_knee)]

# Those table are specific to the knee or the hip
inputML_knee = worktbl.drop(col_specific_hip, axis=1)
inputML_hip = worktbl.drop(col_specific_knee, axis=1)

'''This part of the code is aimed to create the label/output table for classification or regression'''
#Classification
Activation_of_exercise_label = tbl.apply(lambda x: x.notnull().astype(int) if "frequency" in x.name else x)
Activation_of_exercise_label = Activation_of_exercise_label[['patientnumber', 'date', 'patient_id'] + matching]
#Regression
# Select all the columns containing intensity in the table with the different exercise as columns for the label of the classification
matching_intensity = [s for s in exsh_column if "intensity" in s]
intensity_of_exercise_output = tbl[['patientnumber', 'date', 'patient_id'] + matching_intensity]

# Select all the columns containing frequency in the table with the different exercise as columns for the label of the classification
frequency_of_exercise_output = tbl[['patientnumber', 'date', 'patient_id'] + matching]

# ------------------------
# ------------------------
from crossvalidation import crossval
# worktbls = worktbl.drop(['patientnumber', 'date', 'surgery_date', 'patient_id'], axis=1)
# Results_cv = crossval(matching, mapping_exercises, tbl, worktbl)
# save the Results
#
# Results_cv.to_csv(Working_Directory+"\cv\Results_cv_"+str(date.today())+".csv")

# Results = importfeature(matching,mapping_exercises,tbl,worktbls,mapping_questionnaires, mapping_answers)
# save the Results
#
# Results.to_csv(Working_Directory+"\mostimportantfeature\Results_with_previousdexo"+str(date.today())+".csv")


# removing patient operated before 1 november 2017, the 3 in the lines of code are because the tbl doesnt contain the day of the operation.
# tbl['date'] = pd.to_datetime(tbl['date'])

# names = tbl['patient_id'][(tbl['date'] <= '2017-11-3')]
# tbl = tbl.loc[~tbl['patient_id'].isin(names.values)]
