__author__      = "Jérôme Dewandre"
import pandas as pd
import numpy as np
import os
''' This fuction handle the '1A2A3' fromat to usefull features for the machine learning algorithm and split the cell
    into a certain number of column equals to the number_of_diffrerent_responses to the question asked and fill the column
    with one if the patient answered yes to a given question and with 0 otherwise. For example if there is 6 possibilities of
    answers (0, 1, 2, 3, 4, 5) and the cells show '1A2A3' you will have a [ 0 1 1 1 0 0] for this row.
Input: Table: A table with a column containing the 1A2A3 format
       number_of_different_responses: the number of different possibility of answers to the question asked to the patient (ex: 8)
       STring: the string reffered to the column in 'Table' (ex: 'AcHw1')
Output : A dataframe where each row contains 0 or 1 corresponding to the respective '1A2A3' format of the imput'''


def AAunwrap(Table, number_of_diffrerent_responses, STring):
    cname = [STring + '$' + str(i) for i in list(range(0, number_of_diffrerent_responses))]
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

'''This function uses a sql statement and a connection to a database to return a dataframe according to the sql_statement
from the datase in "Connection" '''

def read_from_sql_server(Connection, sql_statement):
    return pd.read_sql(sql_statement, Connection, index_col=None, coerce_float=True, params=None, parse_dates=None,
                       columns=None,
                       chunksize=None)

''''The function merge_exo take as argument two exercises number (example 1001 and 2001) from the tbl table and merge them so it works with the rest of the
code and store it in the table df without changing the name of the old columns
Input: df: 
       exo1: an int representing an exercise to merge
       exo2: and int representing an exercise to merge
Output : A dataframe where each exercise is the result of the merge of the two exercises'''

def merge_exo(ex1,ex2,df):
    dataframe = df.copy()
    strexo1 = str(ex1)+'_frequency'
    strexo2 = str(ex2) + '_frequency'
    df = dataframe[[strexo1,strexo2]]
    newcolumn = pd.DataFrame({str(ex1)+'+'+strexo2:df[strexo1].notna() | df[strexo2].notna()}).astype(float).replace(0, np.nan)

    if ex2==2010:
        #only update exo for heel raise for hip (and not knee surgery) because it impove bcr
        dataframe[strexo2] = newcolumn
    else :
        dataframe[strexo1] = newcolumn

        dataframe[strexo2] = newcolumn
    return dataframe

'''This function add to the workTbl a feature form the Bigtbl under the 1A2A3 fromat under the shape of multiple
columns filled with 0 or 1
Input : String: The name of the column in the Bigtbl
        worktbl: The table in which the features are added
        Bigtbl: A talbe with 1A2A3 format columns
        number_of_diffrerent_responses: for the question String, several possible answers exist,
        number_of_diffrerent_responses is the number of possible answers
        Working_Directory: directory where the tables handling the 1A2A3 fromat are
        redoAAunwrap: Boolean value, False if you are in development, True if you use a new workTbl
Output : the above worktbl modified
'''
def add_to_work(String, workTbl, Bigtbl, number_of_diffrerent_responses,Working_Directory,local):
    if (not local) & (os.path.exists(Working_Directory + "\\filled_" + String + ".csv")):
        os.remove(Working_Directory + "\\filled_" + String + ".csv")
    if not os.path.isfile(Working_Directory + "\\filled_" + String + ".csv"):
        Newtbl = AAunwrap(Bigtbl, number_of_diffrerent_responses, String)
        Newtbl.to_csv(Working_Directory + "\\filled_" + String + ".csv")

    df1 = pd.read_csv(Working_Directory + "\\filled_" + String + ".csv")
    workTbl = pd.concat([workTbl, df1], axis=1, sort=False)
    return workTbl.drop(['Unnamed: 0'], axis=1)

'''this function return a table that can be concatenated with the worktbl and contain trend of information about continous variable 
as the pain, the 'threshold' is the significant level of difference you need to asses. The "number of past day 1" is the numer of days on which you want to
compute the first average that will be compared with the average based on number_of_past_days2. number_of_past_days1 need to be greater than number_of_past_days2
Input : Variable: a continuous variable (for example PaIn2)
        threshold: the significant level of difference you need to asses
        number_of_past_days1: numer of days on which you want to compute the first average
        number_of_past_days2: numer of days on which you want to compute the second average
        Worktbl: a table containing the day, the date, the patientnumber and the value for the variable
Output : a new table containing a table containing the day, the date, the patientnumber and 
 a binary value saying if the difference between the first and the second average decreased and another binary value saying
 if the difference between the first and the second average increased'''
def add_trend_to_worktbl(Variable,threshold,number_of_past_days1,number_of_past_days2,Worktbl):

    if number_of_past_days1 < number_of_past_days2:
        Paintbl = Worktbl[['patientnumber','date','day',Variable]]

        df = Paintbl.groupby('patientnumber').apply(lambda x: x.set_index('date').resample('1D').first())

        df1 = df.groupby(level=0)[Variable].apply(lambda x: x.shift().rolling(min_periods=1,window=number_of_past_days1).mean()).reset_index(name=Variable +'_Average_Past_'+str(number_of_past_days1)+'_days')
        medged_tbl = pd.merge(Paintbl, df1, on=['date', 'patientnumber'], how='left')


        df2 = df.groupby(level=0)[Variable].apply(lambda x: x.shift().rolling(min_periods=1,window=number_of_past_days2).mean()).reset_index(name=Variable +'_Average_Past_'+str(number_of_past_days2)+'_days')
        medged_tbl = pd.merge(medged_tbl, df2, on=['date', 'patientnumber'], how='left')
        medged_tbl[str('Average_pain_increase_for_' + Variable+'_between_'+str(number_of_past_days1)+'and_'+str(number_of_past_days2)+'_previousdays')] = np.where(
                medged_tbl[Variable +'_Average_Past_'+str(number_of_past_days2)+'_days'] - medged_tbl[str(Variable +'_Average_Past_'+str(number_of_past_days1)+'_days')] < -threshold, 1, 0)

        medged_tbl[str('Average_pain_decrease_for_' + Variable+'_between_'+str(number_of_past_days1)+'and_'+str(number_of_past_days2)+'_previousdays')] = np.where(
            medged_tbl[Variable + '_Average_Past_' + str(number_of_past_days2) + '_days'] - medged_tbl[
                str(Variable + '_Average_Past_' + str(number_of_past_days1) + '_days')] > threshold, 1, 0)
        medged_tbl = medged_tbl.drop([Variable +'_Average_Past_'+str(number_of_past_days1)+'_days',Variable +'_Average_Past_'+str(number_of_past_days2)+'_days'], axis=1)
    else :
        medged_tbl =0
        print('Wrong order of pain average days (number_of_past_days 1 and 2)')
    return medged_tbl