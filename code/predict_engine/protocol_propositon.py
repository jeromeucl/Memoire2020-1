__author__      = "Jérôme Dewandre"
'''This file create a table with the prediction for each exercise and the exercise proposed by the physios
it also check for the number of exercises proposed each days. The last part of the file compare the result of the models 
with the results of the protocol to be able to set flags if there is a difference'''
import pandas as pd
import pickle
from datetime import datetime, timedelta


from predict_engine.Create_imput_function import getimput,getexercise_Of_date
from Machine_learning import tbl,worktbl,matching, Working_Directory, localdb
from datetime import date
#Select the patient for with you want to predict data with a startdate and set the location of the models previously built

Location_of_the_models = Working_Directory +'models'


'''Are we in developpement mode?'''
#Yes
if localdb:
    startdate = '2019-8-12'
#NO
else:
    startdate = str(date.today())


# load imput data
imput_worktbl,imput_tbl = getimput(worktbl,tbl,startdate)


# load protocol proposition
protocoltbl_hip = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\comparativetbl\protocol\protocol_hip.csv")
protocoltbl_hip.rename(columns={"Days": "day"}, inplace=True)
protocoltbl_knee = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\comparativetbl\protocol\protocol_knee.csv")
protocoltbl_knee.rename(columns={"Days": "day"}, inplace=True)

# Spilt the patient with hip and knee treatment
hip_patient = imput_tbl.loc[imput_tbl['limb'].str.contains("Hip")][['patient_id']].reset_index().drop('index', axis=1)
knee_patient = imput_tbl.loc[imput_tbl['limb'].str.contains("Knee")][['patient_id']].reset_index().drop('index', axis=1)

# Spilt the worktbl (input for machine learning) with hip and knee patients
knee_worktbl = imput_worktbl[imput_worktbl['patient_id'].isin(list(knee_patient['patient_id'].unique()))].reset_index().drop('index', axis=1)
hip_worktbl = imput_worktbl[imput_worktbl['patient_id'].isin(list(hip_patient['patient_id'].unique()))].reset_index().drop('index', axis=1)

# Spilt the worktbl (input for machine learning) with hip and knee patients and merge it with the exercises of the protocol
prot_pt_tbl_hip = pd.merge(hip_worktbl, protocoltbl_hip, on=['day'], how='left')
prot_pt_tbl_knee = pd.merge(knee_worktbl, protocoltbl_knee, on=['day'], how='left')


#Split the exercise of the hip and the exercises of the hip
matchi_knee = [x for x in matching if x.startswith('1')]
matchi_hip = [x for x in matching if x.startswith('2')]

'''This function is aimed to predict for each patient each day which exercise will or will not be proposed
Input : 
        matchings: exercise list for knee or hip patients
        input: input for machine learning with hip or knee patients
Output : Returntbl: the table of prediciton for each days
'''
def proposition_tbl(matchings,input):
    Returntbl = input[['patient_id', 'patientnumber','date']].copy()
    Returntbl['day'] = input['day']
    for ex_number in matchings:
        clf2 = pickle.load(open(Location_of_the_models+"\modelfor_" + str(ex_number) + ".sav",'rb'))
        workingtbl = input.drop(['patientnumber', 'date', 'surgery_date', 'patient_id'], axis=1).copy()
        Returntbl[str(ex_number).replace('_frequency','')+'_model'] = list(clf2.predict(workingtbl))
    return Returntbl

from datetime import date

'''This part of the code run the proposition_tbl and check the number of exercises each days'''
tablehip = proposition_tbl(matchi_hip,hip_worktbl)
spike_cols = [col for col in tablehip.columns if 'model' in col]
tablehip['number_of_proposed_ex'] = tablehip[spike_cols].sum(axis=1)
#tablehip.to_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\exercise_proposition\Hip_exos_proposition_"+str(date.today())+".csv")

tableknee = proposition_tbl(matchi_knee,knee_worktbl)
spike_cols = [col for col in tableknee.columns if 'model' in col]
tableknee['number_of_proposed_ex'] = tableknee[spike_cols].sum(axis=1)
#tableknee.to_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\exercise_proposition\Knee_exos_proposition_"+str(date.today())+".csv")


'''This function is aimed to predict for each patient each day which exercise will or will not be proposed,
it also contain columns with differences between the proposition of the model and the protocol
Input : 
        prot_pt_tbl: input for machine learning with hip or knee patients containing the result of the protocol
        matchings: exercise list for knee or hip patients
        predtbl: input for machine learning with hip or knee patients
Output : Returntbl: the table of prediciton for each days containing the difference between protocol and models
'''

def compare_protocol_model(prot_pt_tbl, matchings,predtbl):
    Returntbl = predtbl[['patient_id', 'patientnumber']].copy()
    Returntbl['day'] = predtbl['day']

    workingtbl = predtbl.drop(['patientnumber', 'date', 'surgery_date', 'patient_id'], axis=1).copy()
    for ex_number in matchings:
        clf2 = pickle.load(open(Location_of_the_models+"\modelfor_" + str(ex_number) + ".sav",'rb'))
        Returntbl[str(ex_number).replace('_frequency','')+'_model'] = list(clf2.predict(workingtbl))

        #Returntbl[str(ex_number).replace('_frequency','')] =prot_pt_tbl[str(ex_number).replace('_frequency','')]
        Returntbl['Protocol_model_difference_' + ex_number] = (prot_pt_tbl[str(ex_number).replace('_frequency','')]-
                                                           Returntbl[str(ex_number).replace('_frequency','')+'_model']).abs()
    return Returntbl


Hiptblwithflags = compare_protocol_model(prot_pt_tbl_hip, matchi_hip,hip_worktbl)
Kneetblwithflags = compare_protocol_model(prot_pt_tbl_knee, matchi_knee,knee_worktbl)


'''This part of the code is aimed at selecting the exercises when there is more than a given number of exercises'''
import pandas as pd
import pickle
from datetime import datetime, timedelta,date


from predict_engine.Create_imput_function import getimput,getexercise_Of_date

daybefore_predicted = datetime.strftime(pd.to_datetime(startdate) - timedelta(1), '%Y-%m-%d')
number_max_exercises = 6



#Check for the safety of the exercises

femketbl = pd.read_csv("C:\\Users\cocol\Desktop\memoire\\femke\Exercise_rules_FEMKE.csv")
t1 = tableknee.copy()

for column in matchi_knee:
    Safety_day = femketbl[femketbl['exercise_number'] == column.replace('_frequency','')]['start_day']
    t1[column.replace('frequency','model')].loc[t1['day']<int(Safety_day)] = 0
#1. check that the treatment are useful (not proposing the circulation exercise at day 30 post op for example)

# 2. check that the treatment are not redundant

daybefore_day = datetime.strftime(pd.to_datetime(startdate) - timedelta(1), '%Y-%m-%d')
Exercises_done_yesterday = getexercise_Of_date(matchi_knee, tbl, daybefore_day)
Exercises_done_yesterday_filtered = Exercises_done_yesterday[Exercises_done_yesterday['patient_id'].isin(list(tableknee['patient_id'].unique()))].reset_index().drop('index', axis=1)

model_exo = [e for e in list(t1.columns)if e not in ('patient_id', 'patientnumber', 'date', 'day','number_of_proposed_ex')]
for line in list(range(t1.shape[0])):
    '''This part of the code is aimed  to select exercises that were not made yesterday if the number of proposed exercises is higher than 6'''
    a = t1[model_exo]
    b = ~Exercises_done_yesterday_filtered[matchi_knee] + 2
    c = pd.DataFrame(a.values * b.values, columns=a.columns, index=a.index)
    t1.loc[c.sum(axis=1)>= number_max_exercises,model_exo] = c.loc[c.sum(axis=1)>= number_max_exercises]
    t1['number_of_proposed_ex'] = t1.loc[:,model_exo].sum(axis=1)
# 3. Make a recommendation service based on a criteria (for example after doing an exercise, the app could ask: "Was that exercise pleasant/useful?" and from that we could create the recommendation: patient who think that single leg stance is useful also think that hometrainer is useful)
#To DO
# 4. Propose exercises which have a better BCR

'''Are we in developpement mode?'''
if localdb:
    FINALTBL = pd.read_csv(Working_Directory + "metaparam\FINALTBL" + '2019-12-30'+ ".csv")
else:
    FINALTBL = pd.read_csv(Working_Directory + "metaparam\FINALTBL" + str(date.today()) + ".csv")



# 5. Propose exercises which are also proposed by the protocol

# 6. Check for similarities between exos
