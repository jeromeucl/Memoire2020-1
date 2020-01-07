__author__      = "Jérôme Dewandre"
'''This file create a table with the prediction for each exercise and the exercise proposed by the physios
it also check for the number of exercises proposed each days. The last part of the file compare the result of the models 
with the results of the protocol to be able to set flags if there is a difference'''
import pandas as pd
import pickle
from datetime import datetime, timedelta


from predict_engine.Create_imput import getimput
from Machine_learning import tbl,worktbl,matching, Working_Directory, localdb

#Select the patient for with you want to predict data with a startdate and set the location of the models previously built
startdate = '2018-11-3'
Location_of_the_models = 'C:\\Users\cocol\PycharmProjects\Memoire2020\code\modeltoexport'



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
    Returntbl = input[['patient_id', 'patientnumber']].copy()
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



