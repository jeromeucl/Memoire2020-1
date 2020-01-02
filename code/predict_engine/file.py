'''This file create a table with the predictoin for each exercise and the exercise proposed by the physios'''
import pandas as pd
import pickle

from sklearn import tree
from predict_engine.Create_imput import getimput
from Machine_learning import tbl,worktbl,matching

#Select the patient for with you want to predict data with a startdate and set the location of the models previously built
startdate = '2018-11-3'
Location_of_the_models = 'C:\\Users\cocol\PycharmProjects\Memoire2020\code\modeltoexport'


# load imput data
imput_worktbl,imput_tbl = getimput(worktbl,tbl,startdate)


# load protocol and merge it with exercise shceme
protocoltbl_hip = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\comparativetbl\protocol\protocol_hip.csv")
protocoltbl_hip.rename(columns={"Days": "day"}, inplace=True)
protocoltbl_knee = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\comparativetbl\protocol\protocol_knee.csv")
protocoltbl_knee.rename(columns={"Days": "day"}, inplace=True)


hip_patient = imput_tbl.loc[imput_tbl['limb'].str.contains("Hip")][['patient_id']]
knee_patient = imput_tbl.loc[imput_tbl['limb'].str.contains("Knee")][['patient_id']]

knee_worktbl = imput_worktbl[imput_worktbl['patient_id'].isin(list(knee_patient['patient_id'].unique()))]
hip_worktbl = imput_worktbl[imput_worktbl['patient_id'].isin(list(hip_patient['patient_id'].unique()))]

prot_pt_tbl_hip = pd.merge(hip_worktbl, protocoltbl_hip, on=['day'], how='left')
prot_pt_tbl_knee = pd.merge(knee_worktbl, protocoltbl_knee, on=['day'], how='left')




#With flags
def compare_protocol_model(prot_pt_tbl, matchings,predtbl):
    Returntbl = predtbl[['patient_id', 'patientnumber']].copy()
    Returntbl['day'] = predtbl['day']

    workingtbl = predtbl.drop(['patientnumber', 'date', 'surgery_date', 'patient_id'], axis=1).copy()
    for ex_number in matchings:
        clf2 = pickle.load(open(Location_of_the_models+"\modelfor_" + str(ex_number) + ".sav",'rb'))
        Returntbl[str(ex_number).replace('_frequency','')+'_model'] = list(clf2.predict(workingtbl))
        Returntbl['Protocol_model_difference_' + ex_number] = (prot_pt_tbl[str(ex_number).replace('_frequency','')] -
                                                           Returntbl[str(ex_number).replace('_frequency','')+'_model']).abs()
    return Returntbl


#Split the exercise of the hip and the exercises of the hip
matchi_knee = [x for x in matching if x.startswith('1')]
matchi_hip = [x for x in matching if x.startswith('2')]

#Without flags
def proposition_tbl(matchings,input):
    Returntbl = input[['patient_id', 'patientnumber']].copy()
    Returntbl['day'] = input['day']
    for ex_number in matchings:
        clf2 = pickle.load(open(Location_of_the_models+"\modelfor_" + str(ex_number) + ".sav",'rb'))
        workingtbl = input.drop(['patientnumber', 'date', 'surgery_date', 'patient_id'], axis=1).copy()
        Returntbl[str(ex_number).replace('_frequency','')+'_model'] = list(clf2.predict(workingtbl))
    return Returntbl

from datetime import date

tablehip = proposition_tbl(matchi_hip,hip_worktbl)
spike_cols = [col for col in tablehip.columns if 'model' in col]
tablehip['number_of_proposed_ex'] = tablehip[spike_cols].sum(axis=1)
#tablehip.to_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\exercise_proposition\Hip_exos_proposition_"+str(date.today())+".csv")

tableknee = proposition_tbl(matchi_knee,knee_worktbl)
spike_cols = [col for col in tableknee.columns if 'model' in col]
tableknee['number_of_proposed_ex'] = tableknee[spike_cols].sum(axis=1)
#tableknee.to_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\exercise_proposition\Knee_exos_proposition_"+str(date.today())+".csv")


Hiptblwithflags = compare_protocol_model(prot_pt_tbl_hip, matchi_hip,hip_worktbl)
Kneetblwithflags = compare_protocol_model(prot_pt_tbl_knee, matchi_knee,knee_worktbl)