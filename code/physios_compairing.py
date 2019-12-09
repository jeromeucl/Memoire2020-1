'''This file create a table with the predictoin for each exercise and the exercise proposed by the physios'''
import pandas as pd
import pickle
tbl = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\local_tbl.csv")
worktbl = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\localworktbl.csv")
from sklearn import tree
# load protocol and merge it with exercise shceme
protocoltbl_hip = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\comparativetbl\protocol\protocol_hip.csv")
protocoltbl_hip.rename(columns={"Days": "day"}, inplace=True)
protocoltbl_knee = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\comparativetbl\protocol\protocol_knee.csv")
protocoltbl_knee.rename(columns={"Days": "day"}, inplace=True)


hip_patient = tbl[tbl['limb'].str.contains("Hip")][['patient_id']]
knee_patient = tbl[tbl['limb'].str.contains("Knee")][['patient_id']]

knee_worktbl = worktbl[worktbl['patient_id'].isin(list(knee_patient['patient_id'].unique()))]
hip_worktbl = worktbl[worktbl['patient_id'].isin(list(hip_patient['patient_id'].unique()))]

prot_pt_tbl_hip = pd.merge(exercise_scheme, protocoltbl_hip, on=['day'], how='left')
prot_pt_tbl_knee = pd.merge(exercise_scheme, protocoltbl_knee, on=['day'], how='left')
final_knee_worktbl = pd.merge(knee_worktbl, prot_pt_tbl_knee, on=['patient_id', 'day'], how='left')
final_hip_worktbl = pd.merge(hip_worktbl, prot_pt_tbl_hip, on=['patient_id', 'day'], how='left')

matching.remove('4011_frequency')
matching = [x for x in matching if not x.startswith('3')]


def compare_protocol_PT(prot_pt_tbl, ex_number,predtbl):
    Returntbl = prot_pt_tbl[['patient_id', 'patientnumber']].copy()
    Returntbl['day'] = prot_pt_tbl['day']
    Returntbl['PT_decision'] = prot_pt_tbl[ex_number + "_frequency"].notnull().astype(int).to_frame()
    Returntbl['Protocol_advice'] = prot_pt_tbl[ex_number]
    clf2 = pickle.load(open("C:\\Users\cocol\PycharmProjects\Memoire2020\code\modeltoexport\modelfor_" + str(ex_number) + "_frequency.sav",'rb'))
    workingtbl = predtbl.drop(['patientnumber', 'date', 'surgery_date', 'patient_id'], axis=1).copy()

    Returntbl['model_prediction'] = list(clf2.predict(workingtbl))

    Returntbl['PT_Protocol_difference_' + ex_number] = (Returntbl['PT_decision'] -Returntbl['Protocol_advice'] ).abs()
    Returntbl['PT_model_difference_' + ex_number] = (Returntbl['PT_decision'] -Returntbl['model_prediction']).abs()
    Returntbl['Protocol_model_difference_' + ex_number] = (Returntbl['Protocol_advice'] -
                                                           Returntbl['model_prediction']).abs()

    return Returntbl
#knee
matchi_knee = [x for x in matching if x.startswith('1')]
matchi_hip = [x for x in matching if x.startswith('2')]
matchis = matchi_hip + matchi_knee
#for i in matchis:
#
#    table = compare_protocol_PT(final_knee_worktbl, i.replace('_frequency',''),knee_worktbl)
#    table.to_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\comparativetbl\comparetbl_"+i+str(date.today())+".csv")

def proposition_tbl(prot_pt_tbl, matchings,predtbl):
    Returntbl = prot_pt_tbl[['patient_id', 'patientnumber']].copy()
    Returntbl['day'] = prot_pt_tbl['day']
    for ex_number in matchings:
        clf2 = pickle.load(open("C:\\Users\cocol\PycharmProjects\Memoire2020\code\modeltoexport\modelfor_" + str(ex_number) + ".sav",'rb'))
        workingtbl = predtbl.drop(['patientnumber', 'date', 'surgery_date', 'patient_id'], axis=1).copy()

        Returntbl[str(ex_number).replace('_frequency','')+'_model'] = list(clf2.predict(workingtbl))



    return Returntbl

tablehip = proposition_tbl(final_hip_worktbl, matchi_hip,hip_worktbl)
spike_cols = [col for col in tablehip.columns if 'model' in col]
tablehip['number_of_proposed_ex'] = tablehip[spike_cols].sum(axis=1)
tablehip.to_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\exercise_proposition\Hip_exos_proposition_"+str(date.today())+".csv")
tableknee = proposition_tbl(final_knee_worktbl, matchi_knee,knee_worktbl)
spike_cols = [col for col in tableknee.columns if 'model' in col]
tableknee['number_of_proposed_ex'] = tableknee[spike_cols].sum(axis=1)
tableknee.to_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\exercise_proposition\Knee_exos_proposition_"+str(date.today())+".csv")