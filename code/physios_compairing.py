'''This file create a table with the predictoin for each exercise and the exercise proposed by the physios'''
from Machine_learning import *
import pickle

# load protocol and merge it with exercise shceme
protocoltbl_hip = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\comparativetbl\protocol\protocol_hip.csv")
protocoltbl_hip.rename(columns={"Days": "day"}, inplace=True)
protocoltbl_knee = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\comparativetbl\protocol\protocol_knee.csv")
protocoltbl_knee.rename(columns={"Days": "day"}, inplace=True)
prot_pt_tbl_hip = pd.merge(exercise_scheme, protocoltbl_hip, on=['day'], how='left')
prot_pt_tbl_knee = pd.merge(exercise_scheme, protocoltbl_knee, on=['day'], how='left')

knee_patient = tbl[tbl['limb'].str.contains("Hip")][['patient_id']]
hip_patient = tbl[tbl['limb'].str.contains("Knee")][['patient_id']]

knee_worktbl = worktbl[worktbl['patient_id'].isin(list(knee_patient['patient_id'].unique()))]
hip_worktbl = worktbl[worktbl['patient_id'].isin(list(hip_patient['patient_id'].unique()))]

final_knee_worktbl = pd.merge(knee_worktbl, prot_pt_tbl_knee, on=['patient_id', 'day'], how='left')
final_hip_worktbl = pd.merge(hip_worktbl, prot_pt_tbl_hip, on=['patient_id', 'day'], how='left')

matching.remove('4011_frequency')
matching = [x for x in matching if not x.startswith('3')]


def compare_protocol_PT(prot_pt_tbl, ex_number):
    Returntbl = prot_pt_tbl[['patient_id', 'patientnumber']].copy()
    Returntbl['day'] = prot_pt_tbl['day']
    Returntbl['PT_decision'] = prot_pt_tbl[ex_number + "_frequency"].notnull().astype(int).to_frame()[
        ex_number + "_frequency"]
    Returntbl['Protocol_advice'] = prot_pt_tbl[ex_number]
    clf2 = pickle.load(
        open("C:\\Users\cocol\PycharmProjects\Memoire2020\code\modeltoexport\modelfor_" + str(ex_number) + "_frequency.sav",
             'rb'))
    workingtbl = prot_pt_tbl.drop(['patientnumber', 'date', 'surgery_date', 'patient_id'], axis=1)
    Returntbl['model_prediction'] = list(clf2.predict(workingtbl))

    Returntbl['PT_Protocol_difference_' + ex_number] = (
                prot_pt_tbl[ex_number + "_frequency"].notnull().astype(int).to_frame()[ex_number + "_frequency"] -
                prot_pt_tbl[ex_number]).abs()
    Returntbl['PT_model_difference_' + ex_number] = (
            prot_pt_tbl[ex_number + "_frequency"].notnull().astype(int).to_frame()[ex_number + "_frequency"] -
            Returntbl['model_prediction']).abs()
    Returntbl['Protocol_model_difference_' + ex_number] = (Returntbl['Protocol_advice'] -
                                                           Returntbl['model_prediction']).abs()

    return Returntbl


table = compare_protocol_PT(final_knee_worktbl, '1001')
