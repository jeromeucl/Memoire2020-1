'''This file create a table with the predictoin for each exercise and the exercise proposed by the physios'''
from Machine_learning import *
import pickle
#load protocol and merge it with exercise shceme
protocoltbl_hip = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\comparativetbl\protocol\protocol_hip.csv")
protocoltbl_hip.rename(columns={"Days": "day"}, inplace=True)
protocoltbl_knee = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\comparativetbl\protocol\protocol_knee.csv")
protocoltbl_knee.rename(columns={"Days": "day"}, inplace=True)
prot_pt_tbl_hip = pd.merge(exercise_scheme, protocoltbl_hip, on=['day'], how='right')
prot_pt_tbl_knee = pd.merge(exercise_scheme, protocoltbl_knee, on=['day'], how='right')

matching.remove('4011_frequency')
matching = [x for x in matching if not x.startswith('3')]


def compare_protocol_PT(prot_pt_tbl,hip_or_knee):
    Returntbl = prot_pt_tbl[['patient_id','day']].copy()
    Returntbl['day'] = prot_pt_tbl['day']

    exexcise_list = [s for s in prot_pt_tbl.columns if s.isdigit()]
    for ex_number in exexcise_list:
        Returntbl['PT_Protocol_difference_'+hip_or_knee+'_' + ex_number] = (prot_pt_tbl[ex_number + "_frequency"].notnull().astype(int).to_frame()[ex_number + "_frequency"] - prot_pt_tbl[ex_number]).abs()

    return Returntbl
for exo in matching:
    clf2 = pickle.load(open("modeltoexport\\modelfor_"+str(exo)+".sav", 'rb'))
