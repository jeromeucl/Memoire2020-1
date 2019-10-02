#'PaIn2' pain during the day
#'PaIn1 pain during the night
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append('C:\\Users\cocol\Desktop\memoire\Jéjé_work\code')
from Machine_learning import patient_daily_data
from Machine_learning import exercise_scheme

cols = exercise_scheme.columns.drop(
    list(exercise_scheme.filter(regex='intensity')) + list(
        exercise_scheme.filter(regex='actual')))
exercise_scheme = exercise_scheme[cols]
exercise_scheme = exercise_scheme.apply(
    lambda x: x.notnull().astype(int) if "frequency" in x.name else x)

Variabletochec = 'PaIn2'
Pain_threshold = 20
pdt0 = patient_daily_data[['patient_id','day',Variabletochec]]
pdt0 = pdt0.fillna(method='bfill')
pdt0 = pdt0.fillna(method='ffill')




pdt1 = pdt0.copy()
pdt1['day'] -=1
pdt1.rename(columns={Variabletochec: Variabletochec + "+1"}, inplace=True)

pdt2 = pdt0.copy()
pdt2['day'] -=2
pdt2.rename(columns={Variabletochec:str(Variabletochec + "+2")},inplace=True)
pdt3 = pdt0.copy()
pdt3['day'] -=3
pdt3.rename(columns={Variabletochec: str(Variabletochec + "+3")}, inplace=True)
pdt4 = pdt0.copy()
pdt4['day'] -=4
pdt4.rename(columns={Variabletochec: str(Variabletochec + "+4")}, inplace=True)
effectbl = pd.merge(exercise_scheme, pdt0, on=['patient_id', 'day'], how='inner')
effectbl = pd.merge(effectbl , pdt1, on=['patient_id', 'day'], how='inner')
effectbl = pd.merge(effectbl , pdt2, on=['patient_id', 'day'], how='inner')
effectbl = pd.merge(effectbl , pdt3, on=['patient_id', 'day'], how='inner')
effectbl = pd.merge(effectbl , pdt4, on=['patient_id', 'day'], how='inner')
effectbl['diff1'] = np.where(effectbl[Variabletochec] - effectbl[Variabletochec + "+1"] < Pain_threshold, 0, 1)
effectbl['diff2'] = np.where(effectbl[Variabletochec] - effectbl[Variabletochec + "+2"] < Pain_threshold, 0, 1)
effectbl['diff3'] = np.where(effectbl[Variabletochec] - effectbl[Variabletochec + "+3"] < Pain_threshold, 0, 1)
effectbl['diff4'] = np.where(effectbl[Variabletochec] - effectbl[Variabletochec + "+4"] < Pain_threshold, 0, 1)

corrtbl = effectbl.corr()