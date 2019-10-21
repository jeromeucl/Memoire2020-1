# 'PaIn2' pain during the day
# 'PaIn1 pain during the night
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append('C:\\Users\cocol\Desktop\memoire\Jéjé_work\code')
from Machine_learning import patient_daily_data
from Machine_learning import exercise_scheme
from Machine_learning import Working_Directory
from datetime import date

cols = exercise_scheme.columns.drop(
    list(exercise_scheme.filter(regex='intensity')) + list(
        exercise_scheme.filter(regex='actual')))
exercise_scheme = exercise_scheme[cols]
exercise_scheme = exercise_scheme.apply(
    lambda x: x.notnull().astype(int) if "frequency" in x.name else x)

Variabletochec = 'PaIn2'
Pain_threshold = 20
pdt0 = patient_daily_data[['patient_id', 'day', Variabletochec]]
pdt0 = pdt0.fillna(method='bfill')
pdt0 = pdt0.fillna(method='ffill')


def two_var_cor(pd_dy_var_tbl, pain_threshold, tbl_exsch, variabletochec, number_of_different_day):
    cp = pd_dy_var_tbl.copy()
    cp['day'] -= number_of_different_day
    cp.rename(columns={variabletochec: str(number_of_different_day)}, inplace=True)
    table = pd.merge(pd_dy_var_tbl, cp, on=['patient_id', 'day'], how='inner')
    table = pd.merge(tbl_exsch, table, on=['patient_id', 'day'], how='inner')
    table[str('diff' + str(number_of_different_day))] = np.where(
        table[variabletochec] - table[str(number_of_different_day)] < pain_threshold, 0, 1)
    return table


worktl = two_var_cor(pdt0, Pain_threshold, exercise_scheme, Variabletochec, 1)
corrtbl = worktl.corr()
res = corrtbl.iloc[:, -1]
res.drop(res.tail(3).index, inplace=True)
for i in range(2, 30):
    worktl = two_var_cor(pdt0, Pain_threshold, exercise_scheme, Variabletochec, i)
    corrtbl = worktl.corr()
    TR = corrtbl.iloc[:, -1]
    TR.drop(TR.tail(3).index, inplace=True)  # drop last n rows
    res = pd.concat([res, TR], axis=1)

res.max().to_frame()
res.to_csv(Working_Directory + "\Results_pearson_exo_increasepain" + Variabletochec + str(date.today()) + ".csv")
