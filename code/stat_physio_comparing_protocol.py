from Machine_learning import matching
import pandas as pd
from datetime import date
import numpy as np
from sklearn.metrics import balanced_accuracy_score

matchi_knee = [x for x in matching if x.startswith('1')]
matchi_hip = [x for x in matching if x.startswith('2')]

dfObj = pd.DataFrame(
    columns=['exercise', 'PT-sum', 'Prot-sum', 'Model-sum', 'PT-Model-sumdiff', 'PT-Prot-sumdiff', 'Model-prot-sumdiff',
             'BCR_prot_PT', 'BCR_model_PT'])
matchi = matchi_knee + matchi_hip
for i in matchi:
    table = pd.read_csv(
        "C:\\Users\cocol\Desktop\memoire\Jéjé_work\comparativetbl\comparetbl_" + i + '2019-11-15' + ".csv")
    dfObj = dfObj.append(
        {'exercise': i, 'PT-sum': table['PT_decision'].sum(), 'Prot-sum': table['Protocol_advice'].sum(),
         'Model-sum': table['model_prediction'].sum(),
         'PT-Prot-sumdiff': table['PT_Protocol_difference_' + i.replace('_frequency', '')].sum(),
         'PT-Model-sumdiff': table['PT_model_difference_' + i.replace('_frequency', '')].sum(),
         'Model-prot-sumdiff': table['Protocol_model_difference_' + i.replace('_frequency', '')].sum(),
         'BCR_prot_PT': balanced_accuracy_score(list(table['PT_decision'].values),
                                                list(table['Protocol_advice'].values)),
         'BCR_model_PT': balanced_accuracy_score(list(table['PT_decision'].values),
                                                 list(table['model_prediction'].values))}, ignore_index=True)




dfObj['Is_model_uselful?'] = dfObj['PT-Model-sumdiff'] < dfObj['PT-Prot-sumdiff']
dfObj['Is_model_uselful_BCR?'] = dfObj['BCR_model_PT'] > dfObj['BCR_prot_PT']
dfObj.to_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\second pres\ismodeluseful" + str(date.today()) + ".csv")


