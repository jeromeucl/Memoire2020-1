'''This file aime to treat the result of Tree_metaparam and choose for each exercise the best tree'''
__author__      = "Jérôme Dewandre"
import pandas as pd
from datetime import date
'''This file take the results of the metaparm selector and chech witch is the best combination of metaparameters'''

def metatrt(workdir,matching):

    dfObj = pd.DataFrame(
        columns=['exercise_number', 'criterion', 'max_depth', 'min_samples_split', 'min_impurity_decrease', 'Bcr_test','parameters'])
    for exo in matching:
        Df = pd.read_csv(workdir+"metaparam\met" + str(exo) + ".csv")

        ind = Df['Bcr_test'].idxmax(axis=0)
        dfObj = dfObj.append({'exercise_number': exo, 'criterion': Df.loc[ind]['criterion'], 'max_depth': Df.loc[ind]['max_depth'], 'min_samples_split': Df.loc[ind]['min_samples_split'],
                              'min_impurity_decrease': Df.loc[ind]['min_impurity_decrease'],'Bcr_test':Df.loc[ind]['Bcr_test'],'parameters':Df.loc[ind]['parameters']}, ignore_index=True)
    dfObj.to_csv(workdir+"metaparam\FINALTBL"+str(date.today())+".csv")
