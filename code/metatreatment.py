from Machine_learning import matching
import pandas as pd
from datetime import date
'''This file take the results of the metaparm selector and chech witch is the best combination of metaparameters'''


matching.remove('4011_frequency')
matching = [x for x in matching if not x.startswith('3')]

dfObj = pd.DataFrame(
    columns=['exercise_number', 'criterion', 'max_depth', 'min_samples_split', 'min_impurity_decrease', 'Bcr_test'])
for exo in matching:
    Df1 = pd.read_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\metaparam\met" + str(exo) + ".csv")
    Df = Df1.groupby(['criterion', 'max_depth', 'min_samples_split', 'min_impurity_decrease']).mean()
    ind = Df['Bcr_test'].idxmax(axis=0)
    dfObj = dfObj.append({'exercise_number': exo, 'criterion': ind[0], 'max_depth': ind[1], 'min_samples_split': ind[2],
                          'min_impurity_decrease': ind[3],'Bcr_test':Df.loc[ind]['Bcr_test']}, ignore_index=True)
dfObj.to_csv("C:\\Users\cocol\Desktop\memoire\Jéjé_work\metaparam\FINALTBL"+str(date.today())+".csv")