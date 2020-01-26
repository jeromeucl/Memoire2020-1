
import matplotlib.pyplot as plt
import pandas as pd
import pymysql.cursors

# mysql connection
# connection = pymysql.connect(host='173.31.240.35.bc.googleusercontent.com', user='', password='',
#                             db='moveup_dwh', charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
connection = pymysql.connect(host='127.0.0.1', user='root', password='root',
                             db='moveup_dwh', charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)

# get not used exos
sql = "select * from exercise_log"
notused = pd.read_sql(sql, connection, index_col=None, coerce_float=True, params=None, parse_dates=None, columns=None,
                    chunksize=None)

ntu_exercise = pd.DataFrame(notused['exercise number'].value_counts())
ntu_exercise=ntu_exercise.sort_values('exercise number')
print(ntu_exercise.iloc[:,0].sum())
plt.bar(ntu_exercise.index.values, list(ntu_exercise.iloc[:,0]), color='b' )
#plt.hist(notused['day'], bins=40)

plt.show()


# get used exos
sql = "select * from exercise_automation"
used = pd.read_sql(sql, connection, index_col=None, coerce_float=True, params=None, parse_dates=None, columns=None,
                   chunksize=None)
_exercise = pd.DataFrame(used['exercise_number'].value_counts())
_exercise=_exercise.sort_values('exercise_number')
print(_exercise)
print(list(_exercise.iloc[:,0]))
plt.bar(_exercise.index.values, list(_exercise.iloc[:,0]), color='b' )
#plt.hist(used['day'], bins=40)


plt.show()
from Machine_learning import tbl, matching
collist = ['patientnumber', 'patient_id', 'day'] + matching
imput_data_tbl = tbl.apply(lambda column: column.notnull().astype(int) if ('_frequency' in column.name) else column, axis=0)
imput_data_tbl = imput_data_tbl[collist]


import matplotlib.pyplot as plt

a= imput_data_tbl[matching].sum(axis=1)
_ = plt.hist(a,14)  # arguments are passed to np.histogram
plt.title("Number_of_exercises proposed by PT")
plt.show()

'''Diff between tree len 3 and rest of the trees'''

