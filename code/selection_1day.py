import matplotlib.pyplot as plt
import pandas as pd
import pymysql.cursors

# mysql connection
# connection = pymysql.connect(host='173.31.240.35.bc.googleusercontent.com', user='', password='',
#                             db='moveup_dwh', charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
connection = pymysql.connect(host='127.0.0.1', user='root', password='root',
                             db='moveup_dwh', charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)

# get rules
sql = "select * from exercise_rules"
rules = pd.read_sql(sql, connection, index_col=None, coerce_float=True, params=None, parse_dates=None, columns=None,
                    chunksize=None)
with connection.cursor() as cursor:
    # Daily data of all patients less than 90 days after surgery
    sql = "select * from vw_exercise_building"
    cursor.execute(sql)
connection.commit()

results = []
log = []

frame = pd.DataFrame([i.copy() for i in cursor])

plt.hist(frame['day'], bins=400)
plt.show()
