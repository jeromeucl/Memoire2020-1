__author__      = "Jérôme Dewandre"
'''This part of the code is aimed at selecting the exercises when there is more than a given number of exercises'''
import pandas as pd
import pickle
from datetime import datetime, timedelta,date


from predict_engine.Create_imput_function import getimput,getexercise_Of_date
from Machine_learning import tbl,worktbl,matching, Working_Directory, localdb
from predict_engine.protocol_propositon import Hiptblwithflags,Kneetblwithflags,startdate,matchi_knee,matchi_hip

daybefore_predicted = datetime.strftime(pd.to_datetime(startdate) - timedelta(1), '%Y-%m-%d')
number_max_exercises = 6



#1. check that the treatment are useful (not proposing the circulation exercise at day 30 post op for example)
#To DO

# 2. check that the treatment are not redundant

daybefore_day = datetime.strftime(pd.to_datetime(startdate) - timedelta(1), '%Y-%m-%d')
Exercises_done_yesterday = getexercise_Of_date(matchi_knee, tbl, daybefore_day)

# 3. Make a recommendation service based on a criteria (for example after doing an exercise, the app could ask: "Was that exercise pleasant/useful?" and from that we could create the recommendation: patient who think that single leg stance is useful also think that hometrainer is useful)
#To DO
# 4. Propose exercises which have a better BCR

'''Are we in developpement mode?'''
if localdb:
    FINALTBL = pd.read_csv(Working_Directory + "metaparam\FINALTBL" + '2019-12-30'+ ".csv")
else:
    FINALTBL = pd.read_csv(Working_Directory + "metaparam\FINALTBL" + str(date.today()) + ".csv")



# 5. Propose exercises which are also proposed by the protocol

# 6. Check for similarities between exos