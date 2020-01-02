from Machine_learning import worktbl

def getimput(Worktbl,Tbl,date):

    indexes = Worktbl['date'] >= date
    imput_data_work = Worktbl.loc[indexes]
    imput_data_tbl = Tbl.loc[indexes]
    return imput_data_work,imput_data_tbl

#getimput(worktbl,'2018-11-3')


