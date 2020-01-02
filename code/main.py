'''This file is the main file to build the models'''
__author__ = "Jérôme Dewandre"

from pathlib import Path
from Machine_learning import worktbl, matching, tbl, Working_Directory
import sys
from Tree_Metaparam import *
from metatreatment import *
from exportmodel import *

'''Todo'''


# First go to the Machine_learning.py file and set 'localdb' variable to False
# Second in the Machine_learning.py file and set your working directory


def main():
    workdir = Working_Directory

    # Check if those directorys exist and create them if they don't

    Path(workdir + "\metaparam").mkdir(parents=True, exist_ok=True)

    # The models will be stored in this file
    Path(workdir + "\models").mkdir(parents=True, exist_ok=True)

    #Build a table for each exercise containing the result of each hyperparameters based on a stratified kfold
    tree_metaparam(worktbl, tbl, matching, workdir)

    #Select the best hyper-parameters for each exercise
    metatrt(workdir, matching)

    #Build a model for each exercise based on the best hyper parameters and store it in models
    export_best_models(worktbl, tbl, matching, workdir)


if __name__ == '__main__':
    main()
