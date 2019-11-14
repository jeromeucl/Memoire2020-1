#https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html

from sklearn.svm import SVC
from sklearn import tree
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
import numpy
from Machine_learning import *

from sklearn.metrics import balanced_accuracy_score, make_scorer
BCR = make_scorer(balanced_accuracy_score)
# Number of random trials
NUM_TRIALS = 1

# Load the dataset
worktbl = worktbl.drop(['patientnumber', 'date', 'surgery_date', 'patient_id'], axis=1)
X_iris = worktbl
y_iris = tbl['1001_frequency'].notnull().astype(int).to_frame()

# Set up possible values of parameters to optimize over
#p_grid = {"C": [1, 10, 100],"gamma": [.01, .1]}
p_grid = {"max_depth": [2,3,6,10,15,20],"min_samples_split" : [2,4,5,8,10,15,20],"min_impurity_decrease" : [0,0.01,0.03,0.05,0.07,0.1,0.15,0.2,0.3],'criterion':['entropy','gini']}
# We will use a Support Vector Classifier with "rbf" kernel
#svm = SVC(kernel="rbf")

# Arrays to store scores
non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)

# Loop for each trial
for i in range(NUM_TRIALS):
    print('Number of trials = '+str(i))
    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
    # Same as K-Fold but preserves the class distribution within each fold
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=tree.DecisionTreeClassifier(), param_grid=p_grid, cv=inner_cv,
                       iid=False,scoring=BCR)
    clf.fit(X_iris, y_iris)
    non_nested_scores[i] = clf.best_score_
    clf.best_params_
    # Nested CV with parameter optimization
    nested_score = cross_val_score(clf, X=X_iris, y=y_iris, cv=outer_cv,scoring=BCR)
    nested_scores[i] = nested_score.mean()

