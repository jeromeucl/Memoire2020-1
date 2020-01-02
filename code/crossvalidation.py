__author__      = "Jérôme Dewandre"
import pandas as pd
from sklearn import tree
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold  # import KFold


def crossval(Matching, Mapping_exercises, Big_tbl, Worktlb):

    # Build a dataframe for the results
    results_cv = pd.DataFrame(columns=["Exercise","number", "Type_of_algorithm", "mean_Bcr_train", "mean_Bcr_test", 'Used_fold'])
    used_fold = 5
    '''Main loop'''
    # Pich one column corresponding to an exercise at a time and make it the label
    for exercise_number in Matching:
        # Extract the number of the exercise (example: 1001)
        name_of_exercise = exercise_number.replace("_frequency", "")
        # Extract the full name of the exercise (example: Exercise 1 (K): Circulation )
        name_of_exercise = Mapping_exercises['name'][
            Mapping_exercises.index[Mapping_exercises['number'] == int(name_of_exercise)].tolist()].values[0]
        # Create the label for the machine learning algorithm
        label = Big_tbl[exercise_number].notnull().astype(int).to_frame()

        mean_Bcr_train = 0
        mean_Bcr_test = 0
        kf = KFold(n_splits=used_fold, shuffle=True, random_state=42)  # Define the split - into "n_splits" folds
        iter = used_fold
        for train_index, test_index in kf.split(Worktlb):

            # Split the data and the label into test and train set
            train, test = Worktlb.iloc[train_index], Worktlb.iloc[test_index]
            label_train, label_test = label.iloc[train_index], label.iloc[test_index]

            # If trouble in train set

            if sum(label_train.values) == 0:
                print('Issue in kflod of ' + str(name_of_exercise))
                # Do nothing
                mean_Bcr_train = mean_Bcr_train
                iter -= 1
            else:
                # Train prediction
                clf = tree.DecisionTreeClassifier(max_depth=5,class_weight ='balanced')
                clf = clf.fit(train, label_train)
                # Get the most important feature

                # Predict the label for train set
                train_pred = clf.predict(train)

                # confusion_matrix(y_true, y_pred)

                bcr_train = balanced_accuracy_score(label_train, train_pred)
                mean_Bcr_train = mean_Bcr_train + bcr_train

                # Test prediction with the model build on the train set
                test_pred = clf.predict(test)
                # confusion_matrix(y_true, y_pred)

                bcr_test = balanced_accuracy_score(label_test, test_pred)
                mean_Bcr_test = mean_Bcr_test + bcr_test
        # Add everinthing to the Result table
        if iter != 0:
            mean_Bcr_train = mean_Bcr_train / used_fold
            mean_Bcr_test = mean_Bcr_test / used_fold
        results_cv = results_cv.append(
            {"Exercise": name_of_exercise,"number":exercise_number.replace("_frequency", ""), "Type_of_algorithm": "Tree", "mean_Bcr_train": mean_Bcr_train,
             "mean_Bcr_test": mean_Bcr_test, "Used_fold": iter}, ignore_index=True)

    return results_cv
