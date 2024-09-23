import pandas as pd
import numpy

import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve

import dask
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
import dask_ml.model_selection as dcv
import joblib

def tune_random_forest(x_train, y_train) -> None:
    """
    Parameter tuning for random forest.
    """
    param_grid = [{"max_depth": [70, 100, None]}, {"max_features": ["auto", "sqrt"]}, {"n_estimators": [20, 100, 500, 1000, 2000]}]
    random_forest = RandomForestClassifier(random_state=42)

    grid_search = dcv.GridSearchCV(random_forest, param_grid, cv = 5, scoring = "neg_mean_squared_error")
    grid_search.fit(x_train, y_train)
    grid_search.best_params_

def test_random_forest(x_train, y_train, x_test, y_test) -> None:
    """
    Testing the tuned model on the test data
    """
    random_forest = RandomForestClassifier(random_state = 42, max_depth = 70, max_features = "auto", n_estimators = 500)
    random_forest.fit(x_train, y_train)
    
    score = random_forest.score(x_test, y_test)
    accuracy = numpy.mean(score)

    print(accuracy) 

def tune_decision_tree(x_train, y_train) -> None:
    """
    Parameter tuning for decision tree.
    """
    param_grid = [{"max_depth": [70, 100, None]}, {"max_features": ["auto", "sqrt"]}, {"min_samples_leaf": [1, 2, 5, 10]}]
    decision_tree = DecisionTreeClassifier(random_state=42)

    grid_search = dcv.GridSearchCV(decision_tree, param_grid, cv = 5, scoring = "neg_mean_squared_error")
    grid_search.fit(x_train, y_train)
    grid_search.best_params_

def test_decision_tree(x_train, y_train, x_test, y_test) -> None:
    """
    Testing the tuned model on the test data
    """
    decision_tree = DecisionTreeClassifier(random_state = 42, max_depth = 70, max_features = "auto", min_samples_leaf=5)
    decision_tree.fit(x_train, y_train)
    
    score = decision_tree.score(x_test, y_test)
    accuracy = numpy.mean(score)

    print(accuracy)

def tune_k_nearest_neighbours(x_train, y_train) -> None:
    """
    Parameter tuning for k nearest neighbours
    """
    param_grid = [{"n_neighbours": range(1,21)}]
    k_neighbours = KNeighborsClassifier(random_state=42)

    grid_search = dcv.GridSearchCV(k_neighbours, param_grid, cv = 5, scoring = "neg_mean_squared_error")
    grid_search.fit(x_train, y_train)
    grid_search.best_params_

def test_k_nearest_neighbours(x_train, y_train, x_test, y_test) -> None:
    """
    Testing the tuned model on the test data
    """
    k_neighbours = KNeighborsClassifier(random_state = 42, n_neighbors=3)
    k_neighbours.fit(x_train, y_train)
    
    score = k_neighbours.score(x_test, y_test)
    accuracy = numpy.mean(score)

    print(accuracy)

def main():
    dataset = dd.read_csv("star_classification_processed.csv")
    dataset.describe()

    cluster = LocalCluster()
    Client = dask.distributed.Client()

    x = dataset.drop(["class"], axis = 1)
    y = dataset["class"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, shuffle = False)

   # tune_random_forest(x_train, y_train)
    test_random_forest(x_train, y_train, x_test, y_test) #scored 0.978341510111284

   # tune_decision_tree(x_train, y_train)
    test_decision_tree(x_train, y_train, x_test, y_test) #scored 0.9681703960751465

   # tune_k_nearest_neighbours(x_train, y_train)
    test_k_nearest_neighbours(x_train, y_train, x_test, y_test) #scored 0.9296996529855212


if __name__ == "__main__":
    main()