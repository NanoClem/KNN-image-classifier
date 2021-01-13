import pandas as pd
import numpy as np
from functools import reduce

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from knn import knn



def scale_transform(X_train: list, X_test: list) -> list:
    """Applies a scale transformation to data to avoid data leakage.
    
    Parameters
    -----
        X_train (list) -- training data
        X_test (list) -- test data

    Returns
    -----
        list -- both transformed datasets (train / test)
    """
    scaler  = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)



def train_test_validation(dataset: pd.DataFrame, test_size: float=0.25, k: int=3, p: int=2) -> float:
    """Execute a train-test validation on a given dataset with the KNN algorithm.
    
    Parameters
    -----
        dataset (pd.DataFrame) -- dataset as a pandas Dataframe
        test_size (float) -- part of the dataset given for testing (default: 0.25)
        k (int) -- KNN parameter (default: 3)
        p (int) -- Minkowski parameter (default: 2)
    
    Returns
    -----
        float -- global accuracy score.
    """
    X = dataset.drop('target', axis=1)
    Y = dataset.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=1, shuffle=True)

    # scale transformation
    X_train, X_test = scale_transform(X_train, X_test)

    # make preditions
    predictions = knn(X_train, X_test, Y_train, k, p)
    return accuracy_score(Y_test, predictions)



def cross_validation(dataset: pd.DataFrame, n: int=5, k: int=3, p: int=2) -> float:
    """[summary]
    
    Parameters
    -----
        dataset (pd.DataFrame) -- [description]
        n (int) -- [description] (default: 5)
        k (int) -- [description] (default: 3)
        p (int) -- [description] (default: 2)

    Returns
    -----
        float -- [description]
    """
    scores = []
    data   = shuffle(dataset, random_state=5)       # shuffle the dataset
    chunks = np.array_split(data, n)                # split into equals chunks

    for chk in chunks:

        # defining train and test set
        test  = chk
        train = pd.concat([df for df in chunks if not df.equals(chk)])

        # separating data and target
        X_train, Y_train = train.drop('target', axis=1), train.target
        X_test, Y_test   = test.drop('target', axis=1), test.target

        # avoiding data leakage
        X_train, X_test = scale_transform(X_train, X_test)

        # make prediction for the current chunk
        predictions = knn(X_train, X_test, Y_train, k, p)
        accuracy    = accuracy_score(Y_test, predictions)
        scores.append(accuracy)

    sum = reduce(lambda x,y: x+y, scores)
    return sum/len(scores)