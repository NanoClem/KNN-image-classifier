# %%
from functools import reduce
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    """[summary]
    
    Parameters
    -----
        dataset (pd.DataFrame) -- [description]
        test_size (float) -- [description] (default: 0.25)
        k (int) -- [description] (default: 3)
        p (int) -- [description] (default: 2)
    
    Returns
    -----
        float -- [description]
    """
    X = dataset.drop('target', axis=1)
    Y = dataset.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=1, shuffle=True)

    # avoiding data leakage
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


# %%
if __name__ == "__main__":

    from sklearn.datasets import load_wine
    
    # load the wine dataset
    wines = load_wine()
    data  = pd.DataFrame(data=wines.data, columns=wines.feature_names)
    data['target'] = wines.target

# %%
    # separating data and target
    X = data.drop('target', axis=1)
    Y = data.target
# %%
    # explore our dataset
    print(X.describe(), '\n')
    print(data.groupby('target').size())   # showing classes distribution

    # correlation matrix
    from knn_plots import corrMatrix
    corrMatrix(X)
# %%
    # KNN params
    k = 5
    p = 2
    
    # train test validation
    test_size = 0.25
    TT_acc = train_test_validation(data, test_size, k, p)
    print("KNN train_test validation accuracy score : {}".format(TT_acc))

    # cross validation
    n = 5
    cross_acc = cross_validation(data, n, k, p)
    print("KNN cross validation accuracy score : {}".format(cross_acc))
#%%
    # search the optimal value of n for cross validation
    from knn_plots import cross_varrying_effect

    cross_accuracies = []
    nMax = 50
    k = 5
    p = 2

    for ni in tqdm(range(2, nMax)):
        acc = cross_validation(data, ni, k, p)
        cross_accuracies.append(acc)

    cross_varrying_effect(cross_accuracies, nMax)
    best_n = np.argmax(cross_accuracies) + 2
    print(f'Best cross_validation n value : {best_n}')
    
# %%
    # observe the effects of varying k
    from knn_plots import k_varying_effect

    p = 2
    kmax = 100
    test_size = 0.25
    k_accuracies = []

    for ki in tqdm(range(1, kmax)):
        acc = train_test_validation(data, test_size, ki, p)
        k_accuracies.append(acc)

    k_varying_effect(k_accuracies, kmax)
    best_k = np.argmax(k_accuracies) + 1
    print(f'Best k value : {best_k}')
# %%
