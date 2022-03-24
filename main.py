from functools import reduce
from tqdm import tqdm

import pandas as pd
import numpy as np
from statistics import NormalDist



def confidence_interval(data: list, confidence: float=0.95) -> list:
    """Compute the confidence interval for a given dataset.
    
    Parameters
    -----
        data (list) -- dataset
        confidence (float) -- confidence value, should be in [0.90, 0.95, 0.99] (default: 0.95)
    
    Raises
    -----
        ValueError -- the given confidence value is not among [0.90, 0.95, 0.99]
    
    Returns
    -----
        list -- resulting confidence interval

    """
    acceptedConfidences = [0.90, 0.95, 0.99]
    if confidence not in acceptedConfidences:
        raise ValueError(f'given confidence value {confidence} not in {acceptedConfidences}')

    dist = NormalDist.from_samples(data)
    z    = NormalDist().inv_cdf((1 + confidence) / 2.0)
    h    = dist.stdev * z / ((len(data) - 1) ** 0.5)

    return [dist.mean - h, dist.mean + h]



#==============================================================================
#       MAIN
#==============================================================================


if __name__ == "__main__":

    from sklearn.datasets import load_wine
    from validations import train_test_validation, cross_validation

#======================================================================

    # load the wine dataset
    wines = load_wine()
    data  = pd.DataFrame(data=wines.data, columns=wines.feature_names)
    data['target'] = wines.target

    # separating data and target
    X = data.drop('target', axis=1)
    Y = data.target

    # explore our dataset
    print(X.describe(), '\n')
    print(data.groupby('target').size())   # showing classes distribution

    # correlation matrix
    from knn_plots import corrMatrix
    corrMatrix(X)

#======================================================================

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

#======================================================================

    # search the optimal value of n for cross validation
    from knn_plots import cross_varrying_effect

    cross_accuracies = []
    nMax = 12
    k = 5
    p = 2

    for ni in tqdm(range(2, nMax)):
        acc = cross_validation(data, ni, k, p)
        cross_accuracies.append(acc)

    cross_accuracies = list(map(lambda x: x*100, cross_accuracies))
    cross_varrying_effect(cross_accuracies, nMax)
    best_n = np.argmax(cross_accuracies) + 2
    print(f'Best cross_validation n value : {best_n}')


    # Cross-validation confidence interval
    confidence = 0.95
    conf_interval = confidence_interval(cross_accuracies, confidence)
    print(f'{confidence*100}% confidence interval for cross-validation : \n {conf_interval}')

#======================================================================

    # observe the effects of varying k
    from knn_plots import k_varying_effect

    p = 2
    kmax = 101
    test_size = 0.25
    k_accuracies = []

    for ki in tqdm(range(1, kmax)):
        acc = cross_validation(data, best_n, ki, p)
        k_accuracies.append(acc)

    k_accuracies = list(map(lambda x: x*100, k_accuracies))
    k_varying_effect(k_accuracies, kmax)
    best_k = np.argmax(k_accuracies) + 1
    print(f'Best k value : {best_k}')
