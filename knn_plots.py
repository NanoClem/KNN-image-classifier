import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd



def corrMatrix(df: pd.DataFrame):
    """Plot the correlation matrix of the given dataframe.
    
    Parameters
    -----
        df (pd.DataFrame) -- dataframe from which to plot the matrix
    """
    # get correlation matrix
    corr = df.corr()

    # setup plot
    ax = sn.heatmap(corr,  
        square=True,
        vmin=-1,
        vmax=1, center=0,
        cmap=sn.diverging_palette(20, 220, n=200))

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right')

    plt.show()



def k_varying_effect(accuracies: list, min: int=1, max: int=100):
    """Plot the evolution of the k parameter in knn.
    
    Parameters
    -----
        accuracies (list) -- list of computed accuracies for different values of k
        min (int) -- min value of k (default: 1)
        max (int) -- max value of k (default: 100)
    """
    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.plot(range(min, max), accuracies)
    ax.set_xlabel('Nb of Nearest Neighbors (k)')
    ax.set_ylabel('Accuracy (%)')

    plt.show()