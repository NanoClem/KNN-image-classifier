import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd



def corrMatrix(df: pd.DataFrame):
    """[summary]
    
    Parameters
    -----
        df (pd.DataFrame) -- [description]
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
    """[summary]
    
    Parameters
    -----
        accuracies (list) -- [description]
        min (int) -- [description] (default: 1)
        max (int) -- [description] (default: 100)
    """
    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.plot(range(min, max), accuracies)
    ax.set_xlabel('Nb of Nearest Neighbors (k)')
    ax.set_ylabel('Accuracy (%)')

    plt.show()