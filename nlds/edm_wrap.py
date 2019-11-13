import numpy as np
from scipy import io, signal, stats
import pyEDM as edm
import pandas as pd


def iterPredDim(dataframe,lib_inds,pred_inds, Tps, maxdim, tau=1):
    # iterate through all configurations of prediction horizon and embedding dimensions
    dfs = []
    for Tp in Tps:
        dfs.append(edm.EmbedDimension(dataFrame=dataframe, lib=lib_inds, pred=pred_inds, maxE=maxdim, tau=tau, Tp=Tp, columns="X", target="X", showPlot=False))

    df_res = pd.DataFrame(np.array([df['rho'].values for df in dfs]), index=Tps, columns=np.arange(1,maxdim+1))
    return df_res
