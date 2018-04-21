import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, firwin
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
import time

def compute_MI(x, y, bins):
    """ Compute mutual information between two vectors given custom bins.

    Parameters
    ----------
    x, y : array, 1D
        Signals to compute mutual information between.
    bins : integer or array, 1D
        Number of bins (if integer) or bin edges (if array) for 2D histogram.

    Returns
    -------
    mi : float
        Mutual information estimate.

    """

    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def delay_MI(data, bins, max_tau, tau_step=1):
    """ Calculates MI at different delays for a time series.

    Parameters
    ----------
    data : array, 1D
        Time series to compute delayed MI over.
    bins : integer or array, 1D
        Number of bins (if integer) or bin edges (if array) for 2D histogram.
    max_tau : int
        Maximum delay to compute MI, in samples of signal.
    tau_step : int (default=1)
        Step size to advance for consecutive MI calculation, in samples of signal.

    Returns
    -------
    MI_timepoints : array, 1D
        Time points (in samples) at which MI was computed.
    dMI : array, 1D
        Time lagged mutual information.

    """
    MI_timepoints = np.arange(0, max_tau, tau_step)
    dMI = np.zeros(len(MI_timepoints))
    dMI[0] = compute_MI(data, data, bins)
    for ind, tau in enumerate(MI_timepoints[1:]):
        dMI[ind + 1] = compute_MI(data[:-tau], data[tau:], bins)

    return MI_timepoints, dMI


def autocorr(data, max_lag=1000, lag_step=1):
    """ Calculate the signal autocorrelation (lagged correlation)

    Parameters
    ----------
    data : array 1D
        Time series to compute autocorrelation over.
    max_lag : int (default=1000)
        Maximum delay to compute AC, in samples of signal.
    lag_step : int (default=1)
        Step size (lag advance) to move by when computing correlation.

    Returns
    -------
    ac_timepoints : array, 1D
        Time points (in samples) at which correlation was computed.
    ac : array, 1D
        Time lagged (auto)correlation.

    """

    ac_timepoints = np.arange(0, max_lag, lag_step)
    ac = np.zeros(len(ac_timepoints))
    ac[0] = np.sum((data - np.mean(data))**2)
    for ind, lag in enumerate(ac_timepoints[1:]):
        ac[ind + 1] = np.sum((data[:-lag] - np.mean(data[:-lag]))
                             * (data[lag:] - np.mean(data[lag:])))

    return ac_timepoints, ac / ac[0]


def find_valley(data):
    """ Finds the first valley of a (relatively) smooth vector; can be applied to
    lagged MI or autocorrelation. If no valley is found, return the last value,
    which would be the minimum.

    Parameters
    ----------
    data : array, 1D
        Vector to find the first valley in.

    Returns
    -------
    valley_ind : int
        Index in the data where valley is found.

    valley_val : float
        Data value at the valley or minimum.

    """
    # find the first point in data in which first derivative is pos
    if np.where(np.diff(data) > 0.)[0].size:
        valley_ind = np.where(np.diff(data) > 0.)[0][0]
    else:
        # if no such point is found, it means data is monotonically
        # decreasing, so return the end index
        valley_ind = len(data) - 1

    # get the minimum MI (at delay)
    valley_val = data[valley_ind]
    return valley_ind, valley_val


#--------------check over the math again here
# is there a way to update matrix from disk?

# should split up this function
def nn_embed_dist(data, tau=10, max_dim=5, method='dist_mat'):
    """Calculates pairwise distance for all "neighbor" points at every dimension,
    separated by time delay tau, up to the maximum specified dimension.

    Parameters
    ----------
    data : array, 1D
        Data to perform delay embedding over.
    tau : int (default=10)
        Delay between subsequent dimensions (units of samples).
    max_dim : int (default=5)
        Maximum dimension up to which delay embedding is performed.
    method : str (default='dist_mat')
        Method for storing pairwise distance matrix:
        'dist_mat': save distance matrix and add to it for each further
            dimension; faster but more memory intensive.
        'dist_point': recompute distance matrix for all dimensions at each
            embedding dimension. Slower (by a factor of max_dim) but not memory
            intensive

    Returns
    -------
    del_R : numpy-array (2D)
        Change in nearest neighbor distance between consecutive dimensions.
    attr_size : numpy-array (2D)
        Nearest neighbor distance relative to attractor radius (std dev).
    nn_Rsq : numpy-array (2D)
        Distance of nearest neighbor to each time point at every dimension.
    nn_idx : numpy-array (2D)
        Index of nearest neighbor to each time point at every dimension.

    References
    ----------
    Kennel et al., 1992. Phys Rev A

    """
    t0 = time.clock()
    # internal constant to check data length against
    MAX_LEN = 30000

    # number of samples involved in calculation use 1 less extra dimensions
    # worth of data so projection can be calculated for the (d+1)th dimension
    num_samples = len(data) - tau * max_dim

    # nearest neighbor distance and index at every dim
    nn_Rsq = np.zeros((num_samples, max_dim))
    nn_idx = np.zeros((num_samples, max_dim))

    # check for too much data, switch methods if so
    if num_samples > MAX_LEN and method is 'dist_mat':
        method = 'dist_point'
        print('Overriding method, too many data points. Use point-wise calculations.')

    t1 = time.clock()
    print(t1-t0)
    # keeping square distance and tacking on new distance for every added dimension
    # faster but much more memory intensive due to square distance matrix
    # not really feasible with more than 30k points due to memory load
    if method == 'dist_mat':
        # pairwise distance matrix R^2, VERY LARGE!
        Rsq = np.zeros((num_samples, num_samples))
        t2 = time.clock()
        print(t2-t1)
        for dim in range(max_dim):
            print('Dim: %i'%dim)
            print(time.clock()-t2)
            t2=time.clock()
            # loop over embedding dimensions and calculate NN dist
            for idx in range(num_samples):
                # add the R^2 from the new dimension
                Rsq[idx, :] += (data[idx + dim * tau] -
                                data[dim * tau:num_samples + dim * tau])**2.
                # set distance to self as inf
                Rsq[idx, idx] = np.inf

            t3=time.clock()
            print('compute distance.')
            print(t3-t2)
            nn_idx[:, dim] = np.argmin(Rsq, axis=1)
            nn_Rsq[:, dim] = np.min(Rsq, axis=1)
            t4=time.clock()
            print('compute neighbors.')
            print(t4-t3)

    elif method == 'dist_point':
        # re-calculate distance for every point with the addition of every added
        # embedding dimension about 4-6 times slower due to loops & recalculating
        # every dimension but won't break computer
        data_vec = np.expand_dims(data[:num_samples], axis=1)
        t2=time.clock()
        print(t2-t1)
        for dim in range(max_dim):
            print('Dim: %i'%dim)
            t2=time.clock()
            # loop over embedding dimensions and calculate NN dist
            if dim is not 0:
                # first vectorize delayed data after the first dimension
                # appending the data after each dim
                data_vec = np.concatenate(
                    (data_vec, np.expand_dims(
                        data[dim * tau:num_samples + dim * tau], axis=1)),
                    axis=1)

            t3=time.clock()
            print('vectorize')
            print(t3-t2)
            for idx in range(num_samples):
                dist_idx = np.sum((data_vec[idx, :] - data_vec)**2., axis=1)
                # set distance with self to infinity
                dist_idx[idx] = np.inf
                # get nearest neighbor index and distance
                nn_idx[idx, dim] = np.argmin(dist_idx)
                nn_Rsq[idx, dim] = np.min(dist_idx)

            t4=time.clock()
            print('compute distance and neighbors.')
            print(t4-t3)

    elif method == 'sklearn':
        # re-calculate distance for every point with the addition of every added
        # embedding dimension about 4-6 times slower due to loops & recalculating
        # every dimension but won't break computer
        data_vec = np.expand_dims(data[:num_samples], axis=1)
        t2=time.clock()
        print(t2-t1)
        for dim in range(max_dim):
            print('Dim: %i'%dim)
            t2=time.clock()
            # loop over embedding dimensions and calculate NN dist
            if dim is not 0:
                # first vectorize delayed data after the first dimension
                # appending the data after each dim
                data_vec = np.concatenate(
                    (data_vec, np.expand_dims(
                        data[dim * tau:num_samples + dim * tau], axis=1)),
                    axis=1)

            t3=time.clock()
            print('vectorize')
            print(t3-t2)
            # compute nearest neighbor index with sklearn.nearestneighbors
            dist, idx = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(data_vec).kneighbors(data_vec)
            nn_Rsq[:,dim] = dist[:,1]**2
            nn_idx[:,dim] = idx[:,1]
            t4=time.clock()
            print('compute distance and neighbors.')
            print(t4-t3)


    t0 = time.clock()
    # now calculate distance criteria for nn going to next dim
    # use std as an estimate of the attractor size
    RA = np.std(data[:num_samples])
    # change in nn distance to the next dim
    del_R = np.zeros((num_samples, max_dim))
    # nn distance relative to attractor size
    attr_size = np.zeros((num_samples, max_dim))

    for dim in range(max_dim):
        # first, get the point and its nearest neighbors at the current dimension and
        # find the projection at the next dimension, i.e. at the next time delay (dim+1)*tau
        # next dimension of points
        p_next_dim = data[(dim + 1) * tau:num_samples + (dim + 1) * tau]
        # next dimension of nn
        nn_next_dim = data[
            [int(idx + (dim + 1) * tau) for idx in nn_idx[:, dim]]]
        # now calculate distance in the n+1 dimension
        dist_next_dim = abs(p_next_dim - nn_next_dim)

        # calculate the distance criteria
        # (distance gained to nn; #1 from Kennel 1992)
        del_R[:, dim] = dist_next_dim / np.sqrt(nn_Rsq[:, dim])
        # (nn distance relative to attractor size; #2 from Kennel 1992)
        attr_size[:, dim] = np.sqrt(dist_next_dim**2 + nn_Rsq[:, dim]) / RA

    print('compute delR')
    print(time.clock()-t0)
    return del_R, attr_size, nn_Rsq, nn_idx


def nn_attractor_dim(del_R, attr_size, pffn_thr=0.01, R_thr=15., A_thr=2.):
    """ Calculate proportion of false nearest neighbors based on criteria given
    in Kennel et al., 1992, with threshold for new dimension distance and attractor
    size given by user, as well as proportion cut-off to determine attractor size.

    Parameters
    ----------
    del_R : numpy-array (2D)
        Change in nearest neighbor distance between consecutive dimensions.
    attr_size : numpy-array (2D)
        Nearest neighbor distance relative to attractor radius (std dev).
    pffn_thr : float (default=0.01)
        Threshold for proportion of false nearest neighbors criteria.
    R_thr : float (default=15.)
        Threshold for nearest neighbor distance gained criteria.
    A_thr : float (default=2.)
        Threshold for nearest neighbor relative distance criteria.

    Returns
    -------
    attr_dim : int
        Estimated attractor dimension. -1 if computation did not converge to
        passing both distance criteria at any dimension.
    pffn : array (1D)
        Proportion of false nearest neighbor at each dimension.
    """
    num_samples, max_dim = np.shape(del_R)
    pffn = np.zeros(max_dim)
    for dim in range(max_dim):
        # find proportion of false nearest neighbors by either criteria
        pffn[dim] = np.sum(
            np.logical_or(del_R[:, dim] > R_thr, attr_size[:, dim] >
                          A_thr)) * 1. / num_samples

    if np.where(pffn <= pffn_thr)[0].size:
        # find the first dimension at which percent false nearest neighbors
        # is less than the threshold
        attr_dim = np.where(pffn <= pffn_thr)[0][0] + 1
    else:
        attr_dim = -1

    return attr_dim, pffn
#
#
# def predict_at(X_train, X_test, tau=10, dim=3, future=10, nn=10, fit_method='mean'):
#     """
#     Construct the attractor in state space using delay embedding with the training data
#     and predict future values of the test data using future values of the nearest neighbors
#     in the training data, for a specific set of parameter values.
#
#     returns prediction and validation time series
#     """
#     if X_test is 'none':
#         # if no test vector is given, train on self
#         self_train = True
#         X_test = np.array(X_train)
#     else:
#         self_train = False
#
#     ns_train = len(X_train) - max(dim * tau, future)
#     ns_test = len(X_test) - max(dim * tau, future)
#
#     # pairwise distance matrix R^2
#     Rsq = np.zeros((ns_train, ns_test))
#     if self_train:
#         # set diagonal to inf if training using test set
#         Rsq[np.diag_indices(ns_test)] = np.inf
#
#     for d in range(dim):
#         # loop over embedding dimensions and calculate NN dist
#         for idx in range(ns_test):
#             # add the R^2 from the new dimension
#             Rsq[:, idx] += (X_test[idx + d * tau] -
#                             X_train[d * tau:ns_train + d * tau])**2.
#
#     # get nn nearest neighbors
#     nn_idx = np.argsort(Rsq, axis=0)[:nn, :]
#     val = X_test[future:future + ns_test]
#     if fit_method is 'mean':
#         # take average of nearest neighbor's future values
#         pred = np.mean(X_train[nn_idx[:nn, :] + future], axis=0)
#
#     return pred, val
#
#
# def delay_embed_forecast(X_train,
#                          X_test='none',
#                          tau=10,
#                          max_dim=8,
#                          max_future=25,
#                          max_nn=20,
#                          fit_method='mean'):
#     """
#     construct the attractor in state space using delay embedding with the training data
#     and predict future values of the test data using future values of the nearest neighbors
#     in the training data
#     """
#     if X_test is 'none':
#         # if no test vector is given, train on self
#         self_train = True
#         X_test = np.array(X_train)
#     else:
#         self_train = False
#
#     ns_train = len(X_train) - max(dim * tau, future)
#     ns_test = len(X_test) - max(dim * tau, future)
#
#     # pairwise distance matrix R^2
#     Rsq = np.zeros((ns_train, ns_test))
#     if self_train:
#         # set diagonal to inf if training using test set
#         Rsq[np.diag_indices(ns_test)] = np.inf
#
#     rho = np.zeros((max_dim, max_future, max_nn))
#     rmse = np.zeros((max_dim, max_future, max_nn))
#     for dim in range(max_dim):
#         # loop over embedding dimensions and calculate NN dist
#         for idx in range(ns_test):
#             # add the R^2 from the new dimension
#             Rsq[:, idx] += (X_test[idx + dim * tau] -
#                             X_train[dim * tau:ns_train + dim * tau])**2.
#
#         # get nn_max nearest neighbors
#         nn_idx = np.argsort(Rsq, axis=0)[:max_nn, :]
#         for future in range(max_future):
#             val = X_test[future + 1:future + 1 + ns_test]
#             # make prediction for [1:max_future] steps into the future...
#             for nn in range(max_nn):
#                 # using 1 to max_nn number of neighbors
#                 if fit_method is 'mean':
#                     # take average of nearest neighbor's future values
#                     pred = np.mean(
#                         X_train[nn_idx[:nn + 1, :] + future + 1], axis=0)
#
#                 rho[dim, future, nn] = np.corrcoef(pred, val)[0, 1]
#                 rmse[dim, future, nn] = np.sqrt(np.mean((pred - val)**2.))
#
#     return rho, rmse
