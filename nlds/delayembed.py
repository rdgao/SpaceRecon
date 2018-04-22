import numpy as np
import scipy as sp
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors


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
    MI : float
        Mutual information estimate.
    """
    c_xy = np.histogram2d(x, y, bins)[0]
    return mutual_info_score(None, None, contingency=c_xy)


def compute_delay_MI(data, bins, max_tau, tau_step=1):
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
    AC_timepoints : array, 1D
        Time points (in samples) at which correlation was computed.
    AC : array, 1D
        Time lagged (auto)correlation.

    """

    AC_timepoints = np.arange(0, max_lag, lag_step)
    AC = np.zeros(len(AC_timepoints))
    AC[0] = np.sum((data - np.mean(data))**2)
    for ind, lag in enumerate(AC_timepoints[1:]):
        AC[ind + 1] = np.sum((data[:-lag] - np.mean(data[:-lag]))
                             * (data[lag:] - np.mean(data[lag:])))

    return AC_timepoints, AC / AC[0]


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


def delay_embed(data, tau, max_dim):
    """Delay embed data by concatenating consecutive increase delays.

    Parameters
    ----------
    data : array, 1-D
        Data to be delay-embedded.
    tau : int (default=10)
        Delay between subsequent dimensions (units of samples).
    max_dim : int (default=5)
        Maximum dimension up to which delay embedding is performed.

    Returns
    -------
    x : array, 2-D (samples x dim)
        Delay embedding reconstructed data in higher dimension.

    """
    num_samples = len(data) - tau * (max_dim - 1)
    return np.array([data[dim * tau:num_samples + dim * tau] for dim in range(max_dim)]).T


def compute_nn_dist(data, tau=10, max_dim=5):
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

    Returns
    -------
    del_R : numpy-array (2D)
        Change in nearest neighbor distance between consecutive dimensions.
    rel_R : numpy-array (2D)
        Nearest neighbor distance relative to attractor radius (std dev).

    References
    ----------
    Kennel et al., 1992. Phys Rev A
    """
    # get embedded data up to the highest dimension
    data_vec = delay_embed(data, tau=tau, max_dim=max_dim + 1)
    num_samples = data_vec.shape[0]

    # std as an estimate of the attractor size (see ref)
    RA = np.std(data_vec[:, 0])

    # nearest neighbor distance and index at every dim
    del_R = np.zeros_like(data_vec)[:, :-1]  # change in nn distance
    # relative nn distance (wrt attractor size)
    rel_R = np.zeros_like(data_vec)[:, :-1]

    for dim in range(max_dim):
        # compute nearest neighbor index with sklearn.nearestneighbors
        dist, idx = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(
            data_vec[:, :(dim + 1)]).kneighbors(data_vec[:, :(dim + 1)])

        # compute the increase in neighbor distance going to the next dim
        next_d = dim + 1
        ndist = data_vec[:, next_d] - data_vec[idx[:, 1].astype('int'), next_d]
        # gain in neighbor distance relative to current distance
        del_R[:, dim] = abs(ndist) / dist[:, 1]
        # distance of current dim's neighbors in the next dimension
        # relative to attractor size
        rel_R[:, dim] = (ndist**2 + dist[:, 1]**2)**0.5 / RA

    return del_R, rel_R


def compute_attractor_dim(del_R, rel_R, pfnn_thr=0.01, R_thr=15., A_thr=2.):
    """ Calculate proportion of false nearest neighbors based on criteria given
    in Kennel et al., 1992, with threshold for new dimension distance and attractor
    size given by user, as well as proportion cut-off to determine attractor size.

    Parameters
    ----------
    del_R : numpy-array (2D)
        Change in nearest neighbor distance between consecutive dimensions.
    rel_R : numpy-array (2D)
        Nearest neighbor distance relative to attractor radius (std dev).
    pfnn_thr : float (default=0.01)
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
    pfnn : array (1D)
        Proportion of false nearest neighbor at each dimension.
    """
    num_samples, max_dim = del_R.shape
    pfnn = np.zeros(max_dim)
    for dim in range(max_dim):
        # find proportion of false nearest neighbors by either criteria
        crit_1 = del_R[:, dim] > R_thr
        crit_2 = rel_R[:, dim] > A_thr
        pfnn[dim] = np.sum(np.logical_or(crit_1, crit_2)) / num_samples

    if np.where(pfnn <= pfnn_thr)[0].size:
        # find the first dimension at which percent false nearest neighbors
        # is less than the threshold
        attr_dim = np.where(pfnn <= pfnn_thr)[0][0] + 1
    else:
        attr_dim = -1

    return attr_dim, pfnn


def pfnn_de_dim(data, tau=10, max_dim=5, pfnn_thr=0.01, R_thr=15., A_thr=2.):
    """Proportion of False Nearest Neighbor method for determining Delay Embedding
    Attractor Dimension. (Kennel et al., 1992).

    Basically just an utility function that calls the two functions in one go:
        - compute_nn_dist: compute delay-embedded nearest neighbor distances
        - compute_attractor_dim: use distances to determine proportion of false neighbors

    Parameters
    ----------
    data : array, 1D
        Data to perform delay embedding over.
    tau : int (default=10)
        Delay between subsequent dimensions (units of samples).
    max_dim : int (default=5)
        Maximum dimension up to which delay embedding is performed.
    pfnn_thr : float (default=0.01)
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
    pfnn : array (1D)
        Proportion of false nearest neighbor at each dimension.
    """
    del_R, rel_R = compute_nn_dist(data, tau=tau, max_dim=max_dim)
    attr_dim, pfnn = compute_attractor_dim(del_R=del_R, rel_R=rel_R)
    return attr_dim, pfnn

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
