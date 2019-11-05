import numpy as np
from scipy import io, signal
import pandas as pd
from scipy.integrate import odeint


def _lorenz(X, t, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

def sim_lorenz(T, fs, init, args=(10, 2.667, 28)):
    """Simulate a Lorenz attractor

    Parameters
    ----------
    T : float
        Signal length.
    fs : float
        Sampling rate.
    init : tuple
        Initial values.
    args : tuple
        Parameter values for sigma, beta, rho. Defaults to chaos values.

    Returns
    -------
    type
        Description of returned object.

    """
    t = np.arange(0,T,1/fs)
    f = odeint(_lorenz, init, t, args)
    x, y, z = f.T
    return x,y,z

def load_mouse_data(datafolder, i_m, return_type='binned', bin_width=0.01, smooth_param=[0.2, 0.025]):
    """ Load neuropixel data from Stringer et al., 2019

    Parameters
    ----------
    datafolder : str
        Location of data.
    i_m : int
        Which mouse (0,1,2)
    return_type : str
        Return as 'spiketimes', 'binned' (default), or 'smoothed'.
    bin_width : float
        Bin width in seconds, defaults 0.01s.
    smooth_param : [float, float]
        Smoothing parameter for Gaussian window, [win_len, std].

    Returns
    -------
        data, cluster_info.
    """
    mice = ['Krebs', 'Waksman', 'Robbins']
    cluLocs = io.loadmat(datafolder+'cluLoc.mat', squeeze_me = True)
    probe_id = cluLocs['iprobeAll']
    probe_loc = cluLocs['brainLocNameAll']

    # load data and cluster info
    mouse_data = io.loadmat(datafolder+'spks/spks%s_Feb18.mat'%mice[i_m], squeeze_me = True)['spks']
    clu_info = pd.DataFrame(np.array([probe_id[i_m],probe_loc[i_m]]).T, columns=['probe', 'region'])

    print('Grabbing Spikes...')
    spikes_all = []
    for probe in range(len(mouse_data)):
        st = mouse_data[probe][0]
        clu = mouse_data[probe][1]
        # add spike time into each
        spikes_all += [np.sort(st[clu==k]) for k in np.unique(clu)]

    if return_type is 'spiketimes':
        return spikes_all, clu_info

    print('Binning Spikes...')
    t_beg, t_end = _beg_end(spikes_all)
    t_bins = np.arange(t_beg,t_end,bin_width)
    spk_binned = [np.histogram(spks,t_bins)[0] for spks in spikes_all]
    df_spk = pd.DataFrame(np.array(spk_binned).T, index=t_bins[:-1])

    # compute populate rate in various areas
    pop_rate = df_spk.sum(1)
    for reg, grp in clu_info.groupby('region'):
        df_spk.insert(0,reg,df_spk[grp.index].sum(1))
    df_spk.insert(0,'all',pop_rate)

    if return_type is 'binned':
        return df_spk, clu_info

    if return_type is 'smoothed':
        print('Smoothing...')
        win_len, win_std = smooth_param
        win = signal.windows.gaussian(int(win_len/bin_width)+1, win_std/bin_width)
        win/=win.sum()
        bin_smoothed = signal.convolve(df_spk, win[:,None], mode='same')
        df_spk_smo = pd.DataFrame(bin_smoothed, index=t_bins[:-1])
        return df_spk_smo, clu_info

def return_pops(data, df_info):
    pop_list, region_labels = [], []
    for reg, grp in df_info.groupby('region'):
        if type(data) is type(pd.DataFrame()):
            # dataframe is passed in, all good
            pop_list.append(np.squeeze(data[grp.index.values].values))
        else:
            # assume the group indices line up with the data array indices
            pop_list.append(np.squeeze(data[grp.index.values]))
        region_labels.append(reg)
    return pop_list, region_labels

def _beg_end(spikes_all):
    spikes_concat = np.concatenate(spikes_all)
    return np.floor(spikes_concat.min()), np.ceil(spikes_concat.max())
