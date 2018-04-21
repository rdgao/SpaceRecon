import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, firwin
import time


def sim_powerlaw_signal(T, fs, exponent):
    """ Generate a power law time series by spectrally rotating a white noise.

    Parameters
    ----------
    T : float, seconds
        Simulation time.
    fs : float, Hz
        Sampling rate of simulated signal.
    exponent : float
        Desired power-law exponent; alpha in P(f)=f^alpha;

    Returns
    -------
    x : array, 1-D
        Time-series with the desired power-law exponent.

    """
    sig_len = int(T*fs)
    x = np.random.randn(sig_len)
    x_ = rotate_powerlaw(x, fs, delta_f=exponent, f_rotation=0)
    return sp.stats.zscore(x_)

def rotate_powerlaw(data, fs, delta_f, f_rotation=30):
    """Takes a time series and changes its power law exponent via rotation in
    the spectral domain.

    Parameters
    ----------
    data : array, 1-D
        Time-series to be rotated.
    fs : float, Hz
        Sampling rate.
    delta_f : float
        Change in power law exponent to be applied. Positive is counterclockwise
        rotation (flatten), negative is clockwise rotation (steepen).
    f_rotation : float, Hz
        Axis of rotation frequency, such that power at that frequency is unchanged
        by the rotation. Only matters if not further normalizing signal variance.

    Returns
    -------
    x : array, 1-D
        Power-law rotated time-series.

    """

    # compute FFT and frequency axis
    FC = np.fft.fft(data)
    f_axis = np.fft.fftfreq(len(data), 1./fs)

    # make the 1/f mask
    f_mask = np.zeros_like(f_axis)
    f_mask[1:] = 10**(np.log10(np.abs(f_axis[1:]))*delta_f)
    f_mask[0]=1.

    # normalize power at rotation frequency
    f_mask = f_mask/f_mask[np.where(f_axis>=f_rotation)[0][0]]

    return np.real(np.fft.ifft(FC*f_mask))



def inst_cf(data, fs=1000., oscBand=[7., 12.], winLen=1000, stepLen=500):
    """
    Computes instantaneous frequency & other metrics in a single time series
    ----- Args -----
    data : array, 1d
        time series to calculate center frequency, bandwidth
        and power for

    fs : float, Hz
        sampling frequency

    oscBand : (low, high) Hz
        frequency range of bandpassed oscillation
        default: [7,12]

    winLen : int, samples
        size of sliding window to compute stats
        if 0, return raw estimates
        default: 1000

    stepLen : int, samples
        step size to advance sliding window
        default: 500

    ----- Returns -----
    pw: array
        instantaneous power over time

    cf: array
        center frequency over time

    bw: array
        bandwidth, defined as IQR of instantaneous freq, [] if winLen == 1

    """
    o_msg = False
    # design filter
    Ntaps = np.ceil(3 * fs / oscBand[0])
    # force to be odd to avoid spreading peak
    if Ntaps % 2 == 0:
        Ntaps = Ntaps + 1

    # Perform filtering
    taps = firwin(Ntaps, np.array(oscBand) / (fs / 2.), pass_zero=False)
    if o_msg:
        print 'Filtering...'
    #filtered = filtfilt(taps, [1], data)  # two-pass filtering
    filtered = np.convolve(taps, data, 'same')  # one-pass, conv filtering
    if o_msg:
        print 'Computing Hilbert, Power & Phase...'
    # compute signal derivatives
    HT = sp.signal.hilbert(filtered)  # calculate hilbert
    IF = np.diff(np.unwrap(np.angle(HT))) * fs / (
        2 * np.pi)  # calculate instantaneous frequency
    PW = np.log10(abs(HT)**2)

    # moving average & std to compute CF, BW, and PW
    if winLen == 1:
        # window size=1, no filtering
        return PW, IF, []
    else:
        if o_msg:
            print 'Smoothing...'

        # compute output length & pre-allocate array
        outLen = int(np.ceil((np.shape(data)[0] - winLen) / float(
            stepLen))) + 1
        pw, cf, bw = np.zeros((3, outLen))

        # get smoothed power
        wins = slidingWindow(PW, winLen, stepLen)
        for ind, win in enumerate(wins):
            pw[ind] = np.mean(win)

            # get smoothed center freq & bandwidth
        wins = slidingWindow(IF, winLen, stepLen)
        for ind, win in enumerate(wins):
            cf[ind] = np.mean(win)  # smooth center freq
            bw[ind] = np.diff(np.percentile(win, q=[25, 75]), axis=0)

        return pw, cf, bw


def slidingWindow(data, winLen=1000, stepLen=500):
    """
    Returns a generator that will iterate through
    the defined lengths of 1D array with window of
    length winLen and step length of stepLen;
    if not a perfect divisor, last slice gets remaining data
    --- Args ---
    data : array, 1d
        time series to slide over

    winLen : int, samples
        size of sliding window
        default: 1000

    stepLen : int, samples
        step size of window
        default: 500

    --- Return ---
    generator with slices of data
        channel x winLen
    """

    # Pre-compute number of length of output
    # last slice gets whatever data is remaining
    outLen = int(np.ceil((np.shape(data)[0] - winLen) / float(stepLen))) + 1
    # make it rain
    for ind in range(outLen):
        yield data[stepLen * ind:winLen + stepLen * ind]


def bin_spikes(spikeTimes, binnedLen=-1, spkRate=40000., binRate=1000.):
    """
    Takes a vector of spike times and converted to a binarized array, given
    pre-specified spike sampling rate and binned rate
    example use:
        bsp = utils.bin_spikes(spk_times, spkRate=20000., binRate=1250.)
        for ind, win in enumerate(utils.slidingWindow(bsp, winLen=1250, stepLen=1250/2)):
            smob[ind] = sum(win[0])
    --- Args ---
    spikeTimes : array, 1d
        list of spike times

    binnedLen : int
        length of binarized spike train in samples, can be constrained
        by matching LFP vector length
        default: -1, end determined by last spike-time

    spkRate : float
        sampling rate of spike stamps, 1. if spike time vector contain
        actual time stamps of spikes, instead of indices
        default: 40000.

    binRate : float
        rate at which binarized spike train is sampled at
        default: 1000.

    --- Return ---
    binary array of spikes (float)
    """
    if binnedLen == -1:
        # no specified time to truncate, take last spike time as end
        t_end = int(round(spikeTimes[-1] / spkRate * binRate)) + 1
    else:
        # length of binary vector is predetermined
        t_end = binnedLen

    # make binary bins
    bsp = np.zeros(t_end)
    # convert spike index to downsampled index
    inds = spikeTimes / spkRate * binRate
    # truncate spikes to match end time & make int

    for i in inds:
        bsp[i] += 1
    # bsp[inds[inds < t_end].astype(int)] += 1
    return bsp


def smooth_events(eventTimes, values=None, fs=1., winLen=1., stepLen=0.5):
    """
    Takes a vector of event times (or samples) and compute rolling window
    event count (or average ) over the events.
    --- Args ---
    eventTimes: array, 1d
        array of event times (or sample indices)

    values: array, 1d
        array of event values
        default: None

    fs: float
        sampling rate of eventTimes
        default: 1. i.e. represents time in seconds

    winLen: float
        time length of window
        default: 1 second

    stepLen: float
        time length of stepping
        default: 0.5 seconds

    --- Return ---
    array of smoothed values (float)

    """
    # change event indices into timestamps by dividing by fs
    eventTimes = eventTimes * 1. / fs
    outLen = int(np.ceil((eventTimes[-1] - winLen) / float(stepLen))) + 1
    smoothed = np.zeros(outLen)
    if values is None:
        # no values attached, just count how many occurrences (probably spikes)
        for ind in range(outLen):
            smoothed[ind] = np.shape(eventTimes[(eventTimes >= ind * stepLen) &
                                                (eventTimes < winLen + ind *
                                                 stepLen)])[0]
    else:
        # average the values
        for ind in range(outLen):
            smoothed[ind] = np.mean(values[(eventTimes >= ind * stepLen) & (
                eventTimes < winLen + ind * stepLen)])

    return smoothed


def corrcoefp(matrix):
    """
    Takes in matrix and calculates column-wise pair correlation and p-value
    Does what matlab's corr does
    copied from scipy.stats.pearsonr and adapted for 2D matrix
    --- Args ---
    matrix: array, 2D (dim x sample)
        array to calculate correlation over, every pair of dimensions

    --- Return ---
    r: array, 2D
        correlation matrix

    p: array, 2D
        p-value matrix
    """
    r = np.corrcoef(matrix)
    df = matrix.shape[1] - 2
    t_squared = r**2 * (df / ((1.0 - r) * (1.0 + r)))
    p = sp.special.betainc(0.5 * df, 0.5, df / (df + t_squared))
    np.fill_diagonal(p, 1.0)

    return r, p


def corr_plot(C, labels=None, pv=None, pvThresh=0.01, cmap='RdBu', bounds=None):
    """
    Takes in a correlation matrix and draws it, with significance optional
    --- Args ---
    C: array, 2D square matrix
        correlation matrix to be plotted

    labels: list of strings
        label for each row in correlation matrix
        must match dimension of C

    pv: 2d square matrix
        significance matrix, should match dims of C
        default: None

    pvThresh: float
        threshold value to draw significance stars
        default: 0.01

    cmap: str
        colormap of plotted matrix
        default: 'RdBu' - redblue

    bounds: [val1, val2]
        bounds on the colormap
        default: None

    """
    # fill diagonals to zero
    np.fill_diagonal(C, 0.)
    # define color bounds
    if bounds is None:
        vmin = -1.
        vmax = 1.
    else:
        vmin, vmax = bounds

    nDim = np.shape(C)[0]
    # draw the square
    plt.imshow(C, interpolation='none', cmap='RdBu', vmin=vmin, vmax=vmax)
    if labels is not None:
        plt.xticks(np.arange(len(labels)), labels)
        plt.yticks(np.arange(len(labels)), labels)


    plt.xlim(-0.5, nDim-0.5)
    plt.ylim(nDim-0.5, -0.5)
    plt.plot([-0.5, nDim - 0.5], [-0.5, nDim - 0.5], 'k-')
    plt.colorbar(fraction=0.046, pad=0.04, ticks=np.linspace(-1, 1, 5))
    plt.tick_params(length=0)
    plt.box()

    # star squares that are significant
    if pv is not None:
        sigInds = np.where(pv < pvThresh)
        plt.scatter(sigInds[1], sigInds[0], s=50, marker='*', c='k')
