import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def plot_statespace_trial(x, ax=None, step=1, lc='k', alpha=0.8, mark_ind=None, mark_color=None, ms=20):
    """Plot a single statespace trajectory, in 2D or 3D.

    Parameters
    ----------
    x : np array, 2D
        State variables to plot (signal), [dim x len].
    ax : mpl axes handle
        Axes handle to plot on. If pre-generated, enables plotting of multiple trajectories.
    lc : str
        Line color for trajectory plot.
    alpha : float
        alpha value for trajectory plot.
    mark_ind : list of int
        List of indices to plot as circles to highlight instants.
    mark_color : List
        List of colors for highlighted instants.

    """
    if mark_ind is not None and mark_color is None:
        # if no marker color specified, use default
        #mark_color = [None]*len(mark_ind)
        mark_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(mark_ind)]

    sig_len, ndim = x.shape
    if ndim == 2:
        if ax is None:
            ax = plt.subplot(projection=None)

        ax.plot(x[::step,0], x[::step,1], '-', color=lc, alpha=alpha, lw=1)
        if mark_ind is not None:
            ax.scatter(x[mark_ind,0], x[mark_ind,1], marker='o', c=mark_color, s=ms)

    elif ndim > 2:
        if ax is None:
            ax = plt.subplot(projection='3d')

        ax.plot3D(x[::step,0], x[::step,1], x[::step,2], '-', color=lc, alpha=alpha, lw=1)
        if mark_ind is not None:
            ax.scatter(x[mark_ind,0], x[mark_ind,1], x[mark_ind,2], marker='o', c=mark_color, s=ms)

        ax.grid('off')

    return ax
