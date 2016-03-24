import numpy as np
np.random.seed(123)

import matplotlib
matplotlib.rcParams.update({'font.sans-serif' : 'Helvetica',
                            'axes.labelsize': 9,
                            'xtick.labelsize' : 9,
                            'ytick.labelsize' : 9,
                            'axes.titlesize' : 9})

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from hips.plotting.layout import create_figure, create_axis_at_location
from hips.plotting.colormaps import harvard_colors
colors = harvard_colors()

#def make_figure_1():
if __name__ == "__main__":
    """
    Simple mixture model example
    """
    T = 1000
    D = 50
    n = T // D
    lmbda = np.array([100, 10])

    # Simulate latent states
    z = np.random.rand(n) < 0.5
    
    # Simulate real valued spike times
    Ss = []
    Ns = np.zeros(n)
    for i in np.arange(n):
        rate = lmbda[z[i]]
        Ns[i] = np.random.poisson(rate * 0.05)
        Ss.append(i * D + np.random.rand(Ns[i]) * D)

    Ss = np.concatenate(Ss)

    fig = create_figure((5.5, 2.4))
    ax = create_axis_at_location(fig, .75, .5, 4., 1.375)
    ymax = 105
    # Plot the rates
    for i in range(n):
        ax.add_patch(Rectangle([i*D,0], D, lmbda[z[i]],
                               color=colors[z[i]], ec="none", alpha=0.25))
        ax.plot([i*D, (i+1)*D], lmbda[z[i]] * np.ones(2), '-k', lw=2)

        if i < n-1:
            ax.plot([(i+1)*D, (i+1)*D], [lmbda[z[i]], lmbda[z[i+1]]], '-k', lw=2)
            
        # Plot boundaries
        ax.plot([(i+1)*D, (i+1)*D], [0, ymax], ':k', lw=1)
        
        
    # Plot x axis
    plt.plot([0,T], [0,0], '-k', lw=2)

    # Plot spike times
    for s in Ss:
        plt.plot([s,s], [0,60], '-ko', markerfacecolor='k', markersize=4)

    plt.xlabel("time [ms]")
    plt.ylabel("firing rate [Hz]")
    plt.xlim(0,T)
    plt.ylim(-5,ymax)

    ## Now plot the spike count above
    ax = create_axis_at_location(fig, .75, 2., 4., .25)
    for i in xrange(n):
        # Plot boundaries
        ax.plot([(i+1)*D, (i+1)*D], [0, 10], '-k', lw=1)
        ax.text(i*D + D/3.5, 3, "%d" % Ns[i], fontdict={"size":9})
    ax.set_xlim(0,T)
    ax.set_ylim(0,10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.yaxis.labelpad = 27
    ax.set_ylabel("$s$")

    fig.savefig("figure1.pdf")
    fig.savefig("figure1.png", dpi=300)
    
    plt.show()
        
