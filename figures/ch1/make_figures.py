import numpy as np
np.random.seed(123)

import matplotlib
matplotlib.rcParams.update({'font.sans-serif' : 'Helvetica',
                            'axes.labelsize': 10,
                            'xtick.labelsize' : 6,
                            'ytick.labelsize' : 6,
                            'axes.titlesize' : 10})

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from hips.plotting.layout import create_figure, create_axis_at_location


import seaborn as sns
color_names = ["windows blue",
               "amber",
               "crimson",
               "faded green",
               "dusty purple",
               "greyish"]
colors = sns.xkcd_palette(color_names)
sns.set(style="white", palette=sns.xkcd_palette(color_names))


from hips.plotting.colormaps import harvard_colors, gradient_cmap
#colors = harvard_colors()

def make_figure_1():
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

    fig = create_figure((5.5, 2.7))
    ax = create_axis_at_location(fig, .75, .5, 4., 1.375)
    ymax = 105
    # Plot the rates
    for i in range(n):
        ax.add_patch(Rectangle([i*D,0], D, lmbda[z[i]],
                               color=colors[z[i]], ec="none", alpha=0.5))
        ax.plot([i*D, (i+1)*D], lmbda[z[i]] * np.ones(2), '-k', lw=2)

        if i < n-1:
            ax.plot([(i+1)*D, (i+1)*D], [lmbda[z[i]], lmbda[z[i+1]]], '-k', lw=2)
            
        # Plot boundaries
        ax.plot([(i+1)*D, (i+1)*D], [0, ymax], ':k', lw=1)
        
        
    # Plot x axis
    plt.plot([0,T], [0,0], '-k', lw=2)

    # Plot spike times
    for s in Ss:
        plt.plot([s,s], [0,60], '-ko', markerfacecolor='k', markersize=5)

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
    ax.yaxis.labelpad = 30
    ax.set_ylabel("${s_t}$", rotation=0,  verticalalignment='center')

    ## Now plot the latent state above that above
    ax = create_axis_at_location(fig, .75, 2.375, 4., .25)
    for i in xrange(n):
        # Plot boundaries
        ax.add_patch(Rectangle([i*D,0], D, 10,
                            color=colors[z[i]], ec="none", alpha=0.5))

        ax.plot([(i+1)*D, (i+1)*D], [0, 10], '-k', lw=1)
        ax.text(i*D + D/3.5, 3, "u" if z[i]==0 else "d", fontdict={"size":9})
    ax.set_xlim(0,T)
    ax.set_ylim(0,10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.yaxis.labelpad = 30
    ax.set_ylabel("${z_t}$", rotation=0,  verticalalignment='center')

    
    fig.savefig("figure1.pdf")
    fig.savefig("figure1.png", dpi=300)
    
    plt.show()

def make_figure_2():
    # Make a figure of probabilistic models for matrix factorization
    T = 50
    N = 25
    K = 5
    ht = 1.25

    cmap = gradient_cmap([colors[1], np.ones(3), colors[0]])
    kr = 20
    
    # Mixture model
    zint = np.random.randint(K, size=(T,))
    Z = np.zeros((T,K))
    Z[np.arange(T), zint] = 1
    mu = .2*np.random.randn(K) 
    C = mu[:,None] + np.random.randn(K,N)
    
    fig = create_figure((1.8, 1.8))
    ax = create_axis_at_location(fig, .25, .25, .5, ht)
    ax.imshow(np.kron(Z, np.ones((kr,kr))), cmap="Greys", interpolation="none")
    plt.yticks([(T-1)*kr], ["$T$"])
    plt.xticks([(K-1)*kr], ["$K$"])
    ax.set_ylabel("$Z$", rotation=0)
    

    ax = create_axis_at_location(fig, .7, 1.2, ht*(float(N)/T), .5)
    ax.imshow(np.kron(C, np.ones((kr,kr))), cmap=cmap, interpolation="none",
              vmin=-abs(C).max(), vmax=abs(C).max())
    ax.yaxis.tick_right()
    ax.set_title("$C^{\\mathsf{T}}$")
    plt.yticks([(K-1)*kr], ["$K$"])
    plt.xticks([(N-1)*kr], ["$N$"])

    fig.savefig("figure2a.pdf")

    
    plt.show()

if __name__ == "__main__":
    make_figure_1()
