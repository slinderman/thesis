import numpy as np
np.random.seed(123)

from scipy.misc import logsumexp

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

T = 1000
D = 50
n = T // D

def sample_mixture_model(lmbda, p):
    """
    Simple mixture model example
    """
    # Simulate latent states
    z = np.random.rand(n) < p
    
    # Simulate real valued spike times
    Ss = []
    Ns = np.zeros(n)
    for i in np.arange(n):
        rate = lmbda[z[i]]
        Ns[i] = np.random.poisson(rate * 0.05)
        Ss.append(i * D + np.random.rand(Ns[i]) * D)

    Ss = np.concatenate(Ss)

    return Ns, Ss, z
                             
def draw_mixture_figure(Ns, Ss, z, lmbda, filename="figure1.png", saveargs=dict(dpi=300)):
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

    
    #fig.savefig(filename + ".pdf")
    fig.savefig(filename, **saveargs)

    plt.close(fig)
    #plt.show()

def make_mcmc_figures(Ns, Ss, z0, lmbda0, p=0.5, a_lmbda=1, b_lmbda=1, N_iter=100):

    def _poisson_ll(s, l):
        return -l + s*np.log(l)
    
    def _resample():
        # Resample latent states given lmbda
        for i in xrange(n):
            lp0 = _poisson_ll(Ns[i], lmbda[0]) + np.log(p)
            lp1 = _poisson_ll(Ns[i], lmbda[1]) + np.log(1-p)
            p_0 = np.exp(lp0 - logsumexp([lp0, lp1]))
            z[i] = np.random.rand() < 1-p_0

        # Resample lmbda given z
        for k in [0,1]:
            Nk = (z==k).sum()
            Sk = Ns[z==k].sum()
            a_post = a_lmbda + Sk
            b_post = b_lmbda + Nk
            lmbda[k] = np.random.gamma(a_post, 1./b_post)

    # Now run the Gibbs sampler and save out images
    lmbda = lmbda0.copy()
    z = z0.copy()
    for itr in range(N_iter):
        print "Iteration ", itr
        draw_mixture_figure(Ns, Ss, z, lmbda/0.05, filename="itr_%d.jpg" % itr)
        _resample()
                        
if __name__ == "__main__":
    # Sample data
    lmbda = np.array([100, 10])
    p = 0.5
    Ns, Ss, z = sample_mixture_model(lmbda, p)
    draw_mixture_figure(Ns, Ss, z, lmbda)
    
    z0 = np.random.rand(n) < 0.5
    #z0 = np.zeros(n, dtype=np.bool)
    lmbda0 = np.random.gamma(1,1,size=2)
    #lmbda0 = 1 * np.ones(2)
    make_mcmc_figures(Ns, Ss, z0, lmbda0)
