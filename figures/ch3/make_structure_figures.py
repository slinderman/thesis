import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt

import scipy.ndimage as ndimage
from hips.plotting.layout import create_figure, create_axis_at_location
from hips.plotting.colormaps import harvard_colors, gradient_cmap

header  = 0.2
figwidth = 1.3
figsize = (figwidth, figwidth+header)

def set_axis_black(fig, ax):
    fig.patch.set_color("black")
    ax.patch.set_color("black")
    for sp in ax.spines.values():
        sp.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

def plot_1d_type(c, C, colors, blackbkgd=False, figname="new_clustered_neurons.pdf"):
    N = len(c)

    # Count how many times neurons are assigned to each class
    cnts = np.bincount(c, minlength=C)

    # Plot a dot for each neuron
    fig = plt.figure(figsize=figsize)
    fig.patch.set_alpha(0.0)
    border_color_scale = 1.3 if blackbkgd else 0.6

    for i, cnt in enumerate(cnts):
        for j in xrange(cnt):
            plt.plot(1. * i, 1. * j, ls='', marker='o',
                 markeredgecolor=border_color_scale*colors[i], 
                 markerfacecolor=colors[i],
                 markeredgewidth=2, markersize=10)

    plt.axis("off")
    plt.xlim(-1, C)
    plt.ylim(-1, cnts.max()+1)

    plt.savefig(figname)
    plt.show()

def plot_1d_feature(feat, blackbkgd=False, figname="new_feature_neurons.pdf"):
    # Now consider neurons in a 2D feature space
    # Let angle be the feature
    N = len(feat)
    scale = 1.
    cmap = gradient_cmap([colors[0], colors[3], colors[2], colors[1]])
    tocolor = lambda ff: np.array(cmap(ff))[:-1]

    fig = plt.figure(figsize=(3,1))
    fig.patch.set_alpha(0.0)

    border_color_scale = 1.3 if blackbkgd else 0.6

    for n in xrange(N):
        plt.plot(scale * feat[n], 1, ls='', marker='o',
                 markeredgecolor=border_color_scale*tocolor(feat[n]), 
                 markerfacecolor=tocolor(feat[n]),
                 markeredgewidth=2, markersize=8)

    plt.axis("off")
    plt.xlim(-.1 * scale, 1.1 * scale)
    plt.ylim(0, 2)
    plt.savefig(figname)
    plt.show()

def plot_latent_embedding(L, colors, blackbkgd=False, name="new_clustered_embedded_neurons.pdf"):
    N = L.shape[0]
    import networkx as nx
    from graphistician.adjacency import SBMAdjacencyDistribution
    from hips.plotting.graphs import draw_curvy_network
    
    # Sample a network that connects nearby nodes of the same type
    D = np.sum((L[:,None,:] - L[None,:,:])**2, axis=2)
    p_dist = np.exp(-D/2.0 + 0.5)

    # Also sample by type
    p_sbm = np.zeros((N,N))
    for n1 in xrange(N):
        for n2 in xrange(N):
            if np.allclose(colors[n1],colors[n2]):
                p_sbm[n1,n2] = 1.0

    # Remove self connections
    p = p_dist * p_sbm
    np.fill_diagonal(p, 0)
    A = np.random.rand(N,N) < p

    # Trim connections that are too short
    d_thr = np.percentile(D[A], 50)
    A[D < d_thr] = 0
    
    G = nx.DiGraph(A)

    fig = plt.figure(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax = create_axis_at_location(fig, 0,0, figwidth, figwidth)

    # Plot the x and y axes
    lim = 1.2 * np.ceil(np.max(abs(L)))
    from matplotlib.patches import FancyArrowPatch
    ax_color = "w" if blackbkgd else "k"
    ax.add_patch(FancyArrowPatch(posA=(-lim, 0), color=ax_color, posB=(lim, 0), arrowstyle="<|-|>", mutation_scale=10.))
    ax.add_patch(FancyArrowPatch(posA=(0, -lim), color=ax_color, posB=(0, lim), arrowstyle="<|-|>", mutation_scale=10.))
    
    border_color_scale = 1.3 if blackbkgd else 0.6
    node_edge_colors = [border_color_scale * nc for nc in colors]
    edge_color = "w" if blackbkgd else "k"
    draw_curvy_network(G, L, ax, 
                       node_color=colors, 
                       node_edge_color=node_edge_colors, 
                       node_alpha=1.0, 
                       node_radius=.3, 
                       edge_width=1.,
                       edge_color=edge_color)
    
    # Title
    plt.suptitle("Network", fontsize=9)
    
    # ax.set_title("Latent Types $z$", fontsize=10)
    plt.axis("off")
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.savefig(name)
    plt.show()

def plot_fluorescence(X, colors):
    # Plot the time series of activity
    fig = plt.figure(figsize=figsize)
    fig.patch.set_alpha(0.0)
    plt.plot(X.T + 2*np.arange(N), color=colors[n], lw=2)
    plt.axis("off")
    plt.savefig("basic_fluorescence.pdf")
    plt.show()

# Plot the network 
def plot_network(N, c, C, colors, blackbkgd=False):
    import networkx as nx
    from graphistician.adjacency import SBMAdjacencyDistribution
    from hips.plotting.graphs import draw_curvy_network

    # Sort nodes, sample graph
    cs = np.sort(c).astype(np.int)
    p = 0.15 * np.eye(C, k=1)
    p[-1,0] = 0.15
    A = SBMAdjacencyDistribution(N=N, C=C, c=cs, p=p).rvs().astype(np.int)
    G = nx.DiGraph(A)
    pos = nx.circular_layout(G)

    fig = plt.figure(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax = create_axis_at_location(fig, 0, 0, figsize[0], figsize[1])
    #ax = fig.add_subplot(111)
    ax.patch.set_alpha(0.0)
    
    node_colors = [colors[cc] for cc in cs]
    border_color_scale = 1.3 if blackbkgd else 0.6
    node_edge_colors = [border_color_scale * nc for nc in node_colors]
    edge_color = "w" if blackbkgd else "k"
    draw_curvy_network(G, pos, ax, 
                       node_color=node_colors, 
                       node_edge_color=node_edge_colors, 
                       node_alpha=1.0, 
                       node_radius=0.05, 
                       edge_width=1.5,
                       edge_color=edge_color)
    
    #ax.autoscale()
    #ax.set_title("Functional Network $W$", fontsize=10)

    plt.axis("equal")
    plt.axis("off")
    plt.savefig("clustered_network.pdf")
    plt.show()


# Plot a set of clustered firing rates 
def plot_activation(lmbda, Ss, colors, n_to_plot, xlim=None,
                    draw_activation=True, draw_spikes=True,
                    title=None,
                    name="new_clustered_rates.pdf",
                    blackbkgd=False):

    T = lmbda.shape[0]
    tt = np.arange(T)
    lmax = lmbda.max()

    if xlim is None:
        xlim = (0,T)

    from matplotlib.patches import Polygon
    fig = plt.figure(figsize=figsize)
    #ax = create_axis_at_location(fig, 0, 0, figsize[0], figsize[1])
    fig.patch.set_alpha(0.0)

    for i,n in enumerate(n_to_plot):
        ax = plt.subplot(len(n_to_plot),1,i+1)
        ax.patch.set_alpha(0.0)
        
        # If this is the second to last plot, just plot dots
        if i == len(n_to_plot) - 2:
            dot_color = "w" if blackbkgd else "k"
            plt.axis("off")
            ax.plot(T/2. * np.ones(3), np.linspace(.2, lmax, 3), 'o',
                    ls="none", markersize=2, 
                    markerfacecolor=dot_color, markeredgecolor=dot_color)
            ax.set_xlim(xlim)
            ax.set_ylim(0, 1.25*lmax)

            continue


        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xticklabels([])
        ax.set_xticks([])
        # ax.set_yticks([0, 20, 40])
        ax.set_yticks([])
        ax.set_yticklabels([])

        if blackbkgd:
            set_axis_black(fig, ax)
        border_color_scale = 1.3 if blackbkgd else 0.6

        if draw_activation:
            env = np.zeros((T*2,2))
            env[:,0] = np.concatenate((tt, tt[::-1]))
            env[:,1] = np.concatenate((lmbda[:,n], np.zeros(T)))

            ax.add_patch(Polygon(env, facecolor=colors[n], 
                                 edgecolor=border_color_scale*colors[n], 
                                 alpha=0.5, linewidth=2))

            ax.set_xlim(xlim)
            ax.set_ylim(0, 1.25*lmax)

        if draw_spikes:
            spk_color = "w" if blackbkgd else "k"
            for s in Ss[n]:
                ax.plot([s,s], [0, lmax], '-o', color=spk_color,  markersize=2, 
                        markeredgecolor=spk_color, markerfacecolor=spk_color)

            ax.set_xlim(xlim)
            ax.set_ylim(0, 1.25*lmax)

        # Labels
        # ax.set_ylabel("$\\psi_{%d}(t)$" % (i+1), fontsize=8, rotation=90)
        #if i == 0:
        #    ax.set_ylabel("Cell 1", fontsize=10, rotation=90)

        if i == len(n_to_plot)-1:
            ax.set_xlabel("time", fontsize=10, labelpad=0)
            #ax.set_ylabel("Cell N", fontsize=10, rotation=90)


    if title:
        fig.suptitle(title, fontsize=9)

    plt.savefig(name)
    plt.show()


if __name__ == "__main__":
    colors =  harvard_colors()

    # Make a simple synthetic dataset
    N = 20
    T = 100
    S = (np.random.rand(N,T) < 0.1).astype(np.float)
    f = np.exp(-np.arange(20) / 10.)[None,:]
    F = ndimage.convolve(S,f)
    X = F + 0.25*np.random.randn(N,T)

    # Assign the neurons types
    C = 4
    c = np.random.randint(0, C, N)
    #plot_1d_type(c, C, colors, blackbkgd=True, figname="new_clustered_neurons_black.pdf")

    # Give the neurons 1D features
    feat = np.linspace(0, 1, N)
    #plot_1d_feature(feat, blackbkgd=True, figname="new_feature_neurons_black.pdf")

    # Embed the neurons in 2D space
    L = np.zeros((N,2))
    rings = [4,6,10]
    offset = 0
    for j, ringsize in enumerate(rings):
        for i in xrange(ringsize):
            rad = j + 1
            th = j + 0.1 + i * 2 * np.pi / ringsize
            L[offset + i, 0] = rad * np.cos(th)
            L[offset + i, 1] = rad * np.sin(th)

        offset += ringsize
    L = L + 0.1 * np.random.randn(*L.shape)

    # Plot the embedding in gray
    #plot_latent_embedding(L,
    #                      [colors[6] for n in xrange(N)],
    #                      name="new_embedded_neurons_black.pdf")

    # Now plot the embedding with colors to indicate types
    plot_latent_embedding(L,
                          [colors[c[n]] for n in xrange(N)],
                          blackbkgd=False,
                          name="new_clustered_embedded_neurons.pdf")

    # Plot the activation
    n_to_plot = [0,2,7,None,9]
    #plot_activation(F.T, None, [colors[c[n]] for n in xrange(N)], n_to_plot,
    #                draw_spikes=False, xlim=(0,100),
    #                blackbkgd=False,
    #                title="Activation",
    #                name="new_activation.pdf")

    # Plot the noisy activation
    #plot_activation(X.T, None, [colors[c[n]] for n in xrange(N)], n_to_plot,
    #                draw_spikes=False, xlim=(0,100),
    #                blackbkgd=False,
    #                title="Fluorescence",
    #                name="new_observed_fluorescence.pdf")

    # Plot the spikes
    Ss = {n:np.where(S[n])[0] for n in n_to_plot}
    #plot_activation(F.T, Ss, [colors[c[n]] for n in xrange(N)], n_to_plot, xlim=(0,100),
    #                draw_activation=False, draw_spikes=True,
    #                blackbkgd=False,
    #                title="Spike train",
    #                name="new_observed_spikes.pdf")

    #plot_network(N, c, C, colors, blackbkgd=False)
