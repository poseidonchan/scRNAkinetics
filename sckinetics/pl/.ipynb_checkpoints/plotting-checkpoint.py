import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def gene_trends(adata, gene, hue='clusters', pt_key='palantir_pseudotime', ukey='Mu', skey='Ms'):
    """
    plotting gene trends based on cellrank GAM regression model, if not, the raw data will be displayed.
    :param adata:
    :return:
    """
    for i in range(adata.shape[1]):
        if adata.var.index[i] == gene:
            break

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.scatterplot(x=adata.obs[pt_key],
                    y=np.array(adata.layers[ukey][:, i]).reshape(-1),
                    hue=adata.obs[hue], marker='.', edgecolor='none', ax=axes[0])
    axes[0].set_ylabel('unspliced')
    axes[0].set_xlabel(pt_key)
    axes[0].get_legend().remove()

    # Plot 3
    sns.scatterplot(x=adata.obs[pt_key],
                    y=np.array(adata.layers[skey][:, i]).reshape(-1),
                    hue=adata.obs[hue], marker='.', edgecolor='none', ax=axes[1])
    axes[1].set_ylabel('spliced')
    axes[1].set_xlabel(pt_key)
    axes[1].get_legend().remove()

    # Remove top and right spines
    for j in range(2):
        axes[j].spines['top'].set_visible(False)
        axes[j].spines['right'].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(gene + ' gene expression')
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.1), ncol=4)

    plt.show()


def phase_plot(adata, gene=None, hue='clusters', ukey='Mu', skey='Ms'):
    """
    plot gene expression phase plot. x-axis is spliced, y-axis is unspliced.

    :param adata:
    :return:
    """
    for i in range(adata.shape[1]):
        if adata.var.index[i] == gene:
            break

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.scatterplot(x=np.array(adata.layers[skey][:, i]).reshape(-1),
                    y=np.array(adata.layers[ukey][:, i]).reshape(-1),
                    hue=adata.obs[hue], marker='.', edgecolor='none', ax=ax)
    ax.set_ylabel('unspliced')
    ax.set_xlabel('spliced')
    ax.get_legend().remove()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    plt.show()

def kinetics_trends(adata, gene, hue='clusters', pt_key='palantir_pseudotime',
                    kinetics_key={'alpha':'alpha', 'beta':'beta', 'gamma':'gamma'}):
    """
    plotting kinetics trends based on cellrank GAM regression model, if not, the raw data will be displayed.
    :param adata:
    :return:
    """
    for i in range(adata.shape[1]):
        if adata.var.index[i] == gene:
            break

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))


    sns.scatterplot(x=adata.obs[pt_key],
                    y=np.array(adata.layers[kinetics_key['alpha']][:, i]).reshape(-1),
                    hue=adata.obs[hue], marker='.', edgecolor='none', ax=axes[0])
    axes[0].set_ylabel('alpha')
    axes[0].set_xlabel(pt_key)
    axes[0].get_legend().remove()

    # Plot 3
    sns.scatterplot(x=adata.obs[pt_key],
                    y=np.array(adata.layers[kinetics_key['beta']][:, i]).reshape(-1),
                    hue=adata.obs[hue], marker='.', edgecolor='none', ax=axes[1])
    axes[1].set_ylabel('beta')
    axes[1].set_xlabel(pt_key)
    axes[1].get_legend().remove()

    sns.scatterplot(x=adata.obs[pt_key],
                    y=np.array(adata.layers[kinetics_key['gamma']][:, i]).reshape(-1),
                    hue=adata.obs[hue], marker='.', edgecolor='none', ax=axes[2])
    axes[2].set_ylabel('gamma')
    axes[2].set_xlabel(pt_key)
    axes[2].get_legend().remove()

    # Remove top and right spines
    for j in range(3):
        axes[j].spines['top'].set_visible(False)
        axes[j].spines['right'].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    
    fig.suptitle(gene + ' gene expression')
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.1), ncol=4)

    plt.show()


def phase_plot_velocity(adata, gene=None, hue=None,
                        vkey={'unspliced_velocity': "unspliced_velocity",
                              'spliced_velocity': "spliced_velocity"},
                        ukey='Mu', skey='Ms',
                        arrow_length=1e-3,
                        width=.0015,
                        dpi=120,):
    """
    plot velocity in the phase plot.
    :param adata:
    :return:
    """
    for i in range(adata.shape[1]):
        if adata.var.index[i] == gene:
            break
    x = np.array(adata.layers[skey][:, i]).reshape(-1, 1)
    y = np.array(adata.layers[ukey][:, i]).reshape(-1, 1)

    print(i)

    x_v = np.array(adata.layers[vkey['spliced_velocity']][:, i]).reshape(-1, 1)
    y_v = np.array(adata.layers[vkey['unspliced_velocity']][:, i]).reshape(-1, 1)

    X = np.concatenate((x, y), axis=1)
    V = np.concatenate((x_v, y_v), axis=1)

    adata_new = sc.AnnData(X=X, obs=adata.obs)
    adata_new.layers['Ms'] = X
    adata_new.obsm['X_' + gene] = X
    adata_new.layers[gene + '_velocity'] = V

    scv.pl.velocity_embedding(adata_new, basis=gene, X=X, V=V, c=hue, vkey=gene + '_velocity',
                              scale=scale, width=width,arrow_length=arrow_length,
                              dpi=dpi)

