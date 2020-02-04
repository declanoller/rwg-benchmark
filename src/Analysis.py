import os, json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def analyze_solution_weights(dir, cutoff_score):

    with open(os.path.join(dir, 'evo_stats.json'), 'r') as f:
        evo_dict = json.load(f)


    all_weights = evo_dict['all_weights']
    all_scores = evo_dict['all_scores']
    print(f'Dimensionality of weight space: {len(all_weights[0])}')

    print(f'{len(all_weights)} total weights')
    all_weights = np.array([w for i,w in enumerate(all_weights) if all_scores[i]>=cutoff_score])
    #all_weights = np.array(all_weights)
    print(f'{len(all_weights)} weights with mean score above cutoff of {cutoff_score}')

    print('Fitting for varying number of clusters...')
    # This assumes the score starts small and goes up.
    clusters = list(range(1,25))
    kmeans_list = [KMeans(n_clusters=i, random_state=0) for i in clusters]
    kmeans_scores = [k.fit(all_weights).score(all_weights) for k in kmeans_list]

    ptp = np.ptp(kmeans_scores)
    mu = np.mean(kmeans_scores)
    print(f'Total score diff of {ptp}')

    target_perc = 0.85
    target_elbow_score = kmeans_scores[0] + target_perc*ptp

    print(f'target elbow score is {target_elbow_score}')

    score_distances = (np.array(kmeans_scores) - target_elbow_score)**2/ptp**2
    #print(f'Score distances: {score_distances}')

    best_guess_clusters = np.argmin(score_distances) + 1
    print(f'Best guess at clusters is {best_guess_clusters} clusters')
    plt.gca().axvline(best_guess_clusters, linestyle='dashed', alpha=0.5, color='black')

    plt.plot(clusters, (np.array(kmeans_scores) - mu)/ptp)
    plt.plot(clusters, score_distances, 'o', color='black')

    plt.xlabel('N clusters')
    plt.ylabel('Objective score')
    plt.tight_layout()
    plt.savefig(f'K_means_elbow_cutoffscore_{cutoff_score}.png')
    plt.show()
    N_clusters = best_guess_clusters
    #N_clusters = 4
    kmeans = KMeans(n_clusters=N_clusters, random_state=0).fit(all_weights)


    tsne = TSNE(n_components=2)

    #tsne.fit(gnr)
    #print(tsne.embedding_)

    tsne_res = tsne.fit_transform(all_weights)

    collected_tsne = [np.array([r for i,r in enumerate(tsne_res) if kmeans.labels_[i]==c]) for c in range(N_clusters)]

    cols = [
            'dodgerblue',
            'tomato',
            'forestgreen',
            'darkorange',
            'plum',
            'gray',
            'pink',
            'brown',
            'red',
            'yellow',
            'seagreen',
            'cyan',
            'violet',
            'lawngreen',
            'gold',
            'magenta',
            'blue'
            ]
    for i,cluster_tsne in enumerate(collected_tsne):
        plt.scatter(*cluster_tsne.transpose(),
        alpha=0.3, color=cols[i])

    #print(tsne_res)
    '''for i,r in enumerate(tsne_res):
        plt.scatter(*r, alpha=0.3, color=cols[kmeans.labels_[i]])'''

    '''for i in range(len(clouds_list)):
        plt.scatter(*tsne_res[i*N:(i+1)*N].transpose(), alpha=0.3, color=cols[i])'''
    plt.savefig(f'tSNE_cartpole_{N_clusters}_clusters_cutoffscore_{cutoff_score}.png')
    plt.show()
