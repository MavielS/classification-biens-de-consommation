import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.base import clone
from timeit import default_timer as timer

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def sampling_with_ARI(model, df, size_samples, n_iter, stratify=None, debug=True):
    
    model_samp = clone(model)
    
    # Entrainement du modèle sur le dataset complet
    model_pop = model.fit(df)
    
    results = pd.DataFrame(columns=['taille_sample',  'mean_ari', 'std_ari', 'predict_time'])
    
    for sz_sample in size_samples:
        if debug: print(f'Taille du sample {sz_sample}: iteration ', end='')
            
        # J'entraine n_iter fois un modèle avec un échantillon de ma population aléatoire d'une taille sz_sample
        ari_sample = []
        predict_time = 0
        for i in range(n_iter):
            if debug: print(f'{i+1}....', end='')
            df_samp, _ = train_test_split(df, train_size=sz_sample, stratify=stratify) # On ne fixe pas random_state donc le choix va être aléatoire et différent à chaque itération
            
            start = timer()
            model_samp.fit(df_samp)
            
            predict_time += timer() - start
            true_labels = model_pop.predict(df_samp)
            ari = adjusted_rand_score(true_labels, model_samp.labels_)            
            ari_sample.append(ari)
        
        results = results.append(pd.DataFrame({'taille_sample':[sz_sample], 'mean_ari':[np.mean(ari_sample)], 'std_ari':[np.std(ari_sample)], 'predict_time':round(predict_time/n_iter, 3)}))
        if debug: print(f'Temps: {predict_time}')
    return results

def compute_clust_scores_nclust(model, df, n_iter=10,
                                num_clusters=range(2,8), return_pop=False):
#### ATTENTION AU CAS PAR CAS POUR LES MODELES AUTRES QUE KMEANS

    dict_pop_perc_n_clust = {}
    dict_scores_iter = {}
    
    # --- Looping on the number of clusters to compute the scores
    for n_clust in num_clusters:

        silh, dav_bould, cal_harab, distor = [], [], [], []
        pop_perc_iter = pd.DataFrame()

        # Iterations of the same model (stability)
        for j in range(n_iter):
            model = KMeans(n_clusters=n_clust)
            model.fit(df)
            
            ser_clust = pd.Series(data=model.predict(df),
                                  index=df.index,
                                  name="iter_"+str(j))
            if return_pop:
                # Compute pct of clients in each cluster
                pop_perc = 100 * ser_clust.value_counts() / df.shape[0]
                pop_perc.sort_index(inplace=True)
                pop_perc.index = ['clust_'+str(i) for i in pop_perc.index]
                pop_perc_iter = pd.concat([pop_perc_iter, pop_perc.to_frame()],
                                          axis=1)
        
            # Computing scores for iterations
            silh.append(silhouette_score(X=df, labels=ser_clust))
            dav_bould.append(davies_bouldin_score(X=df, labels=ser_clust))
            cal_harab.append(calinski_harabasz_score(X=df, labels=ser_clust))
            distor.append(model.inertia_)

        if return_pop:
            # dict of the population (pct) of clusters iterations
             dict_pop_perc_n_clust[n_clust] = pop_perc_iter.T

        # Dataframe of the results on iterations
        scores_iter = pd.DataFrame({'Silhouette': silh,
                                    'Calinsky_Harabasz': cal_harab,
                                    'Davies_Bouldin': dav_bould,
                                    'Distortion': distor,
                                    })
        dict_scores_iter[n_clust] = scores_iter

    if return_pop:
        return dict_scores_iter, dict_pop_perc_n_clust
    else:
        return dict_scores_iter

    
def plot_scores_vs_n_clust(dict_scores_iter, figsize=(15,3)):
    ''' Plot the 4 mean scores stored in the dictionnary returned by the function
compute_clust_scores_nclust (dictionnary of dataframes of scores (columns)
for each iteration (rows) of the model and for each number of clusters
in a figure with error bars (2 sigmas)'''

    fig = plt.figure(figsize=figsize)
    list_n_clust = list(dict_scores_iter.keys())

    # Generic fonction to unpack dictionary and plot one graph
    def score_plot_vs_nb_clust(scores_iter, name, ax, c=None):
        score_mean = [dict_scores_iter[i].mean().loc[n_score] for i in list_n_clust]
        score_std = np.array([dict_scores_iter[i].std().loc[n_score]\
                            for i in list_n_clust])
        ax.errorbar(list_n_clust, score_mean, yerr=2*score_std, elinewidth=1,
                capsize=2, capthick=1, ecolor='k', fmt='-o', c=c, ms=5,
                barsabove=False, uplims=False)

    li_scores = ['Silhouette', 'Calinsky_Harabasz',
                   'Davies_Bouldin', 'Distortion']
    li_colors = ['r', 'b', 'purple', 'g']

    # Looping on the 4 scores
    i=0
    for n_score, c in zip(li_scores, li_colors):
        i+=1
        ax = fig.add_subplot(1,4,i)
        
        score_plot_vs_nb_clust(dict_scores_iter, n_score, ax, c=c)
        ax.set_xlabel('Number of clusters')
        ax.set_title(n_score+' score')
        ax.grid()

    fig.suptitle('Clustering score vs. number of clusters',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()