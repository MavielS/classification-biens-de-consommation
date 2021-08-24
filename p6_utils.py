import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from datetime import datetime
# Fichier perso
import exploration as ex

from sklearn.feature_extraction.text import CountVectorizer


plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

pd.set_option('plotting.backend', 'plotly')
pd.set_option("display.min_rows", 10)
pd.set_option("display.max_columns", 50)
pd.set_option("max_colwidth", 100)

def plot_n_gramms(data, vectorizer = CountVectorizer(), ngram_ranges: list=[(1,1)], titles=['Unigramme BoW']):
    '''
    Somme les poids associés à chacun des mots
    '''
    
    if len(ngram_ranges) != len(titles):
        print('Merci de définir autant de titre que de ranges')
        return
    
    else:
        
        fig = plt.figure(figsize=(18, len(ngram_ranges)*7))
        plt.suptitle('Somme des poids associés à chaque mot', fontsize=24)

        sns.set_style('darkgrid')
        sns.set_palette(sns.color_palette('Blues'))
        
        for ind, ngram in enumerate(ngram_ranges):
            # Computing
            vec = vectorizer(ngram_range=ngram)
            tf = vec.fit_transform(data)
            tmp = pd.DataFrame(tf.toarray(),columns=vec.get_feature_names()).sum().sort_values(ascending=False)[:10]
            
            # Displaying
            ax = fig.add_subplot(len(ngram_ranges), 1, ind+1)
            ax.set_title(titles[ind])
            sns.barplot(x=tmp.index,y=tmp.values, ax=ax, palette=sns.color_palette('deep'))

        plt.show()
        
def vectorized_ngrams(data, vec, min_words:list, ngram:tuple, debug=True):
    '''
    Construit la vectorisation des mots dans data, en suivant le vectorizer vec qui prend pour paramètre min_words
    
    @param min_words: Nombre d'occurences minimale d'un mot
    @param ngram: n-gramm à prendre en compte
    @param debug: Nous permet d'observer la taille des vecteurs créés ainsi que leur sparcity
    '''
    
    vectorized_words_values = []
    vectorizers = []
    
    for c in min_words:
        tmp = vec(min_df=c, ngram_range=ngram)

        vectorized_words_values.append(tmp.fit_transform(data))
        vectorizers.append(tmp)
        
    if debug:
        for ind in range(0, len(vectorized_words_values)):
            # Materialize the sparse data
            data_dense = vectorized_words_values[ind].todense()

            print(min_words[ind], end=' mots -- ')
             # Compute Sparsicity = Percentage of Non-Zero cells
            print(f"(ngramm:{ngram}) Taille: {vectorized_words_values[ind].shape}, Sparcity: {((data_dense > 0).sum()/data_dense.size)*100:.3f} %")

    return vectorized_words_values, vectorizers

def plot_aris_score(list_vectorized_values, model, min_words, true_labels, title):
    '''
    Nous affiche l'ARI score en fonction du min_words choisi de notre clustering
    '''
    aris = []

    for ind in range(0,len(list_vectorized_values)):
        X = list_vectorized_values[ind]
        model.fit(X)

        aris.append(adjusted_rand_score(true_labels, model.labels_))


    fig = plt.figure(figsize=(18,6))
    plt.suptitle('Metriques d\'évaluation VS Mots min. dans le BoW', fontsize=18)


    ax = fig.add_subplot(1,1,1)
    sns.lineplot(x=min_words, y=aris, color='r', marker='o')
    ax.set_title(title)
    ax.set_xlabel('Nombre d\'occurences minimale')

    plt.show()

def plot_all_aris_score(vectorized_values_list: list, true_labels, x_value, model, legends, title='Score ARI VS Occurences minimale d\'un mot'):
    '''
    Affiche l'ensemble de mes ARIs score sur le même graphique
    '''
    
    fig = go.Figure()

    for name_ind, data in enumerate(vectorized_values_list):
        aris = []
        
        # Calcule l'ARI pour une configuration donnée
        for ind in range(0,len(data)):
            X = data[ind]
            model.fit(X)
            aris.append(adjusted_rand_score(true_labels, model.labels_))
            
        # Affiche ce score sur notre figure
        fig.add_trace(go.Scatter(x=x_value, y=aris,
                    mode='lines+markers',
                    name=legends[name_ind]))
    fig.update_layout(title=title, width=2000, height=800, xaxis_title="Occurence mini.", yaxis_title="Score ARI")
    
    fig.show()