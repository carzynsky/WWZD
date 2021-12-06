import umap
import umap.plot
import json
from re import T
import jsonlines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, json
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.manifold import TSNE

# file paths
filePathRpHerbertKgr10Data = 'data/rp_herbert-kgr10.json'
rpFilePath = 'newData/rp.jsonl'
herbertFilePath = 'newData/herbert.jsonl'

api = Flask(__name__)

@api.route('/ids', methods=['GET'])
def get_ids():
    return json.dumps({
            "ids": Ids
        })

@api.route('/rp', methods=['GET'])
def get_rp():
    return json.dumps({
        "data": Rp
    })

@api.route('/rp/<rp_id>', methods=['GET'])
def get_rp_by_id(rp_id):
    rpId = rp_id
    print(rpId)
    metaData = filter(lambda x: x['id'] == rpId, Rp)
    return json.dumps({
        "data": list(metaData)[0]
    })

@api.route('/pca', methods=['GET'])
def get_results_of_pca():
    list = A_2.tolist()
    return json.dumps({
            "data": list
        })

@api.route('/umap', methods=['GET'])
def get_results_of_umap():
    return json.dumps({
            "data": B
        })
@api.route('/tsne', methods=['GET'])
def get_results_of_tsne():
    return json.dumps({
        'data': T
    })

def pca():
    # open and load file
    print('Set pca components')
    pca = PCA(n_components=2)
    print('Started fitting...')
    pca.fit(bertData)
    print(pca.explained_variance_ratio_)
    global A_2
    A_2 = pca.transform(bertData)
    print(A_2.shape)
    fig = px.scatter(A_2, x=0, y=1,title='PCA')
    fig.show()

def startUmap(n_neighbors=5, min_dist=0.3, metric='cosine'):
    print(f'Starting umap (n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})...')
    global B
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                      min_dist=min_dist,
                      metric=metric)
    fit = reducer.fit(bertData)
    draw_embedding(fit, Labels,  show=True, name=f"umap_{n_neighbors}-{min_dist}-{metric}")

    embedding = reducer.transform(bertData)
    B = embedding.tolist()
    print(type(B))

def draw_embedding(fit, labels, show=True, name="plot"):
    p = umap.plot.points(fit, labels=np.array(labels))
    if show:
        umap.plot.plt.legend(bbox_to_anchor=(1.05, 1), loc=2, mode="expand")
        umap.plot.plt.show()
    umap.plot.plt.legend(bbox_to_anchor=(1.05, 1), loc=2, mode="expand")
    umap.plot.plt.savefig(f"./plots/{name}.png")

def umapNeighboursRange(metric='cosine', min_dist=0.0):
    print('Starting umap for various n_neighbours...')
    for n in (2, 5, 10, 20, 50, 100, 200):
        startUmap(n_neighbors=n, min_dist=min_dist, metric=metric)

def umapMetricsRange(n_neighbors=5, min_dist=0.3):
    print('Starting umap for various metrics...')
    for m in ('euclidean', 'correlation', 'cosine', 'chebyshev'):
        startUmap(metric=m,n_neighbors=n_neighbors, min_dist=min_dist)

def umapDistRange(n_neighbors=5 ,metric='cosine'):
    print('Starting umap for various min_dist...')
    for d in (0.0, 0.1, 0.25, 0.5, 0.8, 0.99):
        startUmap(min_dist=d ,n_neighbors=n_neighbors, metric='metric')

def tsne():
    # Reduce dimensionality to 2 with t-SNE.
    # Perplexity is roughly the number of close neighbors you expect a
    # point to have.

    # n-iter: Maximum number of iterations for the optimization. 
    # Should be at least 250.
    arr = np.array(bertData)
    print('t-SNE fit and transform started')
    tsne = TSNE(n_components = 2, perplexity=10, init='random', learning_rate='auto', n_iter=1500).fit_transform(arr)
    global T
    T = tsne.tolist()

def preload():
    global bertData
    global Ids
    global Rp
    global Labels
    bertData = []
    Ids = []
    print('Loading herbert data')
    with jsonlines.open(herbertFilePath) as f:
        for line in f.iter():
            bertData.append(line['features'])
            Ids.append(line['id'])

    Rp = []
    Labels = []
    print('Loading rp data...')
    with jsonlines.open(rpFilePath) as f:
        for line in f.iter():
            Rp.append(line)
            Labels.append(line['label'])

if __name__ == '__main__':
    preload()
    
    # pca()
    startUmap(metric='cosine', n_neighbors=5, min_dist=0.0)
    # tsne()
    
    api.run() 