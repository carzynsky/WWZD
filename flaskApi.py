import umap
import json
from re import T
import jsonlines
import pandas as pd
import numpy as np
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
    pca.fit(A)
    print(pca.explained_variance_ratio_)
    global A_2
    A_2 = pca.transform(A)
    print(A_2.shape)
    fig = px.scatter(A_2, x=0, y=1,title='PCA')
    fig.show()

def startUmap():
    print('Starting umap...')
    global B
    B = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(A)
    B = B.tolist()
    print(type(B))

def tsne():
    # Reduce dimensionality to 2 with t-SNE.
    # Perplexity is roughly the number of close neighbors you expect a
    # point to have.

    # n-iter: Maximum number of iterations for the optimization. 
    # Should be at least 250.
    arr = np.array(A)
    print('t-SNE fit and transform started')
    tsne = TSNE(n_components = 2, perplexity=10, init='random', learning_rate='auto', n_iter=1500).fit_transform(arr)
    global T
    T = tsne.tolist()

def preload():
    global A
    global Ids
    global Rp
    A = []
    Ids = []
    print('Loading herbert data')
    with jsonlines.open(herbertFilePath) as f:
        for line in f.iter():
            A.append(line['features'])
            Ids.append(line['id'])

    Rp = []
    print('Loading rp data...')
    with jsonlines.open(rpFilePath) as f:
        for line in f.iter():
            Rp.append(line)

if __name__ == '__main__':
    preload()
    
    pca()
    #startUmap()
    #tsne()
    
    api.run() 