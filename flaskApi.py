import umap
import umap.plot
import json
from re import T
import jsonlines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, json, request
from flask_cors import CORS
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.manifold import TSNE
import random 

# file paths
rpFilePath = 'newData/rp.jsonl'
herbertFilePath = 'newData/herbert.jsonl'
configPath = 'config.json'

api = Flask(__name__)
cors = CORS(api)

@api.route('/ids', methods=['GET'])
def get_ids():
    return json.dumps({
            "ids": Ids
        })

@api.route('/rps', methods=['GET'])
def get_rps():
    quer_params = request.args
    _Rp = Rp
    if(len(quer_params) > 0):
        ids = quer_params['id'].replace(' ', '').split(',')
        _Rp = []
        for id in ids:
            _Rp.append(list(filter(lambda x: x['id'] == id, Rp))[0])
        
    return json.dumps({
        "data": list(_Rp)
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
    data = request.args['data']
    return json.dumps({
            "series": A_pca[data]
        })

@api.route('/umap', methods=['GET'])
def get_results_of_umap():
    data = request.args['data']
    return json.dumps({
            "series": B[data]
        })
@api.route('/tsne', methods=['GET'])
def get_results_of_tsne():
    data = request.args['data']
    return json.dumps({
        'series': T[data]
    })

def pca():
    print('[PCA] fit and transform')
    global A_pca
    A_pca = {
        'herbert': PCA(n_components=2).fit_transform(bertData).tolist(),
        'dKleczekBert': PCA(n_components=2).fit_transform(dKleczekBertData).tolist(),
        'stDistiluse': PCA(n_components=2).fit_transform(stDistiluseData).tolist(),
        'tfidf': PCA(n_components=2).fit_transform(tfidfData).tolist()
    }

    print('[PCA] format data')
    for key, value in A_pca.items():
        x  = [[round(cords[0], 3), round(cords[1], 3)] for cords in value]
        x = prepareDtoData(x)
        A_pca[key] = x

    if(draw == False):
        return

    # fig = px.scatter(A_pca, x=0, y=1,title='PCA', color=Labels)
    # fig.write_image('plots/pca.png')
    # fig.show()

def prepareDtoData(dataList):
    tmp = []
    for i in range(len(dataList)):
        tmp.append([Labels[i], {'name': Ids[i], 'x': dataList[i][0], 'y': dataList[i][1]}])

    list_dto = []
    for label in UniqueLabels:
        series = filter(lambda x: x[0] == label, tmp)
        series = list(series)
        seriesWithoutLabel = []
        for s in series:
            seriesWithoutLabel.append(s[1])
 
        list_dto.append({
            'data': seriesWithoutLabel,
            'name': label,
            'color': f'rgba({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)})'
        })
    return list_dto

def startUmap(n_neighbors=5, min_dist=0.3, metric='cosine'):
    print(f'[UMAP] Parameters (n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})')
    global B
    print('[UMAP] Fit and transform')
    B = {
        'herbert': umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric).fit_transform(bertData).tolist(),
        'dKleczekBert': umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric).fit_transform(dKleczekBertData).tolist(),
        'stDistiluse': umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric).fit_transform(stDistiluseData).tolist(),
        'tfidf': umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric).fit_transform(tfidfData).tolist()
    }

    print('[UMAP] format data')
    for key, value in B.items():
        x  = [[round(cords[0], 3), round(cords[1], 3)] for cords in value]
        x = prepareDtoData(x)
        B[key] = x

    if(draw == False):
        return

    #draw_embedding(embedding, Labels,  show=True, name=f"umap_{n_neighbors}-{min_dist}-{metric}")

def draw_embedding(data, labels, show=True, name="plot"):
    # p = umap.plot.points(data, labels=np.array(labels))
    # umap.plot.plt.legend(bbox_to_anchor=(1, 1.05), loc=2)
    # umap.plot.plt.savefig(f"./plots/{name}.png")
    # if show:
        # fig.show()
        # hover_data = pd.DataFrame({'index':np.arange(len(labels)),
        #                    'label':np.array(labels)})
        # umap.plot.interactive(data, labels=np.array(labels), hover_data=hover_data)
        # umap.plot.plt.show()
    fig = px.scatter(data, x=0, y=1,title='UMAP', color=labels)
    fig.write_image(f'plots/{name}.png')
    if show:
        fig.show()


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

    print('[TSNE] Fit and transform')
    global T
    T = {
        'herbert': TSNE(n_components = 2, perplexity=5, init='random', learning_rate='auto', n_iter=1000).fit_transform(np.array(bertData)).tolist(),
        'dKleczekBert': TSNE(n_components = 2, perplexity=5, init='random', learning_rate='auto', n_iter=1000).fit_transform(np.array(dKleczekBertData)).tolist(),
        'stDistiluse': TSNE(n_components = 2, perplexity=5, init='random', learning_rate='auto', n_iter=1000).fit_transform(np.array(stDistiluseData)).tolist(),
        'tfidf': TSNE(n_components = 2, perplexity=5, init='random', learning_rate='auto', n_iter=1000).fit_transform(np.array(tfidfData)).tolist()
    }

    for key, value in T.items():
        x = [[round(cords[0], 3), round(cords[1], 3)] for cords in value]
        x = prepareDtoData(x)
        T[key] = x

    if(draw == False):
        return

    # fig = px.scatter(tsne, x=0, y=1,title='t-SNE', color=Labels)
    # fig.write_image('plots/tSNE.png')
    # fig.show()

def preload():
    global bertData
    global tfidfData
    global dKleczekBertData
    global stDistiluseData
    global Ids
    global Rp
    global Labels
    global UniqueLabels

    bertData = []
    Ids = []
    print('[PRELOAD] Loading herbert data')
    with jsonlines.open(herbertFilePath) as f:
        for line in f.iter():
            bertData.append(line['features'])
            Ids.append(line['id'])

    tfidfData = []
    print('[PRELOAD] Loading tfidf data')
    with jsonlines.open('./newData/tfidf.jsonl') as file:
        for line in file.iter():
            tfidfData.append(line['features'])
    
    dKleczekBertData = []
    print('[PRELOAD] Loading dkleczek_bert data')
    with jsonlines.open('./newData/dkleczek_bert.jsonl') as file:
        for line in file.iter():
            dKleczekBertData.append(line['features'])

    stDistiluseData = []
    print('[PRELOAD] Loading st_distiluse data')
    with jsonlines.open('./newData/st_distiluse.jsonl') as file:
        for line in file.iter():
            stDistiluseData.append(line['features'])

    Rp = []
    Labels = []
    print('[PRELOAD] Loading rp data')
    with jsonlines.open(rpFilePath) as f:
        for line in f.iter():
            Rp.append(line)
            Labels.append(line['label'])
    
    UniqueLabels = []
    for label in Labels:
        if(label in UniqueLabels):
            continue
        UniqueLabels.append(label)

def readConfig():
    global draw
    draw = False
    with open(configPath, 'r') as file:
        data = json.load(file)
        draw = data['drawPlots']
        print(draw)

if __name__ == '__main__':
    readConfig()
    preload()
    pca()
    startUmap(metric='cosine', n_neighbors=4, min_dist=0.0)
    tsne()
    
    api.run() 