import json
import jsonlines
from numpy.core.fromnumeric import shape, transpose
import pandas as pd
import numpy as np
from flask import Flask, json
from sklearn.decomposition import PCA
import umap
import statistics

# file paths
filePathRpHerbertKgr10Data = 'data/rp_herbert-kgr10.json'
rpFilePath = 'newData/rp.jsonl'
herbertFilePath = 'newData/herbert.jsonl'

api = Flask(__name__)

@api.route('/ids', methods=['GET'])
def get_ids():
    return json.dumps(
        {
            "ids": Ids
        })

@api.route('/rp', methods=['GET'])
def get_rp():
    return json.dumps(
    {
        "data": Rp
    })

@api.route('/rp/<rp_id>', methods=['GET'])
def get_rp_by_id(rp_id):
    rpId = rp_id
    print(rpId)
    metaData = filter(lambda x: x['id'] == rpId, Rp)
    return json.dumps(
    {
        "data": list(metaData)[0]
    })

@api.route('/pca', methods=['GET'])
def get_results_of_pca():
    list = A_2.tolist()
    return json.dumps(
        {
            "data": list
        })

@api.route('/umap', methods=['GET'])
def get_results_of_umap():
    return json.dumps(
        {
            "data": "none"
        })

def pca():
    # open and load file
    print('Set pca components')
    pca = PCA(n_components=2)
    print('Started fitting...')
    pca.fit(A)
    # print(pca.explained_variance_ratio_)
    global A_2
    A_2 = pca.transform(A)
    # print(A_2.shape)

def umap():
    global B
    B = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(A)
    print(type(B))

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

    print(len(Rp))
if __name__ == '__main__':
    preload()
    pca()
    #umap()
    api.run() 

   # pca raczej nie
   # umap zaimportowac, 
   # tsne dlugooo
   # lda