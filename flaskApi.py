import json
from numpy.core.fromnumeric import shape, transpose
import pandas as pd
import numpy as np
from flask import Flask, json
from sklearn.decomposition import PCA
import umap
import statistics

# file paths
filePathRpHerbertKgr10Data = 'data/rp_herbert-kgr10.json'
filePathRp = 'data/rp.json'

api = Flask(__name__)

@api.route('/rp', methods=['GET'])
def get_rp():
    jsonDf = json_df.to_json(force_ascii=False)
    return json.dumps(
    {
        "data": jsonDf
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

@api.route('/hello', methods=['GET'])
def get_hello():
    return json.dumps(
        {
            "msg": "test response"
        })

def pca():
    # open and load file
    print('Set pca components')
    pca = PCA(n_components=128)
    print('Started fitting...')
    pca.fit(A)
    print(pca.explained_variance_ratio_)
    global A_2
    A_2 = pca.transform(A)
    print(A_2.shape)

# def umap():
#     global B
#     B = umap.UMAP(n_neighbors=5,
#                       min_dist=0.3,
#                       metric='correlation').fit_transform(A)
#     print(type(B))

def preload():
    print('Load rpHerbertKgr10 data')
    f = open(filePathRpHerbertKgr10Data)
    rpHerbertKgr10Data_df = json.load(f)
    global A
    A = rpHerbertKgr10Data_df['data']
    global json_df
    json_df = pd.read_json(filePathRp, encoding='utf-8')

if __name__ == '__main__':
    preload()
    pca()
    #umap()
    api.run() 