import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
import torch
from sklearn.neighbors import kneighbors_graph


def spatial_construct_graph1(adata, radius=150):
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    A = np.zeros((coor.shape[0], coor.shape[0]))

    # print("coor:", coor)
    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)

    for it in range(indices.shape[0]):
        A[[it] * indices[it].shape[0], indices[it]] = 1

    print('The graph contains %d edges, %d cells.' % (sum(sum(A)), adata.n_obs))
    print('%.4f neighbors per cell on average.' % (sum(sum(A)) / adata.n_obs))

    graph_nei = torch.from_numpy(A)

    graph_neg = torch.ones(coor.shape[0], coor.shape[0]) - graph_nei

    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    return sadj, graph_nei, graph_neg


def features_construct_graph(features, k=15, pca=None, mode="connectivity", metric="cosine"):
    print("start features construct graph")
    if pca is not None:
        features = dopca(features, dim=pca).reshape(-1, 1)
    # print("k: ", k)
    # print("features_construct_graph features", features.shape)
    A = kneighbors_graph(features, k + 1, mode=mode, metric=metric, include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    # index = np.argwhere(A > 0)
    # np.savetxt('./result/fadj.csv', index, delimiter=',')
    fadj = sp.coo_matrix(A, dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    # nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    # nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    return fadj  # , nfadj
