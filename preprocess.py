from __future__ import division
from __future__ import print_function


import os
import argparse
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from config import Config
from construction import features_construct_graph, spatial_construct_graph1


DEFAULT_DLPFC_DATASET = "151507"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DLPFC_RAW_ROOT = os.path.join(SCRIPT_DIR, "data", "raw", "DLPFC")
DEFAULT_DLPFC_SAVE_ROOT = os.path.join(SCRIPT_DIR, "dataset")


def normalize(adata, highly_genes=3000):
    print("start select HVGs")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    # Keep HVG-aligned raw/count-like values for ZINB target and semantic OT distributions.
    adata.layers["counts"] = adata.X.copy()
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata


def validate_counts_layer(adata):
    if "counts" not in adata.layers:
        raise ValueError('adata.layers["counts"] is missing before saving data.h5ad.')
    counts_shape = adata.layers["counts"].shape
    if counts_shape[1] != len(adata.var_names):
        raise ValueError(
            f'adata.layers["counts"] has {counts_shape[1]} genes, '
            f"but adata.var_names has {len(adata.var_names)} genes."
        )
    print(f'counts layer saved: shape={counts_shape}, dtype={adata.layers["counts"].dtype}')


def load_ST_file(path, highly_genes, k, radius):
    labels_path = os.path.join(path, "metadata.tsv")

    labels = pd.read_table(labels_path, sep='\t')
    labels = labels["layer_guess_reordered"].copy()
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()
    _, ground = np.unique(ground, return_inverse=True)

    adata1 = sc.read_visium(path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata1.var_names_make_unique()
    obs_names = np.array(adata1.obs.index)
    positions = adata1.obsm['spatial']

    data = np.delete(adata1.X.toarray(), NA_labels, axis=0)
    obs_names = np.delete(obs_names, NA_labels, axis=0)
    positions = np.delete(positions, NA_labels, axis=0)

    adata = ad.AnnData(pd.DataFrame(data, index=obs_names, columns=np.array(adata1.var.index), dtype=np.float32))

    adata.var_names_make_unique()
    adata.obs['ground_truth'] = labels
    adata.obs['ground'] = ground
    adata.obsm['spatial'] = positions
    adata.obs['array_row'] = adata1.obs['array_row']
    adata.obs['array_col'] = adata1.obs['array_col']
    adata.uns['spatial'] = adata1.uns['spatial']
    adata.var['gene_ids'] = adata1.var['gene_ids']
    adata.var['feature_types'] = adata1.var['feature_types']
    adata.var['genome'] = adata1.var['genome']
    adata.var_names_make_unique()

    adata = normalize(adata, highly_genes=highly_genes)
    fadj = features_construct_graph(adata.X, k=k)
    sadj, graph_nei, graph_neg = spatial_construct_graph1(adata, radius=radius)

    adata.obsm["fadj"] = fadj
    adata.obsm["sadj"] = sadj
    adata.obsm["graph_nei"] = graph_nei.numpy()
    adata.obsm["graph_neg"] = graph_neg.numpy()
    adata.var_names_make_unique()
    return adata


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--raw_root", type=str, default=DEFAULT_DLPFC_RAW_ROOT)
    parse.add_argument("--save_root", type=str, default=DEFAULT_DLPFC_SAVE_ROOT)
    args = parse.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset = DEFAULT_DLPFC_DATASET
    print(dataset)

    raw_data_dir = os.path.join(args.raw_root, dataset)
    savepath = args.save_root
    config_file = os.path.join(SCRIPT_DIR, 'config', 'DLPFC.ini')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    config = Config(config_file)
    adata = load_ST_file(raw_data_dir, config.fdim, config.k, config.radius)
    print("k=", config.k)
    print("saving")
    validate_counts_layer(adata)
    adata.write(os.path.join(savepath, '{}.h5ad'.format(dataset)))
    print("done")
