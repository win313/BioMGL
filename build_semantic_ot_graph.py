from __future__ import print_function

import argparse
import os
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from tqdm import tqdm


EPS = 1e-12
NOT_AVAILABLE = "Not available"
DLPFC_IDS = {"151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672"}

GENE_PROMPT_SYSTEM = (
    "System:\n"
    "You are an expert in molecular biology and bioinformatics. Encode the intrinsic "
    "biological semantics of a single gene based on the provided gene symbol and NCBI Gene "
    "summary. Focus on molecular functions, biological processes, pathway involvement, "
    "regulatory roles, and other relevant functional characteristics. Base the encoding on "
    "the provided information and avoid unsupported speculation."
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_ID = os.environ.get("SEMST_DATASET_ID", "151507")
DEFAULT_DATA_ROOT = os.environ.get("SEMST_DATA_ROOT", os.path.join(PROJECT_ROOT, "dataset"))
DEFAULT_ANNOTATION_INPUT_ROOT = os.environ.get("SEMST_ANNOTATION_INPUT_ROOT", os.path.join(PROJECT_ROOT, "gene-annotation"))
DEFAULT_SEMADJ_ROOT = os.environ.get("SEMST_SEMADJ_ROOT", os.path.join(PROJECT_ROOT, "gene_embeddings"))
QWEN3_GGUF_FILENAME = "Qwen3-30B-A3B-Thinking-2507-UD-Q4_K_XL.gguf"
DEFAULT_QWEN3_GGUF_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    QWEN3_GGUF_FILENAME,
)
DEFAULT_ADATA_PATH = os.environ.get(
    "SEMST_ADATA_PATH",
    os.path.join(PROJECT_ROOT, "dataset", "151507.h5ad"),
)
DEFAULT_ANNOTATION_TEMPLATE_PATH = os.environ.get(
    "SEMST_ANNOTATION_TEMPLATE_PATH",
    os.path.join(PROJECT_ROOT, "gene-annotation", "151507-ncbi.csv"),
)
DEFAULT_GENE_EMBEDDING_PATH = os.environ.get(
    "SEMST_GENE_EMBEDDING_PATH",
    os.path.join(PROJECT_ROOT, "gene_embeddings", "151507_gene_embeddings.npy"),
)
DEFAULT_EMBEDDING_MODEL = os.environ.get("SEMST_LLM_MODEL_PATH", DEFAULT_QWEN3_GGUF_MODEL_PATH)
DEFAULT_EMBEDDING_DEVICE = "cuda"
DEFAULT_EMBEDDING_BATCH_SIZE = 1
DEFAULT_EMBEDDING_MAX_LENGTH = 1536
DEFAULT_SAVE_GENE_EMBEDDING_PATH = os.environ.get(
    "SEMST_SAVE_GENE_EMBEDDING_PATH",
    os.path.join(PROJECT_ROOT, "gene_embeddings", "151507_gene_embeddings.npy"),
)
DEFAULT_OT_EXPRESSION_LAYER = "X"
DEFAULT_CANDIDATE_EXPRESSION_LAYER = "X"
DEFAULT_CANDIDATE_USE_PCA = False#做候选点之前是否需要对表达矩阵做PCA降维，默认不需要
DEFAULT_CANDIDATE_PCA_DIM = 50#候选点搜索时 PCA 降维后的维度
DEFAULT_TOP_L = 50#选取高表达基因的个数
DEFAULT_CANDIDATE_K = 500#候选点的个数
DEFAULT_DISABLE_CANDIDATE_SELECTION = True#计算候选点的开关，False：使用候选点  True：关闭候选点
DEFAULT_CANDIDATE_METRIC = "cosine"
DEFAULT_SEM_K = 10
DEFAULT_SINKHORN_REG = 0.05
DEFAULT_SINKHORN_ITERS = 100
DEFAULT_SINKHORN_TOL = 1e-6
DEFAULT_SEM_ADJ_KEY = "semadj"
DEFAULT_OUTPUT_SEMADJ_PATH = os.path.join(
    PROJECT_ROOT,
    "gene_embeddings",
    "151507_biological_semantic_spot_adjacency_k10.npz",
)
DEFAULT_OUTPUT_ADATA_PATH = None
DEFAULT_SAVE_GENE_COST_PATH = None


def resolve_dataset_group(dataset_id):
    normalized = str(dataset_id).strip()
    lowered = normalized.lower()
    if normalized in DLPFC_IDS or lowered == "dlpfc":
        return "DLPFC"
    if lowered in {"e1s1", "embryo"}:
        return "Embryo"
    if lowered in {"hbc", "human_breast_cancer"}:
        return "Human_Breast_Cancer"
    if lowered in {"mba", "mouse_brain_anterior"}:
        return "Mouse_Brain_Anterior"
    if lowered == "mvc":
        return "MVC"
    return normalized


def safe_path_component(value):
    value = str(value).strip()
    value = re.sub(r'[<>:"/\\|?*\s]+', "_", value)
    value = value.strip("._")
    if not value:
        raise ValueError("dataset_id cannot be empty when resolving paths.")
    return value


def dataset_path_parts(dataset_id):
    dataset_id = str(dataset_id).strip()
    dataset_group = resolve_dataset_group(dataset_id)
    safe_group = safe_path_component(dataset_group)
    safe_dataset_id = safe_path_component(dataset_id)
    if safe_group == safe_dataset_id:
        return [safe_group]
    return [safe_group, safe_dataset_id]


def resolve_adata_path(dataset_id, data_root):
    return os.path.join(data_root, *dataset_path_parts(dataset_id), "data.h5ad")

#基因的名称、基因的NCBI文件地址
def resolve_annotation_template_path(dataset_id, annotation_input_root):
    return os.path.join(annotation_input_root, *dataset_path_parts(dataset_id), "annotation_template_ncbi_filled.csv")


def resolve_gene_embedding_path(dataset_id, semadj_root):
    dataset_group = safe_path_component(resolve_dataset_group(dataset_id))
    safe_dataset_id = safe_path_component(dataset_id)
    return os.path.join(semadj_root, dataset_group, f"{safe_dataset_id}_gene_embeddings.npy")


def resolve_semadj_path(dataset_id, semadj_root):
    dataset_group = safe_path_component(resolve_dataset_group(dataset_id))
    safe_dataset_id = safe_path_component(dataset_id)
    return os.path.join(
        semadj_root,
        dataset_group,
        f"{safe_dataset_id}_biological_semantic_spot_adjacency_k{DEFAULT_SEM_K}.npz",
    )


def resolve_runtime_paths(args):
    args.adata_path = args.adata_path or resolve_adata_path(args.dataset_id, args.data_root)
    args.annotation_template_path = args.annotation_template_path or resolve_annotation_template_path(
        args.dataset_id,
        args.annotation_input_root,
    )
    args.gene_embedding_path = args.gene_embedding_path or resolve_gene_embedding_path(
        args.dataset_id,
        args.semadj_root,
    )
    args.save_gene_embedding_path = args.save_gene_embedding_path or args.gene_embedding_path
    if not args.output_semadj_path or args.output_semadj_path == "__AUTO__":
        args.output_semadj_path = resolve_semadj_path(args.dataset_id, args.semadj_root)
    return args


def to_dense_float_array(x):
    if sp.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def l2_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, EPS)


def to_1d_embedding(embedding):
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[0] == 0:
            raise ValueError("GGUF returned an empty token embedding array.")
        return arr.mean(axis=0).astype(np.float32)
    raise ValueError(f"Unexpected GGUF embedding shape: {arr.shape}")


def clean_field(value):
    if value is None or pd.isna(value):
        return NOT_AVAILABLE
    value = str(value).strip()
    if not value:
        return NOT_AVAILABLE
    return value


def build_gene_level_text(ncbi_summary, gene_symbol):
    return (
        f"{GENE_PROMPT_SYSTEM}\n\n"
        "User:\n"
        f"Gene symbol: {clean_field(gene_symbol)}\n"
        f"NCBI gene summary: {clean_field(ncbi_summary)}"
    )


def read_annotation_template(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"annotation_template_path not found: {path}. "
            "Generate it with preprocess/inspect_annotation_inputs.py."
        )

    ext = os.path.splitext(path)[1].lower()
    sep = "\t" if ext in [".tsv", ".txt"] else ","
    df = pd.read_csv(path, sep=sep)
    required_columns = ["gene_symbol", "ncbi_summary"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "annotation_template_path is missing required columns: "
            + ", ".join(missing_columns)
        )

    df["gene_symbol"] = df["gene_symbol"].astype(str)
    return df


def build_gene_text_table(annotation_template_path):
    template = read_annotation_template(annotation_template_path)
    rows = []

    for _, row in template.iterrows():
        gene_symbol = clean_field(row["gene_symbol"])
        ncbi_summary = clean_field(row["ncbi_summary"])
        rows.append({
            "gene_symbol": gene_symbol,
            "ncbi_summary": ncbi_summary,
            "gene_text": build_gene_level_text(
                ncbi_summary=ncbi_summary,
                gene_symbol=gene_symbol,
            ),
        })

    return pd.DataFrame(rows)


def is_gguf_model(model_name_or_path):
    return str(model_name_or_path).lower().endswith(".gguf")


def resolve_gguf_model_path(model_name_or_path):
    if os.path.exists(model_name_or_path):
        return model_name_or_path

    filename = os.path.basename(model_name_or_path)
    parent_root = os.path.dirname(PROJECT_ROOT)
    candidates = [
        DEFAULT_QWEN3_GGUF_MODEL_PATH,
        os.path.join(PROJECT_ROOT, filename),
        os.path.join(PROJECT_ROOT, "models", filename),
        os.path.join(parent_root, filename),
        os.path.join(parent_root, "models", filename),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return model_name_or_path


def generate_gene_embeddings_with_gguf(gene_texts, model_name_or_path, batch_size, max_length):
    try:
        import llama_cpp
        from llama_cpp import Llama
    except (ImportError, OSError, RuntimeError) as exc:
        raise ImportError(
            "Generating embeddings from a GGUF model requires a working "
            "llama-cpp-python installation. If the error mentions GLIBC, the "
            "prebuilt llama-cpp-python wheel is incompatible with this server; "
            "reinstall llama-cpp-python from source in the target conda "
            "environment, or pass --gene_embedding_path to an existing .npy "
            "embedding file, or pass --embedding_model to a non-GGUF "
            "Transformers model path."
        ) from exc

    model_name_or_path = resolve_gguf_model_path(model_name_or_path)
    if not os.path.exists(model_name_or_path):
        raise FileNotFoundError(
            f"GGUF model file was not found: {model_name_or_path}\n"
            "Put Qwen3-30B-A3B-Thinking-2507-UD-Q4_K_XL.gguf in the project models/ "
            "directory, or set SEMST_LLM_MODEL_PATH / --embedding_model to its full path."
        )

    n_gpu_layers = int(os.environ.get("SEMST_GGUF_N_GPU_LAYERS", "-1"))
    n_ctx = int(os.environ.get("SEMST_GGUF_N_CTX", str(max_length)))
    n_batch = int(os.environ.get("SEMST_GGUF_N_BATCH", "512"))
    pooling_type = getattr(llama_cpp, "LLAMA_POOLING_TYPE_MEAN", None)

    print(f"Loading GGUF gene embedding model: {model_name_or_path}")
    print(f"GGUF settings: n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}, n_batch={n_batch}")
    llama_kwargs = dict(
        model_path=model_name_or_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_batch=n_batch,
        embedding=True,
        verbose=False,
    )
    if pooling_type is not None:
        llama_kwargs["pooling_type"] = pooling_type
    llm = Llama(**llama_kwargs)

    outputs = []
    for start in tqdm(range(0, len(gene_texts), batch_size), desc="Embedding gene texts with GGUF"):
        batch_texts = gene_texts[start:start + batch_size]
        response = llm.create_embedding(batch_texts)
        data = sorted(response["data"], key=lambda item: item["index"])
        batch_embeddings = [to_1d_embedding(item["embedding"]) for item in data]
        embedding_dims = {embedding.shape[0] for embedding in batch_embeddings}
        if len(embedding_dims) != 1:
            raise ValueError(f"GGUF returned inconsistent embedding dimensions: {sorted(embedding_dims)}")
        outputs.append(np.stack(batch_embeddings, axis=0).astype(np.float32))

    return np.concatenate(outputs, axis=0).astype(np.float32)


def generate_gene_embeddings(gene_texts, model_name_or_path, batch_size, max_length, device):
    if is_gguf_model(model_name_or_path):
        return generate_gene_embeddings_with_gguf(
            gene_texts=gene_texts,
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            max_length=max_length,
        )

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Generating embeddings requires torch and transformers. "
            "Install them or pass --gene_embedding_path."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_kwargs = {"trust_remote_code": True}
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError('DEFAULT_EMBEDDING_DEVICE is "cuda", but torch.cuda.is_available() is False.')
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval().to(device)
    outputs = []

    with torch.no_grad():
        for start in tqdm(range(0, len(gene_texts), batch_size), desc="Embedding gene texts"):
            batch_texts = gene_texts[start:start + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            model_outputs = model(**inputs, output_hidden_states=False)
            last_hidden = model_outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).to(last_hidden.dtype)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(EPS)
            outputs.append(pooled.float().cpu().numpy())
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

    return np.concatenate(outputs, axis=0).astype(np.float32)


def load_or_generate_gene_embeddings(args, adata):
    gene_symbols = adata.var_names.astype(str).to_numpy()
    gene_text_df = build_gene_text_table(
        annotation_template_path=args.annotation_template_path,
    )
    print(f"Built {len(gene_text_df)} gene prompts from {args.annotation_template_path}")

    if (
        args.gene_embedding_path
        and os.path.exists(args.gene_embedding_path)
        and not args.force_rebuild_gene_embeddings
    ):
        embeddings = np.load(args.gene_embedding_path).astype(np.float32)
        print(f"Loaded cached gene embeddings from {args.gene_embedding_path}")
    else:
        if not args.embedding_model:
            if args.gene_embedding_path:
                raise ValueError(
                    f"Gene embedding file was not found: {args.gene_embedding_path}. "
                    "Set DEFAULT_EMBEDDING_MODEL / pass --embedding_model to generate it, "
                    "or set DEFAULT_GENE_EMBEDDING_PATH to an existing .npy file."
                )
            raise ValueError("Pass --embedding_model to generate gene embeddings, or pass --gene_embedding_path.")
        embeddings = generate_gene_embeddings(
            gene_texts=gene_text_df["gene_text"].astype(str).tolist(),
            model_name_or_path=args.embedding_model,
            batch_size=args.embedding_batch_size,
            max_length=args.embedding_max_length,
            device=args.embedding_device,
        )
        save_embedding_path = args.save_gene_embedding_path or args.gene_embedding_path
        if save_embedding_path:
            output_dir = os.path.dirname(save_embedding_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            np.save(save_embedding_path, embeddings)
            print(f"Saved gene embeddings to {save_embedding_path}")

    if embeddings.shape[0] != len(gene_text_df):
        raise ValueError(
            f"Embedding rows ({embeddings.shape[0]}) do not match annotation template rows ({len(gene_text_df)})."
        )

    index_by_gene = {g: i for i, g in enumerate(gene_text_df["gene_symbol"].astype(str))}
    missing = [g for g in gene_symbols if g not in index_by_gene]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"annotation_template_path is missing {len(missing)} adata genes, e.g. {preview}"
        )
    order = [index_by_gene[g] for g in gene_symbols]
    embeddings = embeddings[order]

    return embeddings


def semantic_cost_matrix(gene_embeddings):
    normalized = l2_normalize(gene_embeddings)
    cost = cosine_distances(normalized).astype(np.float32)
    cost = np.clip(cost, 0.0, None)
    max_cost = float(cost.max())
    if max_cost > 0:
        cost = cost / max_cost
    np.fill_diagonal(cost, 0.0)
    return cost


def get_expression_matrix(adata, layer):
    if layer == "X":
        matrix = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(f'adata.layers["{layer}"] not found.')
        matrix = adata.layers[layer]
    matrix = to_dense_float_array(matrix)
    matrix[matrix < 0] = 0
    return matrix


def get_candidate_expression_matrix(adata, layer, use_pca, pca_dim):
    if layer == "X":
        matrix = to_dense_float_array(adata.X)
    else:
        if layer not in adata.layers:
            raise KeyError(f'adata.layers["{layer}"] not found.')
        matrix = to_dense_float_array(adata.layers[layer])

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if use_pca:
        n_components = min(pca_dim, matrix.shape[0] - 1, matrix.shape[1])
        if n_components <= 0:
            raise ValueError("Cannot run PCA because the candidate expression matrix is too small.")
        matrix = PCA(n_components=n_components, random_state=0).fit_transform(matrix)
    return matrix.astype(np.float32)


def top_l_distributions(expression, top_l):
    n_spots, n_genes = expression.shape
    top_l = min(top_l, n_genes)
    indices = []
    probs = []

    for i in range(n_spots):
        row = expression[i]
        positive = np.flatnonzero(row > 0)
        if positive.size == 0:
            indices.append(np.array([], dtype=np.int64))
            probs.append(np.array([], dtype=np.float32))
            continue

        if positive.size > top_l:
            local = np.argpartition(row[positive], -top_l)[-top_l:]
            chosen = positive[local]
        else:
            chosen = positive

        weights = row[chosen].astype(np.float32)
        total = float(weights.sum())
        if total <= 0:
            indices.append(np.array([], dtype=np.int64))
            probs.append(np.array([], dtype=np.float32))
        else:
            order = np.argsort(-weights)
            chosen = chosen[order]
            weights = weights[order] / total
            indices.append(chosen.astype(np.int64))
            probs.append(weights.astype(np.float32))

    return indices, probs


def sinkhorn_distance(p, q, cost, reg, max_iter, tol):
    if p.size == 0 or q.size == 0:
        return np.inf
    if reg <= 0:
        raise ValueError("--sinkhorn_reg must be positive.")

    kernel = np.exp(-cost / reg).astype(np.float64)
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    u = np.ones_like(p)
    v = np.ones_like(q)

    for _ in range(max_iter):
        u_prev = u
        u = p / np.maximum(kernel.dot(v), EPS)
        v = q / np.maximum(kernel.T.dot(u), EPS)
        if np.max(np.abs(u - u_prev)) < tol:
            break

    transport = (u[:, None] * kernel) * v[None, :]
    return float(np.sum(transport * cost))


def candidate_neighbors(expression, candidate_k, metric):
    n_spots = expression.shape[0]
    if candidate_k <= 0 or candidate_k >= n_spots - 1:
        return [np.array([j for j in range(n_spots) if j != i], dtype=np.int64) for i in range(n_spots)]

    n_neighbors = min(candidate_k + 1, n_spots)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nbrs.fit(expression)
    _, indices = nbrs.kneighbors(expression)

    neighbors = []
    for i in range(n_spots):
        row = indices[i]
        row = row[row != i]
        neighbors.append(row.astype(np.int64))
    return neighbors


def build_semantic_ot_graph(ot_expression, candidate_expression, gene_cost, top_l, candidate_k, sem_k,
                            sinkhorn_reg, sinkhorn_iters, sinkhorn_tol,
                            candidate_metric, disable_candidate_selection,
                            return_distances=False):
    n_spots = ot_expression.shape[0]
    if candidate_expression.shape[0] != n_spots:
        raise ValueError("OT expression and candidate expression must have the same number of spots.")

    top_indices, top_probs = top_l_distributions(ot_expression, top_l)
    if disable_candidate_selection:
        neighbors = [
            np.array([j for j in range(n_spots) if j != i], dtype=np.int64)
            for i in range(n_spots)
        ]
        print("Candidate selection disabled: computing semantic OT against all other spots.")
    else:
        neighbors = candidate_neighbors(candidate_expression, candidate_k, candidate_metric)

    rows = []
    cols = []
    data = []
    spot_distances = None
    if return_distances:
        spot_distances = np.full((n_spots, n_spots), np.inf, dtype=np.float32)
        np.fill_diagonal(spot_distances, 0.0)

    for i in tqdm(range(n_spots), desc="Computing semantic OT graph"):
        pair_distances = []
        pi = top_probs[i]
        gi = top_indices[i]
        if pi.size == 0:
            continue

        for j in neighbors[i]:
            qj = top_probs[j]
            gj = top_indices[j]
            if qj.size == 0:
                continue
            sub_cost = gene_cost[np.ix_(gi, gj)]
            distance = sinkhorn_distance(
                pi,
                qj,
                sub_cost,
                reg=sinkhorn_reg,
                max_iter=sinkhorn_iters,
                tol=sinkhorn_tol,
            )
            if np.isfinite(distance):
                pair_distances.append((int(j), distance))
                if spot_distances is not None:
                    spot_distances[i, int(j)] = np.float32(distance)

        if not pair_distances:
            continue
        pair_distances.sort(key=lambda x: x[1])
        selected = pair_distances[:min(sem_k, len(pair_distances))]
        for j, distance in selected:
            rows.append(i)
            cols.append(j)
            data.append(1.0)

    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_spots, n_spots), dtype=np.float32)
    adj = adj.maximum(adj.T).tocoo()
    adj.setdiag(0)
    adj.eliminate_zeros()
    if spot_distances is not None:
        spot_distances = np.minimum(spot_distances, spot_distances.T)
        np.fill_diagonal(spot_distances, 0.0)
        return adj, spot_distances
    return adj


def save_outputs(adata, semadj, args):
    if args.output_semadj_path:
        output_dir = os.path.dirname(args.output_semadj_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        sp.save_npz(args.output_semadj_path, semadj)
        print(f"Saved semantic OT graph to {args.output_semadj_path}")

    if args.output_adata_path:
        output_dir = os.path.dirname(args.output_adata_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        adata.obsp[args.sem_adj_key] = semadj.tocsr()
        adata.write(args.output_adata_path)
        print(f'Saved AnnData with adata.obsp["{args.sem_adj_key}"] to {args.output_adata_path}')


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--annotation_input_root", type=str, default=DEFAULT_ANNOTATION_INPUT_ROOT)
    parser.add_argument("--semadj_root", type=str, default=DEFAULT_SEMADJ_ROOT)
    parser.add_argument("--adata_path", type=str, default=DEFAULT_ADATA_PATH)
    parser.add_argument("--annotation_template_path", type=str, default=DEFAULT_ANNOTATION_TEMPLATE_PATH,
                        help="CSV/TSV generated by preprocess/inspect_annotation_inputs.py. Must contain gene_symbol and ncbi_summary.")
    parser.add_argument("--gene_embedding_path", type=str, default=DEFAULT_GENE_EMBEDDING_PATH)
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_MODEL,
                        help="Optional local/HF model path used only when --gene_embedding_path is absent.")
    parser.add_argument("--embedding_device", type=str, default=DEFAULT_EMBEDDING_DEVICE)
    parser.add_argument("--embedding_batch_size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--embedding_max_length", type=int, default=DEFAULT_EMBEDDING_MAX_LENGTH)
    parser.add_argument("--save_gene_embedding_path", type=str, default=DEFAULT_SAVE_GENE_EMBEDDING_PATH)
    parser.add_argument("--force_rebuild_gene_embeddings", action="store_true",
                        help="Regenerate gene embeddings from annotation_template_path even if gene_embedding_path exists.")
    parser.add_argument("--ot_expression_layer", type=str, default=DEFAULT_OT_EXPRESSION_LAYER,
                        help='Expression source for spot gene distributions used in OT. Defaults to preprocessed adata.X.')
    parser.add_argument("--candidate_expression_layer", type=str, default=DEFAULT_CANDIDATE_EXPRESSION_LAYER,
                        help='Expression source for candidate neighbor search. Defaults to preprocessed adata.X.')
    parser.add_argument("--candidate_use_pca", action="store_true", default=DEFAULT_CANDIDATE_USE_PCA,
                        help="Run PCA before candidate neighbor search.")
    parser.add_argument("--candidate_pca_dim", type=int, default=DEFAULT_CANDIDATE_PCA_DIM)
    parser.add_argument("--top_l", type=int, default=DEFAULT_TOP_L)
    parser.add_argument("--candidate_k", type=int, default=DEFAULT_CANDIDATE_K,
                        help="Expression-neighbor candidates per spot before OT. Use <=0 for all pairs.")
    parser.add_argument("--disable_candidate_selection", action="store_true",
                        default=DEFAULT_DISABLE_CANDIDATE_SELECTION,
                        help="Disable expression-based candidate screening and compute OT from each spot to all other spots.")
    parser.add_argument("--candidate_metric", type=str, default=DEFAULT_CANDIDATE_METRIC)
    parser.add_argument("--sem_k", type=int, default=DEFAULT_SEM_K)
    parser.add_argument("--sinkhorn_reg", type=float, default=DEFAULT_SINKHORN_REG)
    parser.add_argument("--sinkhorn_iters", type=int, default=DEFAULT_SINKHORN_ITERS)
    parser.add_argument("--sinkhorn_tol", type=float, default=DEFAULT_SINKHORN_TOL)
    parser.add_argument("--sem_adj_key", type=str, default=DEFAULT_SEM_ADJ_KEY)
    parser.add_argument("--output_semadj_path", type=str, default=DEFAULT_OUTPUT_SEMADJ_PATH)
    parser.add_argument("--output_adata_path", type=str, default=DEFAULT_OUTPUT_ADATA_PATH)
    parser.add_argument("--save_gene_cost_path", type=str, default=DEFAULT_SAVE_GENE_COST_PATH)
    parser.add_argument("--save_spot_ot_distance_path", type=str, default=None)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args = resolve_runtime_paths(args)
    print(f"dataset_id: {args.dataset_id}")
    print(f"adata_path: {args.adata_path}")
    print(f"annotation_template_path: {args.annotation_template_path}")
    print(f"gene_embedding_path: {args.gene_embedding_path}")
    print(f"save_gene_embedding_path: {args.save_gene_embedding_path}")
    print(f"output_semadj_path: {args.output_semadj_path}")
    if not args.output_semadj_path and not args.output_adata_path:
        raise ValueError("Pass --output_semadj_path, --output_adata_path, or both.")

    adata = sc.read_h5ad(args.adata_path)
    print(f"Loaded preprocessed AnnData from {args.adata_path}")
    print(f"adata.X shape: {adata.X.shape}")

    gene_embeddings = load_or_generate_gene_embeddings(args, adata)
    gene_cost = semantic_cost_matrix(gene_embeddings)
    if args.save_gene_cost_path:
        output_dir = os.path.dirname(args.save_gene_cost_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        np.save(args.save_gene_cost_path, gene_cost)
        print(f"Saved gene semantic cost matrix to {args.save_gene_cost_path}")

    ot_expression = get_expression_matrix(adata, args.ot_expression_layer)
    print(f"OT expression layer: {args.ot_expression_layer}")
    if args.disable_candidate_selection:
        candidate_expression = ot_expression
        print("Candidate selection disabled: computing OT from each spot to all other spots.")
    else:
        candidate_expression = get_candidate_expression_matrix(
            adata,
            layer=args.candidate_expression_layer,
            use_pca=args.candidate_use_pca,
            pca_dim=args.candidate_pca_dim,
        )
        print(f"Candidate expression layer: {args.candidate_expression_layer}")

    semadj_result = build_semantic_ot_graph(
        ot_expression=ot_expression,
        candidate_expression=candidate_expression,
        gene_cost=gene_cost,
        top_l=args.top_l,
        candidate_k=args.candidate_k,
        sem_k=args.sem_k,
        sinkhorn_reg=args.sinkhorn_reg,
        sinkhorn_iters=args.sinkhorn_iters,
        sinkhorn_tol=args.sinkhorn_tol,
        candidate_metric=args.candidate_metric,
        disable_candidate_selection=args.disable_candidate_selection,
        return_distances=bool(args.save_spot_ot_distance_path),
    )
    if args.save_spot_ot_distance_path:
        semadj, spot_ot_distance = semadj_result
        output_dir = os.path.dirname(args.save_spot_ot_distance_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(args.save_spot_ot_distance_path, ot_distance=spot_ot_distance)
        print(f"Saved spot OT distance matrix to {args.save_spot_ot_distance_path}")
    else:
        semadj = semadj_result

    print(f"Semantic graph: {semadj.shape[0]} spots, {semadj.nnz} nonzero directed entries after symmetrization.")
    save_outputs(adata, semadj, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


