# ==========================================================
# pdg_to_graph_qemu.py  (FINAL STABLE VERSION)
# 功能：
#   - 支持单 .dot 文件或 .dot 目录
#   - 节点语义 OneHot 编码（强制维度稳定）
#   - 结构特征计算（5 维）
#   - degree 特征（1 维）
#   - 输出 PyG Data（任何情况下维度一致）
# ==========================================================

import os
import torch
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.utils import from_networkx, degree
from torch_geometric.data import Data
from typing import List

# ==========================================================
# 全局 OneHotEncoder
# ==========================================================
global_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
global_vocab_fitted = False


# ==========================================================
# 工具：递归获取真实 .dot 文件
# ==========================================================
def _expand_real_dot_files(path: str) -> List[str]:
    real_dots = []
    if os.path.isfile(path) and path.endswith(".dot"):
        real_dots.append(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".dot"):
                    real_dots.append(os.path.join(root, f))
    return real_dots


# ==========================================================
# 构建全局节点语义词表
# ==========================================================
def build_global_node_vocab(dot_paths: List[str]):
    all_labels = []
    real_dot_files = []

    for p in dot_paths:
        real_dot_files.extend(_expand_real_dot_files(p))

    if len(real_dot_files) == 0:
        raise RuntimeError("[ERROR] No real .dot files found.")

    for path in real_dot_files:
        try:
            graph = nx.drawing.nx_pydot.read_dot(path)
            for _, attr in graph.nodes(data=True):
                label = str(attr.get("label", "")).strip('"')
                all_labels.append([label])
        except Exception:
            continue

    if len(all_labels) == 0:
        raise RuntimeError("[ERROR] No valid node labels found. Cannot build vocab.")

    global global_encoder, global_vocab_fitted
    global_encoder.fit(all_labels)
    global_vocab_fitted = True

    print(f"[OK] OneHot vocab built, size = {len(global_encoder.categories_[0])}")


def init_global_encoder(dot_paths: List[str]):
    if not global_vocab_fitted:
        build_global_node_vocab(dot_paths)


# ==========================================================
# 安全 OneHot（强制维度对齐）
# ==========================================================
def _safe_semantic_onehot(labels: List[List[str]], vocab_size: int, num_nodes: int):
    """
    保证输出 shape = [num_nodes, vocab_size]
    """
    if labels is None or len(labels) == 0:
        return torch.zeros((num_nodes, vocab_size), dtype=torch.float)

    try:
        feat = global_encoder.transform(labels)
        feat = torch.tensor(feat, dtype=torch.float)

        if feat.size(1) != vocab_size:
            fixed = torch.zeros((feat.size(0), vocab_size), dtype=torch.float)
            fixed[:, :feat.size(1)] = feat
            return fixed

        return feat
    except Exception:
        return torch.zeros((num_nodes, vocab_size), dtype=torch.float)


# ==========================================================
# 加载单个 dot（或目录）为 PyG Data
# ==========================================================
def load_dot_as_pyg_graph_with_features(dot_path: str) -> Data:
    if not global_vocab_fitted:
        raise RuntimeError("[ERROR] Encoder not initialized. Call init_global_encoder() first.")

    # ------------------------------
    # 目录兼容
    # ------------------------------
    if os.path.isdir(dot_path):
        real_dots = _expand_real_dot_files(dot_path)
        if len(real_dots) == 0:
            raise RuntimeError(f"[ERROR] No .dot files found in directory: {dot_path}")
        real_dots.sort()
        dot_path = real_dots[0]

    # ------------------------------
    # 读取 dot
    # ------------------------------
    try:
        graph = nx.drawing.nx_pydot.read_dot(dot_path)
    except Exception:
        graph = nx.Graph()

    node_names = list(graph.nodes)
    vocab_size = len(global_encoder.categories_[0])
    expected_dim = vocab_size + 6  # 5 structure + 1 degree

    # ------------------------------
    # 空图兜底
    # ------------------------------
    if len(node_names) == 0:
        return Data(
            x=torch.zeros((1, expected_dim), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )

    # ------------------------------
    # 节点语义 OneHot（稳定）
    # ------------------------------
    semantic_labels = [
        [str(graph.nodes[n].get("label", "")).strip('"')]
        for n in node_names
    ]

    semantic_features = _safe_semantic_onehot(
        semantic_labels, vocab_size, len(node_names)
    )

    # ------------------------------
    # 结构特征
    # ------------------------------
    undirected = graph.to_undirected()

    centrality = nx.degree_centrality(undirected)
    degree_dict = dict(undirected.degree())
    closeness = nx.closeness_centrality(undirected)
    pagerank = nx.pagerank(undirected)

    info_centrality = {n: 0.0 for n in undirected.nodes}
    try:
        if len(undirected.nodes) > 1:
            if nx.is_connected(undirected):
                info_centrality.update(nx.current_flow_closeness_centrality(undirected))
            else:
                largest_cc = max(nx.connected_components(undirected), key=len)
                sub = undirected.subgraph(largest_cc).copy()
                info_centrality.update(nx.current_flow_closeness_centrality(sub))
    except Exception:
        pass

    for n in graph.nodes:
        graph.nodes[n]["centrality"] = float(centrality.get(n, 0.0))
        graph.nodes[n]["degree"] = float(degree_dict.get(n, 0.0))
        graph.nodes[n]["closeness"] = float(closeness.get(n, 0.0))
        graph.nodes[n]["pagerank"] = float(pagerank.get(n, 0.0))
        graph.nodes[n]["info_centrality"] = float(info_centrality.get(n, 0.0))

    pyg_data = from_networkx(graph)

    # ------------------------------
    # 结构特征拼接（5 维）
    # ------------------------------
    struct_feats = []
    for k in ["centrality", "degree", "closeness", "pagerank", "info_centrality"]:
        struct_feats.append(
            torch.tensor(
                [graph.nodes[n].get(k, 0.0) for n in node_names],
                dtype=torch.float
            ).view(-1, 1)
        )
    structure_feat = torch.cat(struct_feats, dim=1)

    # ------------------------------
    # degree 特征（1 维）
    # ------------------------------
    if pyg_data.edge_index.numel() > 0:
        deg_tensor = degree(
            pyg_data.edge_index[0],
            num_nodes=pyg_data.num_nodes
        ).view(-1, 1)
    else:
        deg_tensor = torch.zeros((pyg_data.num_nodes, 1), dtype=torch.float)

    # ------------------------------
    # 最终拼接 + 强制校验
    # ------------------------------
    pyg_data.x = torch.cat(
        [semantic_features, structure_feat, deg_tensor], dim=1
    )

    if pyg_data.x.size(1) != expected_dim:
        raise RuntimeError(
            f"[PDG FEATURE ERROR] Feature dim mismatch: "
            f"{pyg_data.x.size(1)} != {expected_dim} | file = {dot_path}"
        )

    # ------------------------------
    # 最终兜底（防止空节点）
    # ------------------------------
    if pyg_data.x.size(0) == 0:
        pyg_data.x = torch.zeros((1, expected_dim), dtype=torch.float)
        pyg_data.edge_index = torch.zeros((2, 0), dtype=torch.long)

    return pyg_data
