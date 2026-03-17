# # utils/ast_to_graph_5_4_1.py
# import os
# import json
# import torch
# import torch.nn.functional as F
# from torch_geometric.data import Data
#
# def build_global_vocab(root_folder=None, file_list=None, save_path="vocab_ast.pt", force_rebuild=False):
#     """
#     构建或加载 AST 全局节点类型词表（vocab），仅基于训练集。
#     - root_folder: (可选) 根目录，包含 Vul/No-Vul 子目录。
#     - file_list: (推荐) 仅用于训练集文件路径列表（优先使用）。
#     - save_path: 保存路径（默认 vocab_ast.pt）。
#     - force_rebuild: True 则强制重建并覆盖文件。
#     """
#     if os.path.exists(save_path) and not force_rebuild:
#         vocab = torch.load(save_path)
#         print(f"[Info] ✅ 已加载现有词表: {save_path} ({len(vocab)} types)")
#         return vocab
#
#     node_types = set()
#
#     def collect_types(node):
#         node_type = node.get('type', None)
#         if node_type:
#             node_types.add(node_type)
#         for c in node.get('children', []):
#             collect_types(c)
#
#     if file_list is not None:
#         # 使用训练集的 json 文件路径列表
#         for fpath in file_list:
#             try:
#                 with open(fpath, 'r', encoding='utf-8') as f:
#                     ast_json = json.load(f)
#                     collect_types(ast_json)
#             except Exception as e:
#                 print(f"[!] Error parsing {fpath}: {e}")
#     elif root_folder is not None:
#         # 兼容旧逻辑（使用整个目录）
#         for label in ['Vul', 'No-Vul']:
#             folder = os.path.join(root_folder, label)
#             if not os.path.exists(folder):
#                 continue
#             for filename in os.listdir(folder):
#                 if filename.endswith('.json'):
#                     fpath = os.path.join(folder, filename)
#                     try:
#                         with open(fpath, 'r', encoding='utf-8') as f:
#                             ast_json = json.load(f)
#                             collect_types(ast_json)
#                     except Exception as e:
#                         print(f"[!] Error parsing {fpath}: {e}")
#
#     vocab = {nt: idx for idx, nt in enumerate(sorted(node_types))}
#     torch.save(vocab, save_path)
#     print(f"[Info] ✅ 构建新词表并保存至 {save_path}, 共 {len(vocab)} 个节点类型")
#     return vocab
#
#
# def parse_ast_json_to_pyg(ast_json, label, vocab):
#     nodes = []
#     edges = []
#
#     def traverse(node, parent_id=None):
#         node_id = len(nodes)
#         nodes.append(node.get('type', 'Unknown'))
#         if parent_id is not None:
#             edges.append((parent_id, node_id))
#         for c in node.get('children', []):
#             traverse(c, node_id)
#
#     traverse(ast_json)
#     indices = [vocab.get(t, 0) for t in nodes]
#     if len(indices) == 0:
#         indices = [0]
#     x = F.one_hot(torch.tensor(indices), num_classes=len(vocab)).to(torch.float)
#
#     if edges:
#         edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#     else:
#         edge_index = torch.empty((2, 0), dtype=torch.long)
#
#     y = torch.tensor([1 if label == 'Vul' else 0], dtype=torch.long)
#     return Data(x=x, edge_index=edge_index, y=y)
#
#
# def load_ast_graph_from_json(json_path, vocab_path="vocab_ast.pt"):
#     if not os.path.exists(vocab_path):
#         raise FileNotFoundError(f"[✘] AST vocab not found: {vocab_path}. Run build_global_vocab on training data first.")
#     vocab = torch.load(vocab_path)
#     with open(json_path, 'r', encoding='utf-8') as f:
#         ast_json = json.load(f)
#     return parse_ast_json_to_pyg(ast_json, label='No-Vul', vocab=vocab)
#
#
# def load_single_ast_json(json_path, vocab_path="vocab_ast.pt"):
#     return load_ast_graph_from_json(json_path, vocab_path=vocab_path)
#对应5——4

# utils/ast_to_graph_5_4_1.py
import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


# =========================================================
# 构建 / 加载 AST 全局词表
# =========================================================
def build_global_vocab(root_folder=None, file_list=None,
                       save_path="vocab_ast.pt", force_rebuild=False):
    """
    构建或加载 AST 全局节点类型词表（vocab）
    """
    if os.path.exists(save_path) and not force_rebuild:
        vocab = torch.load(save_path)
        print(f"[Info] ✅ 已加载 AST 词表: {save_path} ({len(vocab)} types)")
        return vocab

    node_types = set()

    def collect_types(node):
        node_type = node.get("type", None)
        if node_type:
            node_types.add(node_type)
        for c in node.get("children", []):
            collect_types(c)

    if file_list is not None:
        for fpath in file_list:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    ast_json = json.load(f)
                collect_types(ast_json)
            except Exception as e:
                print(f"[Warn] AST parse failed: {fpath} ({e})")

    elif root_folder is not None:
        for label in ["Vul", "No-Vul"]:
            folder = os.path.join(root_folder, label)
            if not os.path.isdir(folder):
                continue
            for fn in os.listdir(folder):
                if fn.endswith(".json"):
                    fpath = os.path.join(folder, fn)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            ast_json = json.load(f)
                        collect_types(ast_json)
                    except Exception as e:
                        print(f"[Warn] AST parse failed: {fpath} ({e})")

    vocab = {nt: i for i, nt in enumerate(sorted(node_types))}
    torch.save(vocab, save_path)
    print(f"[Info] ✅ 构建新 AST 词表: {save_path} ({len(vocab)} types)")
    return vocab


# =========================================================
# AST JSON → PyG Graph
# =========================================================
def parse_ast_json_to_pyg(ast_json, label, vocab):
    nodes = []
    edges = []

    def traverse(node, parent_id=None):
        node_id = len(nodes)
        nodes.append(node.get("type", "Unknown"))
        if parent_id is not None:
            edges.append((parent_id, node_id))
        for c in node.get("children", []):
            traverse(c, node_id)

    traverse(ast_json)

    if len(nodes) == 0:
        nodes = ["Unknown"]

    indices = [vocab.get(t, 0) for t in nodes]
    x = F.one_hot(
        torch.tensor(indices),
        num_classes=len(vocab)
    ).float()

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    y = torch.tensor([1 if label == "Vul" else 0], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


# =========================================================
# 新接口：加载单个 AST JSON
# =========================================================
def load_ast_graph_from_json(json_path, vocab_path="vocab_ast.pt"):
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"[✘] AST vocab not found: {vocab_path}. "
            f"Please run build_global_vocab first."
        )

    vocab = torch.load(vocab_path)

    with open(json_path, "r", encoding="utf-8") as f:
        ast_json = json.load(f)

    return parse_ast_json_to_pyg(ast_json, label="No-Vul", vocab=vocab)


def load_single_ast_json(json_path, vocab_path="vocab_ast.pt"):
    return load_ast_graph_from_json(json_path, vocab_path)


# =========================================================
# 🔥 兼容旧代码的关键修复函数（本次重点）
# =========================================================
def load_dataset_from_folder(folder_path, vocab_path="vocab_ast.pt"):
    """
    【兼容接口】
    旧版本 FeatureFusionDataset / train 脚本依赖此函数
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"[✘] Folder not found: {folder_path}")

    graphs = []
    for fn in os.listdir(folder_path):
        if fn.endswith(".json"):
            fpath = os.path.join(folder_path, fn)
            try:
                g = load_ast_graph_from_json(fpath, vocab_path)
                graphs.append(g)
            except Exception as e:
                print(f"[Warn] Failed to load AST: {fpath} ({e})")

    return graphs
