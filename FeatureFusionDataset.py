import os
import torch
from torch.utils.data import Dataset

from utils.ast_to_graph_5_4_1 import load_single_ast_json
from utils.pdg_to_graph_qemu import load_dot_as_pyg_graph_with_features
from models.hgcn import TrueHGCN
from models.bayesian_gcn import AdvancedBayesianGATGCN

from torch_geometric.data import Data

def resolve_real_dot_file(pdg_path: str) -> str:
    """
    输入:
        - 真实 .dot 文件路径
        - 或形如 xxx/xxxx.dot/ 的目录
    输出:
        - 一个可被 networkx.read_dot 读取的 .dot 文件
    """
    if os.path.isfile(pdg_path) and pdg_path.endswith(".dot"):
        return pdg_path

    if not os.path.isdir(pdg_path):
        raise FileNotFoundError(f"[PDG] Path not found: {pdg_path}")

    # 扫描目录内所有 .dot 文件
    dot_files = []
    for root, _, files in os.walk(pdg_path):
        for f in files:
            if f.endswith(".dot"):
                dot_files.append(os.path.join(root, f))

    if len(dot_files) == 0:
        raise RuntimeError(f"[PDG] No .dot file found in directory: {pdg_path}")

    dot_files.sort()
    return dot_files[0]


def extract_graph_structure_features(pyg_data):
    x = pyg_data.x
    num_nodes = max(1, x.size(0))
    avg_degree = pyg_data.edge_index.size(1) / num_nodes if hasattr(pyg_data, 'edge_index') else 0.0
    max_centrality = float(x[:, -2].max().item()) if x.size(1) >= 2 else 0.0
    avg_pagerank = float(x[:, -1].mean().item()) if x.size(1) >= 1 else 0.0
    return torch.tensor([avg_degree, max_centrality, avg_pagerank], dtype=torch.float)

# ==========================================================
# FusionFeatureDataset
# ==========================================================
class FusionFeatureDataset(Dataset):
    def __init__(self, ast_root, pdg_root, train=True):
        self.train = train
        self.samples = []


        ast_map = {}
        for label_dir in ["Vul", "No-Vul"]:
            full_dir = os.path.join(ast_root, label_dir)
            if not os.path.isdir(full_dir):
                continue
            for f in os.listdir(full_dir):
                if f.endswith(".json"):
                    key = os.path.splitext(f)[0]
                    ast_map[key] = os.path.join(full_dir, f)


        pdg_map = {}
        for label_dir, label in [("Vul", 1), ("No-Vul", 0)]:
            full_dir = os.path.join(pdg_root, label_dir)
            if not os.path.isdir(full_dir):
                continue

            for name in os.listdir(full_dir):
                entry_path = os.path.join(full_dir, name)
                key = os.path.splitext(name)[0]

                # 单文件或目录（兼容目录以 .dot 结尾）
                if (os.path.isfile(entry_path) and name.endswith(".dot")) or \
                   (os.path.isdir(entry_path) and name.endswith(".dot")):
                    pdg_map[key] = (entry_path, label)


        common_keys = sorted(set(ast_map) & set(pdg_map))
        for k in common_keys:
            self.samples.append((k, ast_map[k], pdg_map[k][0], pdg_map[k][1]))


        if len(self.samples) == 0:
            raw_asts = list(ast_map.values())
            raw_pdgs = list(pdg_map.values())
            min_len = min(len(raw_asts), len(raw_pdgs))
            for i in range(min_len):
                self.samples.append((f"idx_{i}", raw_asts[i], raw_pdgs[i][0], raw_pdgs[i][1]))


        if len(self.samples) == 0:
            raise RuntimeError(
                "[FATAL] FusionFeatureDataset 构建失败\n"
                f"AST_ROOT = {ast_root}\n"
                f"PDG_ROOT = {pdg_root}\n"
            )


        ast_sample = load_single_ast_json(self.samples[0][1])
        self.ast_model = TrueHGCN(
            in_channels=ast_sample.x.size(1),
            hidden_channels=64,
            out_channels=64
        ).eval()


        pdg_sample_path = resolve_real_dot_file(self.samples[0][2])
        pdg_sample = load_dot_as_pyg_graph_with_features(pdg_sample_path)

        self.pdg_model = AdvancedBayesianGATGCN(
            in_channels=pdg_sample.x.size(1),
            hidden_channels=64,
            out_channels=64
        ).eval()


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, ast_path, pdg_path, label = self.samples[idx]

        # ---------- AST ----------
        ast_data = load_single_ast_json(ast_path)
        ast_data.batch = torch.zeros(ast_data.num_nodes, dtype=torch.long)
        with torch.no_grad():
            ast_feat = self.ast_model(ast_data.x, ast_data.edge_index, ast_data.batch).view(-1)

        # ---------- PDG ----------
        pdg_real_path = resolve_real_dot_file(pdg_path)
        pdg_data = load_dot_as_pyg_graph_with_features(pdg_real_path)
        pdg_data.batch = torch.zeros(pdg_data.num_nodes, dtype=torch.long)

        with torch.no_grad():
            if pdg_data.num_nodes == 0:
                pdg_feat = torch.zeros(self.pdg_model.out_channels, dtype=torch.float)
            else:
                _, pdg_feat = self.pdg_model(pdg_data.x, pdg_data.edge_index, pdg_data.batch)
                pdg_feat = pdg_feat.view(-1)

        # ---------- 节点统计 ----------
        if pdg_data.num_nodes == 0 or pdg_data.x.size(0) == 0:
            node_info1 = torch.zeros(pdg_data.x.size(1), dtype=torch.float)
            node_info2 = torch.zeros_like(node_info1)
        else:
            node_info1 = pdg_data.x.mean(dim=0)
            node_info2 = pdg_data.x.max(dim=0)[0]

        # ---------- 结构特征 ----------
        struct_feat = extract_graph_structure_features(pdg_data)

        # ---------- 训练扰动 ----------
        if self.train:
            node_info1 += torch.randn_like(node_info1) * 0.01
            node_info2 += torch.randn_like(node_info2) * 0.01
            scale = torch.randn(1).item() * 0.05 + 1.0
            ast_feat *= scale
            pdg_feat *= scale
            struct_feat *= scale

        return ast_feat.float(), pdg_feat.float(), struct_feat.float(), node_info1.float(), node_info2.float(), label
