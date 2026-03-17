
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from dataset.FeatureFusionDataset import FusionFeatureDataset
from models.fusion_mlp import FusionMLP
from utils.pdg_to_graph import build_global_node_vocab

# ------------------- 参数配置 -------------------
AST_ROOT = "./ast_json_output"
PDG_ROOT = "./pdg_dot_files"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 100
LR = 5e-4
PATIENCE = 20
MC_T = 8

NODE_GAUSSIAN_STD = 0.02
GRAPH_SCALE_LOW = 0.95
GRAPH_SCALE_HIGH = 1.05

# ------------------- 混淆矩阵 -------------------
def plot_confusion(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["No-Vul", "Vul"])
    plt.yticks([0, 1], ["No-Vul", "Vul"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.savefig(save_path)
    plt.close()

# ------------------- collate_fn -------------------
def collate_fn(batch):
    ast, pdg, struct, n1, n2, y = [], [], [], [], [], []
    for (a, p, s, x1, x2), label in batch:
        ast.append(a)
        pdg.append(p)
        struct.append(s)
        n1.append(x1)
        n2.append(x2)
        y.append(label)
    return (ast, pdg, struct, n1, n2), y

# ===============================================================
def train():
    dot_paths = []
    for sub in ["Vul", "No-Vul"]:
        d = os.path.join(PDG_ROOT, sub)
        if os.path.isdir(d):
            dot_paths += [
                os.path.join(d, f)
                for f in os.listdir(d)
                if f.endswith(".dot")
            ]
    if dot_paths:
        build_global_node_vocab(dot_paths)

    raw_dataset = FusionFeatureDataset(AST_ROOT, PDG_ROOT)

    data, labels = [], []
    for i in tqdm(range(len(raw_dataset)), desc="加载样本"):
        ast, pdg, struct, n1, n2, y = raw_dataset[i]
        data.append(((ast, pdg, struct, n1, n2), int(y)))
        labels.append(int(y))

    trainval_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.1,
        stratify=labels,
        random_state=42
    )

    trainval_data = [data[i] for i in trainval_idx]
    trainval_labels = [labels[i] for i in trainval_idx]
    test_data = [data[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    tr_idx, va_idx = train_test_split(
        np.arange(len(trainval_labels)),
        test_size=0.1,
        stratify=trainval_labels,
        random_state=42
    )

    train_loader = DataLoader(
        [trainval_data[i] for i in tr_idx],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        [trainval_data[i] for i in va_idx],
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn
    )

    sa, sp, ss, sn1, sn2 = trainval_data[0][0]
    model = FusionMLP(
        sa.numel(), sp.numel(), ss.numel(),
        sn1.numel(), sn2.numel(),
        hidden_dim=128,
        num_classes=2,
        dropout=0.3
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.6, patience=3
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 1.5]).to(DEVICE)
    )

    best_f1, wait = 0.0, 0

    # ================== Train ==================
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_pred, tr_true = [], []

        for (a, p, s, n1, n2), y in tqdm(train_loader, desc=f"[Epoch {epoch}]"):
            a = torch.stack(a).to(DEVICE)
            p = torch.stack(p).to(DEVICE)
            s = torch.stack(s).to(DEVICE)
            n1 = torch.stack(n1).to(DEVICE)
            n2 = torch.stack(n2).to(DEVICE)
            y = torch.tensor(y).to(DEVICE)

            scale = torch.rand(p.size(0), 1, device=DEVICE) * \
                    (GRAPH_SCALE_HIGH - GRAPH_SCALE_LOW) + GRAPH_SCALE_LOW
            p = p * scale
            n1 = n1 + torch.randn_like(n1) * NODE_GAUSSIAN_STD
            n2 = n2 + torch.randn_like(n2) * NODE_GAUSSIAN_STD

            optimizer.zero_grad()
            logits, _ = model(a, p, s, n1, n2, mc_T=1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            tr_pred += logits.argmax(1).cpu().tolist()
            tr_true += y.cpu().tolist()

        tr_acc = accuracy_score(tr_true, tr_pred)
        tr_pre = precision_score(tr_true, tr_pred, average="macro", zero_division=0)
        tr_rec = recall_score(tr_true, tr_pred, average="macro", zero_division=0)
        tr_f1 = f1_score(tr_true, tr_pred, average="macro")

        print(
            f"[Epoch {epoch}] "
            f"Train Acc={tr_acc:.4f} "
            f"P={tr_pre:.4f} R={tr_rec:.4f} F1={tr_f1:.4f}"
        )

        # ================== Test (每个 Epoch) ==================
        model.eval()
        te_pred, te_true = [], []
        with torch.no_grad():
            for (a, p, s, n1, n2), y in test_loader:
                a = torch.stack(a).to(DEVICE)
                p = torch.stack(p).to(DEVICE)
                s = torch.stack(s).to(DEVICE)
                n1 = torch.stack(n1).to(DEVICE)
                n2 = torch.stack(n2).to(DEVICE)
                probs, _ = model(a, p, s, n1, n2, mc_T=MC_T)
                te_pred += probs.argmax(1).cpu().tolist()
                te_true += y

        te_acc = accuracy_score(te_true, te_pred)
        te_pre = precision_score(te_true, te_pred, average="macro", zero_division=0)
        te_rec = recall_score(te_true, te_pred, average="macro", zero_division=0)
        te_f1 = f1_score(te_true, te_pred, average="macro")

        print(
            f"           Test  "
            f"Acc={te_acc:.4f} "
            f"P={te_pre:.4f} R={te_rec:.4f} F1={te_f1:.4f}"
        )

        # ================== Val ==================
        va_pred, va_true = [], []
        with torch.no_grad():
            for (a, p, s, n1, n2), y in val_loader:
                a = torch.stack(a).to(DEVICE)
                p = torch.stack(p).to(DEVICE)
                s = torch.stack(s).to(DEVICE)
                n1 = torch.stack(n1).to(DEVICE)
                n2 = torch.stack(n2).to(DEVICE)
                probs, _ = model(a, p, s, n1, n2, mc_T=MC_T)
                va_pred += probs.argmax(1).cpu().tolist()
                va_true += y

        va_f1 = f1_score(va_true, va_pred, average="macro")
        scheduler.step(va_f1)

        if va_f1 > best_f1:
            best_f1 = va_f1
            wait = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("    💾 Saved best model.")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("    ⏹️ Early stopping triggered.")
                break

    # ================== Final Test ==================
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    te_pred = []
    with torch.no_grad():
        for (a, p, s, n1, n2), y in test_loader:
            a = torch.stack(a).to(DEVICE)
            p = torch.stack(p).to(DEVICE)
            s = torch.stack(s).to(DEVICE)
            n1 = torch.stack(n1).to(DEVICE)
            n2 = torch.stack(n2).to(DEVICE)
            probs, _ = model(a, p, s, n1, n2, mc_T=MC_T)
            te_pred += probs.argmax(1).cpu().tolist()

    plot_confusion(test_labels, te_pred, "confusion_test.png")


if __name__ == "__main__":
    train()
