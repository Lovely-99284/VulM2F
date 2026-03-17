
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchbnn as bnn

class ParallelProjectionBlock(nn.Module):
    """
    Small / Large Kernel Parallel Projection
    """

    def __init__(self, in_dim, hidden_dim=64, dropout=0.2):
        super().__init__()

        self.small_path = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.large_path = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.small_path(x) + self.large_path(x)


class BayesianSemanticBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        semantic_dim=32,
        dropout=0.3,
        prior_mu=0.0,
        prior_sigma=0.1
    ):
        super().__init__()

        self.net = nn.Sequential(
            bnn.BayesLinear(
                prior_mu=prior_mu,
                prior_sigma=prior_sigma,
                in_features=in_dim,
                out_features=semantic_dim
            ),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class BayesianBranch(nn.Module):
    def __init__(
        self,
        in_dim,
        bottleneck_dim=64,
        semantic_dim=32,
        dropout=0.3,
        prior_mu=0.0,
        prior_sigma=0.1
    ):
        super().__init__()

        self.projection = ParallelProjectionBlock(
            in_dim,
            hidden_dim=bottleneck_dim,
            dropout=dropout * 0.5
        )

        self.semantic = BayesianSemanticBlock(
            bottleneck_dim,
            semantic_dim,
            dropout=dropout,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma
        )

    def forward(self, x):
        return self.semantic(self.projection(x))



class BranchAttention(nn.Module):
    def __init__(self, n_branches, semantic_dim):
        super().__init__()
        self.score = nn.Linear(semantic_dim, 1, bias=False)

    def forward(self, feats):
        stack = torch.stack(feats, dim=1)  # (B, N, D)
        B, N, D = stack.size()
        scores = self.score(stack.view(B * N, D)).view(B, N)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        fused = (stack * weights).sum(dim=1)
        return fused, weights.squeeze(-1)



class FusionMLP(nn.Module):

    def __init__(
        self,
        ast_dim,
        pdg_dim,
        struct_dim,
        node1_dim,
        node2_dim,
        bottleneck_dim=64,
        semantic_dim=32,
        hidden_dim=32,
        num_classes=2,
        dropout=0.3,
        prior_mu=0.0,
        prior_sigma=0.1
    ):
        super().__init__()


        self.ast_branch = BayesianBranch(ast_dim, bottleneck_dim, semantic_dim, dropout, prior_mu, prior_sigma)
        self.pdg_branch = BayesianBranch(pdg_dim, bottleneck_dim, semantic_dim, dropout, prior_mu, prior_sigma)
        self.struct_branch = BayesianBranch(struct_dim, bottleneck_dim, semantic_dim, dropout, prior_mu, prior_sigma)
        self.node1_branch = BayesianBranch(node1_dim, bottleneck_dim, semantic_dim, dropout, prior_mu, prior_sigma)
        self.node2_branch = BayesianBranch(node2_dim, bottleneck_dim, semantic_dim, dropout, prior_mu, prior_sigma)


        self.attention = BranchAttention(n_branches=5, semantic_dim=semantic_dim)


        self.bayes_bottleneck = bnn.BayesLinear(
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            in_features=semantic_dim * 5,
            out_features=hidden_dim
        )

        self.bayes_dropout = nn.Dropout(p=0.5)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5),
            bnn.BayesLinear(
                prior_mu=prior_mu,
                prior_sigma=prior_sigma,
                in_features=hidden_dim,
                out_features=num_classes
            )
        )

    # -----------------------------------------------------------
    def forward_once(self, ast, pdg, struct, n1, n2):
        feats = [
            self.ast_branch(ast),
            self.pdg_branch(pdg),
            self.struct_branch(struct),
            self.node1_branch(n1),
            self.node2_branch(n2),
        ]


        fused = torch.cat(feats, dim=1)
        weights = None

              z = F.relu(self.bayes_bottleneck(fused))
        z = self.bayes_dropout(z)

        logits = self.classifier(z)
        return logits, weights

    # -----------------------------------------------------------
    def forward(self, ast, pdg, struct, n1, n2, mc_T=1):
        if mc_T > 1:
            probs = []
            for _ in range(mc_T):
                logits, _ = self.forward_once(ast, pdg, struct, n1, n2)
                probs.append(F.softmax(logits, dim=1))
            return torch.stack(probs).mean(dim=0), None
        else:
            return self.forward_once(ast, pdg, struct, n1, n2)
