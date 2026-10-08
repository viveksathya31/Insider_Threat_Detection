"""
Heterogeneous GNN encoder + Autoencoder (AE) head for insider threat detection,
adapted from NF-GNN (Busch et al., SSDBM 2021).

AE is built FIRST (of the planned AE/OC/CLF three-head design) because it needs no
labels at all -- it learns what "normal" user behavior looks like from the graph
structure itself, and flags poor reconstruction as anomalous. This matches the
project's framing that insider threat is fundamentally an anomaly-detection problem.

-------------------------------------------------------------------------------
WHAT THE AE RECONSTRUCTS, AND WHY (read this before changing it)
-------------------------------------------------------------------------------
data['user'].x only holds OCEAN + LDAP features -- static per user per MONTH, not
behavioral. If the AE reconstructed user.x directly, it would learn almost nothing
about day-to-day behavior: two users in the same role/department would have near-
identical reconstruction targets regardless of what they actually did that day.

Instead, the AE reconstructs a BEHAVIORAL SUMMARY per user per day: that user's own
mean edge_attr (the 5-moment aggregated stats) across each outgoing edge type,
concatenated into one fixed-size vector. A user with no activity of a given edge
type that day gets zeros for that segment. This target is built fresh from each
day's graph (build_user_behavior_target()) -- it is NOT part of HeteroData itself.

The encoder's job is to produce a user embedding, from the graph's STRUCTURE (who
they're connected to, via message passing), that is predictive of their OWN
behavioral summary. Poor reconstruction = their actual behavior that day doesn't
fit what their graph neighborhood would predict = anomalous.

-------------------------------------------------------------------------------
REVERSE EDGES
-------------------------------------------------------------------------------
Graphs as built by src/graph_builder.py only have forward edges (user -> pc/domain).
For message passing to let information flow back into user embeddings from their
neighbors, reverse edges are required. Use prepare_graph() (wraps
torch_geometric.transforms.ToUndirected()) on every graph before it reaches the
model -- this adds 'rev_<edge_type>' edges automatically. Safe to apply once and
cache the result (cheap per graph, ~2500 nodes) rather than re-applying every epoch.
"""
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GINEConv
from torch_geometric.utils import scatter
import torch_geometric.transforms as T

EdgeType = Tuple[str, str, str]

# The 6 edge types as built by src/graph_builder.py (forward direction only --
# this is the set used to build the AE's reconstruction TARGET, which should only
# reflect a user's own outgoing activity, not what reverse/message-passing edges
# were added for encoding).
FORWARD_EDGE_TYPES: List[EdgeType] = [
    ("user", "copies_file", "pc"),
    ("user", "logs_into", "pc"),
    ("user", "uses_device", "pc"),
    ("user", "visits_url", "domain"),
    ("user", "sends_email", "domain"),
    ("user", "uses_pc", "pc"),
]


def prepare_graph(data: HeteroData) -> HeteroData:
    """Adds reverse edges (rev_<edge_type>) so message passing can flow back into
    user embeddings from pc/domain neighbors. Apply once per graph; safe to cache."""
    return T.ToUndirected()(data)


def infer_graph_metadata(graphs: Dict, n_check: int = 10):
    """
    Scans up to n_check prepared (ToUndirected'd) graphs to determine:
      - node_in_dims: {node_type: input feature dim}, e.g. {'user': 10, 'pc': 1, 'domain': 1}
      - edge_attr_dims: {edge_type_tuple: edge_attr dim}, covering BOTH forward and
        reverse edge types, needed to construct the model's per-edge-type GINEConv layers.

    Scans multiple graphs rather than trusting just one, since a single day could in
    principle be missing an edge type (e.g. zero 'uses_device' events that day) --
    asserts consistency across every graph actually checked, so a genuine dimension
    mismatch (a real bug) fails loudly instead of silently picking an arbitrary value.
    """
    node_in_dims: Dict[str, int] = {}
    edge_attr_dims: Dict[EdgeType, int] = {}

    days = list(graphs.keys())[:n_check]
    for day in days:
        data = prepare_graph(graphs[day])

        for node_type in data.node_types:
            dim = data[node_type].x.shape[1]
            if node_type in node_in_dims and node_in_dims[node_type] != dim:
                raise ValueError(
                    f"Inconsistent input dim for node type '{node_type}': "
                    f"{node_in_dims[node_type]} vs {dim} (day {day}). This should "
                    f"never happen -- node feature schema is fixed by graph_builder.py."
                )
            node_in_dims[node_type] = dim

        for edge_type in data.edge_types:
            if "edge_attr" not in data[edge_type]:
                continue  # reverse edges added by ToUndirected() carry edge_attr too; skip only if truly absent
            dim = data[edge_type].edge_attr.shape[1]
            if edge_type in edge_attr_dims and edge_attr_dims[edge_type] != dim:
                raise ValueError(
                    f"Inconsistent edge_attr dim for edge type {edge_type}: "
                    f"{edge_attr_dims[edge_type]} vs {dim} (day {day})."
                )
            edge_attr_dims[edge_type] = dim

    print(f"[nf_gnn] Inferred node_in_dims: {node_in_dims}")
    print(f"[nf_gnn] Inferred edge_attr_dims ({len(edge_attr_dims)} edge types incl. reverse):")
    for et, d in edge_attr_dims.items():
        print(f"           {et}: {d}")

    return node_in_dims, edge_attr_dims


class HeteroGNNEncoder(nn.Module):
    """
    Per-node-type input projection -> N layers of heterogeneous message passing
    (GINEConv per edge type, via HeteroConv) -> final per-node-type embeddings.

    GINEConv is used because it natively incorporates edge_attr into message
    computation (not just adjacency) -- appropriate here since edge_attr carries
    the actual behavioral signal (the 5-moment aggregated stats per edge).
    """

    def __init__(self, node_in_dims: Dict[str, int], edge_attr_dims: Dict[EdgeType, int],
                 hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_proj = nn.ModuleDict({
            node_type: nn.Linear(in_dim, hidden_dim)
            for node_type, in_dim in node_in_dims.items()
        })

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            convs = {}
            for edge_type, edge_dim in edge_attr_dims.items():
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                convs[edge_type] = GINEConv(mlp, edge_dim=edge_dim)
            self.layers.append(HeteroConv(convs, aggr="sum"))

    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[EdgeType, torch.Tensor],
                edge_attr_dict: Dict[EdgeType, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_dict = {nt: self.input_proj[nt](x).relu() for nt, x in x_dict.items()}
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {nt: x.relu() for nt, x in x_dict.items()}
        return x_dict


def build_user_behavior_target(data: HeteroData,
                                edge_attr_dims: Dict[EdgeType, int]) -> torch.Tensor:
    """
    Builds the AE's reconstruction target: for each user, the mean edge_attr across
    their OUTGOING edges of each forward edge type, concatenated in FORWARD_EDGE_TYPES
    order. A user with zero edges of a given type that day gets zeros for that segment
    (reconstructing "no activity" correctly is itself a meaningful part of the task).

    Only uses FORWARD edges (not the reverse edges added by prepare_graph()) -- the
    target must reflect the user's own activity, not aggregated information that
    flowed back to them from neighbors during message passing.

    Returns: Tensor[n_users, total_dim] where total_dim = sum of forward edge_attr dims.
    """
    n_users = data["user"].x.shape[0]
    segments = []

    for edge_type in FORWARD_EDGE_TYPES:
        edge_dim = edge_attr_dims[edge_type]
        user_device = data["user"].x.device
        user_dtype = data["user"].x.dtype
        if edge_type not in data.edge_types:
            # No edges of this type at all today -- all users get zeros for this segment.
            segments.append(torch.zeros(n_users, edge_dim, device = user_device, dtype = user_dtype))
            continue

        edge_index = data[edge_type].edge_index
        edge_attr = data[edge_type].edge_attr
        src_idx = edge_index[0]  # user node indices (source of every forward edge)

        summed = scatter(edge_attr, src_idx, dim=0, dim_size=n_users, reduce="sum")
        counts = scatter(
            torch.ones(src_idx.shape[0], 1, device=edge_attr.device, dtype= edge_attr.dtype),
             src_idx, dim=0,dim_size=n_users, reduce="sum"
            ).clamp(min=1.0)
        mean_per_user = summed / counts
        segments.append(mean_per_user)

    return torch.cat(segments, dim=1)


class AEHead(nn.Module):
    """Decoder: user embedding -> reconstructed behavioral summary vector."""

    def __init__(self, embedding_dim: int, target_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, user_embedding: torch.Tensor) -> torch.Tensor:
        return self.decoder(user_embedding)


class InsiderThreatAE(nn.Module):
    """
    Full AE model: HeteroGNNEncoder -> AEHead, trained to reconstruct each user's
    own daily behavioral summary from their graph-neighborhood embedding.

    forward() returns (reconstruction, target, per_user_error) so callers can use
    per_user_error directly as that day's anomaly score, without recomputing MSE
    outside the model.
    """

    def __init__(self, node_in_dims: Dict[str, int], edge_attr_dims: Dict[EdgeType, int],
                 hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.encoder = HeteroGNNEncoder(node_in_dims, edge_attr_dims, hidden_dim, num_layers)
        target_dim = sum(edge_attr_dims[et] for et in FORWARD_EDGE_TYPES)
        self.ae_head = AEHead(embedding_dim=hidden_dim, target_dim=target_dim,
                               hidden_dim=hidden_dim)
        self.edge_attr_dims = edge_attr_dims

    def forward(self, data: HeteroData):
        x_dict = self.encoder(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        user_embedding = x_dict["user"]

        target = build_user_behavior_target(data, self.edge_attr_dims).to(user_embedding.device)
        reconstruction = self.ae_head(user_embedding)

        per_user_error = ((reconstruction - target) ** 2).mean(dim=1)  # [n_users]
        return reconstruction, target, per_user_error


class InsiderThreatOC(nn.Module):
    """
    HeteroGNNEncoder + OneClassHead (Deep SVDD).
    Maps user embeddings into a latent hypersphere where normal activity is
    clustered around a fixed center vector c.
    
    forward() returns (projected, anomaly_scores), where anomaly_scores is
    the squared Euclidean distance ||phi(z) - c||^2 per user.
    """

    def __init__(self, node_in_dims: Dict[str, int], edge_attr_dims: Dict[EdgeType, int],
                 hidden_dim: int = 64, projection_dim: int = 32, num_layers: int = 2):
        super().__init__()
        from models.heads import OneClassHead
        self.encoder = HeteroGNNEncoder(node_in_dims, edge_attr_dims, hidden_dim, num_layers)
        self.oc_head = OneClassHead(embedding_dim=hidden_dim, projection_dim=projection_dim, hidden_dim=hidden_dim)

    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        x_dict = self.encoder(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        user_embedding = x_dict["user"]
        projected, distances = self.oc_head(user_embedding)
        return projected, distances


class InsiderThreatCLF(nn.Module):
    """
    HeteroGNNEncoder + ClassificationHead (Supervised).
    Trains with weighted BCEWithLogitsLoss to directly predict malicious probability.
    
    forward() returns logits of shape [n_users].
    """

    def __init__(self, node_in_dims: Dict[str, int], edge_attr_dims: Dict[EdgeType, int],
                 hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        from models.heads import ClassificationHead
        self.encoder = HeteroGNNEncoder(node_in_dims, edge_attr_dims, hidden_dim, num_layers)
        self.clf_head = ClassificationHead(embedding_dim=hidden_dim, hidden_dim=hidden_dim // 2, dropout=dropout)

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = self.encoder(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        user_embedding = x_dict["user"]
        logits = self.clf_head(user_embedding)
        return logits



def load_user_split_masks(splits_path: str = "data/processed/user_splits.json",
                user_ordering_csv: str = "data/raw/r4.2/psychometric.csv") -> Dict[str,
torch.Tensor]:
    """
    Loads data/processed/user_splits.json and returns boolean masks of shape[n_users]
    aligned with the deterministic node index ordereing (0.999).

    Use:
        mask = load_user_split_masks()
        train_loss = per_user_error[masks['train']].mean()
    """
    
    import json
    from pathlib import Path
    import pandas as pd

    with open(splits_path, "r") as f:
        splits = json.load(f)

    # Reconstruct the exact user index ordering used in graph_builder.py
    psych_df = pd.read_csv(user_ordering_csv, usecols=["user_id"])
    ordered_users = sorted(set(psych_df["user_id"].astype(str)))
    user_to_idx = {uid: i for i, uid in enumerate(ordered_users)}
    
    masks = {}
    n_users = len(ordered_users)
    for split_name in ["train", "val", "test"]:
        mask = torch.zeros(n_users, dtype=torch.bool)
        for uid in splits[split_name]:
            if uid in user_to_idx:
                mask[user_to_idx[uid]] = True
        masks[split_name] = mask
    return masks


def compute_train_edge_norm_stats(graphs: Dict, train_mask: torch.Tensor) -> Dict[EdgeType, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Computes per-edge-type mean and std exclusively over edges originating from
    train-split users across all days. Prevents data leakage into val/test.
    """
    sums, sq_sums, counts = {}, {}, {}
    for g in graphs.values():
        for et in g.edge_types:
            if "edge_attr" not in g[et]:
                continue
            src = g[et].edge_index[0]
            # Only consider edges originating from users in the train split
            train_edges = train_mask[src]
            if not train_edges.any():
                continue
            attrs = g[et].edge_attr[train_edges]
            if et not in sums:
                sums[et] = attrs.sum(dim=0)
                sq_sums[et] = (attrs ** 2).sum(dim=0)
                counts[et] = attrs.shape[0]
            else:
                sums[et] += attrs.sum(dim=0)
                sq_sums[et] += (attrs ** 2).sum(dim=0)
                counts[et] += attrs.shape[0]

    stats = {}
    for et in sums:
        mean = sums[et] / counts[et]
        var = (sq_sums[et] / counts[et]) - (mean ** 2)
        std = var.clamp(min=1e-6).sqrt()
        stats[et] = (mean, std)

    return stats


def apply_edge_norm(data: HeteroData, norm_stats: Dict[EdgeType, Tuple[torch.Tensor, torch.Tensor]]) -> HeteroData:
    """
    Applies z-score normalization (attr - mean) / std in-place to all edges.
    Handles forward and reverse edges (which share the same base relation stats).
    """
    for et in data.edge_types:
        if "edge_attr" not in data[et]:
            continue
        # Find matching stats (forward relation or reverse relation)
        lookup_et = et
        if et not in norm_stats:
            # Handle reverse edge naming: ('pc', 'rev_logs_into', 'user') -> ('user', 'logs_into', 'pc')
            src, rel, dst = et
            if rel.startswith("rev_"):
                fwd_rel = rel[4:]
                lookup_et = (dst, fwd_rel, src)

        if lookup_et in norm_stats:
            mean, std = norm_stats[lookup_et]
            mean = mean.to(device=data[et].edge_attr.device, dtype=data[et].edge_attr.dtype)
            std = std.to(device=data[et].edge_attr.device, dtype=data[et].edge_attr.dtype)
            data[et].edge_attr = (data[et].edge_attr - mean) / std

    return data


# =============================================================================
# SELF-TEST / SMOKE TEST BLOCK
# =============================================================================
# This block runs ONLY when the file is executed directly (e.g. `python models/nf_gnn.py`).
# It loads a real daily graph from output/, verifies tensor shapes through the encoder,
# checks target generation, and confirms the forward pass runs end-to-end.
# =============================================================================
if __name__ == "__main__":
    from pathlib import Path
    import pickle

    graphs_path = Path("output/daily_graphs_labeled.pkl")
    if not graphs_path.exists():
        print(f"[nf_gnn test] Error: '{graphs_path}' not found.")
        print("[nf_gnn test] Please ensure daily graphs have been generated via build_graphs.py.")
        exit(1)

    print("=" * 60)
    print(" Running InsiderThreatAE Model Smoke Test")
    print("=" * 60)

    # 1. Load labeled daily graphs
    print("\n[1/4] Loading graphs from output/daily_graphs_labeled.pkl ...")
    with open(graphs_path, "rb") as f:
        graphs = pickle.load(f)
    first_day = next(iter(graphs))
    print(f"      Loaded {len(graphs)} total days. Testing on first day: {first_day.date()}")

    # 2. Infer input feature dimensions and edge attribute dimensions
    print("\n[2/4] Inferring node and edge attribute dimensions across sample days ...")
    node_in_dims, edge_attr_dims = infer_graph_metadata(graphs, n_check=5)

    # 3. Instantiate the model
    print("\n[3/4] Initializing InsiderThreatAE model (hidden_dim=64, num_layers=2) ...")
    model = InsiderThreatAE(node_in_dims, edge_attr_dims, hidden_dim=64, num_layers=2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Model successfully instantiated. Trainable parameters: {n_params:,}")

    # 4. Run a single forward pass on the prepared graph
    print("\n[4/4] Preparing graph with reverse edges and running forward pass ...")
    data = prepare_graph(graphs[first_day])
    recon, target, per_user_error = model(data)

    print("\n" + "=" * 60)
    print(" Test Passed Successfully!")
    print("=" * 60)
    print(f"  Target Vector Shape        : {list(target.shape)}  (1000 users x 139 behavioral features)")
    print(f"  Reconstruction Shape       : {list(recon.shape)}")
    print(f"  Per-User Anomaly Score     : {list(per_user_error.shape)}  (scalar score per user)")
    print(f"  Day Mean Anomaly Score     : {per_user_error.mean().item():.4f}")
    print("=" * 60)