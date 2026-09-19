"""
Heterogeneous GNN for per-day (user, day) malicious classification, using
edge_attr (the aggregated mean/std/skew/kurtosis behavioral features from
src/aggregator.py) in message passing -- not just graph structure.

Architecture: 2-layer HeteroConv wrapping GINEConv per edge type. GINEConv
projects each relation's edge_attr (dims vary: 11/22/31) into hidden_dim via
its own internal edge_dim projection, then folds it into the message before
aggregation -- so e.g. an unusually high std-dev of file-copy sizes on one
edge actually influences that edge's contribution, not just its presence.

Still deliberately has NO learnable per-user-index embedding table -- only
structural signal, edge features, and static OCEAN traits. Avoids the model
memorizing "user #137 is malicious" from train and repeating it on val/test.

Requires reverse edges (apply ToUndirected before calling this model) --
otherwise 'user' never appears as a destination type and its embeddings
never update. ToUndirected duplicates edge_attr onto reverse edges too, so
edge features flow both directions.

Usage:
    from model import HeteroGNN, to_undirected, get_edge_dims
    data = to_undirected(data)
    model = HeteroGNN(data.metadata(), edge_dims=get_edge_dims(data), hidden_dim=64)
    out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GINEConv
from torch_geometric.transforms import ToUndirected

# Apply once to every graph before it reaches the model. Adds reverse edges
# (e.g. ('pc', 'rev_logs_into', 'user')) so message passing can update user
# node representations, and duplicates edge_attr onto those reverse edges.
to_undirected = ToUndirected()


def get_edge_dims(data) -> dict:
    """Extract {edge_type: edge_attr_dim} from a sample HeteroData graph.
    Call this once on any graph (post to_undirected) to build the dict the
    model constructor needs -- edge_attr dims are fixed by the feature
    engineering, so this doesn't need to be recomputed per graph."""
    return {
        et: data[et].edge_attr.shape[-1]
        for et in data.edge_types
        if "edge_attr" in data[et]
    }


class HeteroGNN(nn.Module):
    def __init__(self, metadata, edge_dims: dict, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        """
        Args:
            metadata: (node_types, edge_types) tuple from data.metadata()
                (call AFTER applying to_undirected, so reverse edges are
                included).
            edge_dims: {edge_type: int} from get_edge_dims(data) -- the
                edge_attr feature dim for each relation.
            hidden_dim: hidden channel size for all node types.
            num_layers: number of HeteroConv layers (2 is the baseline).
            dropout: dropout applied between layers.
        """
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = node_types
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        # Per-node-type input projection -- raw feature dims differ (OCEAN
        # traits for user, 1-dim placeholder for domain/pc), so project
        # everything to hidden_dim before the first conv layer. Built
        # lazily on first forward() call since we don't know raw dims here.
        self.input_proj = nn.ModuleDict()
        self._input_dims_known = False

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                edge_dim = edge_dims.get(edge_type)
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                conv_dict[edge_type] = GINEConv(mlp, edge_dim=edge_dim)
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.classifier = nn.Linear(hidden_dim, 1)  # binary logit per user node

    def _ensure_input_proj(self, x_dict):
        if self._input_dims_known:
            return
        for node_type, x in x_dict.items():
            self.input_proj[node_type] = nn.Linear(x.shape[1], self.hidden_dim).to(x.device)
        self._input_dims_known = True

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        self._ensure_input_proj(x_dict)

        h_dict = {nt: self.input_proj[nt](x) for nt, x in x_dict.items()}

        for i, conv in enumerate(self.convs):
            h_dict = conv(h_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            h_dict = {nt: F.relu(h) for nt, h in h_dict.items()}
            if i < len(self.convs) - 1:
                h_dict = {nt: F.dropout(h, p=self.dropout, training=self.training)
                          for nt, h in h_dict.items()}

        user_logits = self.classifier(h_dict["user"]).squeeze(-1)  # [num_users]
        return user_logits