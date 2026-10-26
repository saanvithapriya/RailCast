"""Graph Neural Network experiment for downstream delay prediction using PyTorch Geometric.

Architectural design:
- Network topology: Corridor B station network (108 station nodes, 214 directed section edges).
- Station static features: junction flag, distance along chain, historical training delay.
- Spatio-temporal message passing: 2-layer GCNConv aggregates connectivity and network topology
  along the line into learned station representations.
- Fusion regressor: Concatenates the node embedding for the train's current station with the
  train's dynamic observation features (current delay, elapsed time, weather, priority)
  into an MLP prediction head.
- Leakage safety: Station graph structure and node priors are strictly timetable-derived or
  fitted on training-window delay tables.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.graph.station_graph import build_station_tables
from src.graph.timetable import load_timetable

DEVICE = torch.device("cpu")


def load_station_graph_tensor(
    train_delay_table: Optional[pd.DataFrame] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load Corridor B station graph edge_index and static node feature matrix."""
    tt = load_timetable()
    nodes_df, edges_df, _ = build_station_tables(tt)

    # 1. Edge index (2, 214)
    u = edges_df["from_node"].to_numpy(dtype=np.int64)
    v = edges_df["to_node"].to_numpy(dtype=np.int64)
    edge_index = torch.tensor(np.stack([u, v], axis=0), dtype=torch.long)

    # 2. Node features (108, d_node)
    # is_junction, km_from_new_delhi / 1000.0, trains_per_day / 200.0, hist_mean_delay / 60.0
    n_nodes = len(nodes_df)
    is_junc = nodes_df["is_junction"].astype(float).to_numpy()
    km = (nodes_df["km_from_new_delhi"].astype(float) / 1000.0).to_numpy()
    tpd = (nodes_df["trains_per_day"].astype(float) / 200.0).to_numpy()

    if train_delay_table is not None and "mean_arrival_delay" in train_delay_table.columns:
        hist_delay = (train_delay_table["mean_arrival_delay"].astype(float) / 60.0).to_numpy()
    else:
        hist_delay = np.zeros(n_nodes, dtype=float)

    feat_matrix = np.stack([is_junc, km, tpd, hist_delay], axis=1).astype(np.float32)
    node_features = torch.tensor(feat_matrix, dtype=torch.float32)

    return edge_index, node_features


class StationGNNModule(nn.Module):
    """Spatio-Temporal Graph Neural Network for delay prediction."""

    def __init__(
        self,
        node_feat_dim: int = 4,
        node_emb_dim: int = 16,
        train_feat_dim: int = 26,
        hidden_dim: int = 64,
    ):
        super().__init__()
        # GNN backbone for station topology
        self.conv1 = GCNConv(node_feat_dim, node_emb_dim)
        self.conv2 = GCNConv(node_emb_dim, node_emb_dim)

        # Regressor head combining train state + station embedding
        total_dim = train_feat_dim + node_emb_dim
        self.fc1 = nn.Linear(total_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor,
        train_features: torch.Tensor,
        node_indices: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Message passing over station network
        h_station = F.relu(self.conv1(node_features, edge_index))
        h_station = self.conv2(h_station, edge_index)  # (108, node_emb_dim)

        # 2. Gather station embedding for each train's location
        train_node_emb = h_station[node_indices]  # (B, node_emb_dim)

        # 3. Fuse train state with station graph embedding
        fused = torch.cat([train_features, train_node_emb], dim=-1)  # (B, total_dim)

        # 4. MLP prediction head
        z = F.relu(self.fc1(fused))
        z = F.relu(self.fc2(z))
        return self.out(z).squeeze(-1)


class GNNRegressor:
    """Scikit-learn compatible wrapper for the StationGNNModule."""

    def __init__(
        self,
        horizon: int = 1,
        epochs: int = 15,
        lr: float = 0.003,
        batch_size: int = 1024,
        random_state: int = 42,
    ):
        self.horizon = horizon
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.model: Optional[StationGNNModule] = None
        self.edge_index: Optional[torch.Tensor] = None
        self.node_features: Optional[torch.Tensor] = None
        self.node_col_idx: int = 3  # index of current_node in FEATURE_COLUMNS

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GNNRegressor":
        """Train GNN regressor on features X and targets y."""
        torch.manual_seed(self.random_state + self.horizon)
        np.random.seed(self.random_state + self.horizon)

        self.edge_index, self.node_features = load_station_graph_tensor()

        n_samples, n_features = X.shape
        node_indices_np = np.clip(X[:, self.node_col_idx].astype(np.int64), 0, 107)

        self.model = StationGNNModule(
            node_feat_dim=self.node_features.shape[1],
            node_emb_dim=16,
            train_feat_dim=n_features,
            hidden_dim=64,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        nodes_t = torch.tensor(node_indices_np, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X_t, nodes_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_nodes, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(self.edge_index, self.node_features, batch_x, batch_nodes)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict downstream delay using trained GNN."""
        if self.model is None or self.edge_index is None or self.node_features is None:
            raise RuntimeError("GNNRegressor must be fitted before calling predict().")

        self.model.eval()
        node_indices_np = np.clip(X[:, self.node_col_idx].astype(np.int64), 0, 107)

        X_t = torch.tensor(X, dtype=torch.float32)
        nodes_t = torch.tensor(node_indices_np, dtype=torch.long)

        with torch.no_grad():
            preds = self.model(self.edge_index, self.node_features, X_t, nodes_t)
            return preds.cpu().numpy().astype(np.float32)
