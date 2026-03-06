# src/models/train_gnn.py
# ============================================
# GRAPH NEURAL NETWORK — SUPPLY CHAIN GCN
# Architecture: 3-Layer Graph Convolutional Network
# Task 1: Node-level delay probability prediction
# Task 2: Spatial bottleneck propagation modeling
# ============================================

import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, BatchNorm
from torch_geometric.utils import from_networkx
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import get_logger

logger = get_logger("GNNTrainer")

# ============================================
# PATHS
# ============================================
GRAPH_OBJECT_PATH  = "data/processed/graph_object.gpickle"
FEATURE_MATRIX_PATH = "data/processed/feature_matrix.csv"
GNN_MODEL_PATH     = "models/gnn_supply_chain.pth"
GNN_METADATA_PATH  = "models/gnn_metadata.pkl"

os.makedirs("models", exist_ok=True)


# ============================================
# GNN ARCHITECTURE
# ============================================
class SupplyChainGNN(nn.Module):
    """
    3-Layer Graph Convolutional Network for
    supply chain delay prediction.

    Architecture:
    ┌─────────────────────────────────────────┐
    │  Input: Node Features (7 features)      │
    │         [risk, capacity, betweenness,   │
    │          in_degree, out_degree,          │
    │          closeness, degree_cent]         │
    │                    │                    │
    │  Layer 1: GCNConv(7 → 32) + BatchNorm  │
    │           + ReLU + Dropout(0.3)          │
    │                    │                    │
    │  Layer 2: GCNConv(32 → 16) + BatchNorm │
    │           + ReLU + Dropout(0.2)          │
    │                    │                    │
    │  Layer 3: GCNConv(16 → 8)              │
    │           + ReLU                        │
    │                    │                    │
    │  Output Head 1: Linear(8 → 1)          │
    │    → Delay Probability (Sigmoid)        │
    │                                         │
    │  Output Head 2: Linear(8 → 1)          │
    │    → Delay Hours (ReLU — non-negative) │
    └─────────────────────────────────────────┘
    """

    def __init__(self, num_node_features: int):
        super(SupplyChainGNN, self).__init__()

        # ---- Graph Convolutional Layers ----
        self.conv1     = GCNConv(num_node_features, 32)
        self.bn1       = BatchNorm(32)

        self.conv2     = GCNConv(32, 16)
        self.bn2       = BatchNorm(16)

        self.conv3     = GCNConv(16, 8)

        # ---- Dual Output Heads ----
        # Head 1: Classification (Is this route high-risk?)
        self.head_clf  = nn.Linear(8, 1)

        # Head 2: Regression (How many hours of delay?)
        self.head_reg  = nn.Linear(8, 1)

        # ---- Dropout ----
        self.dropout1  = nn.Dropout(p=0.3)
        self.dropout2  = nn.Dropout(p=0.2)

    def forward(self, data: Data) -> tuple:
        x, edge_index = data.x, data.edge_index

        # ---- Layer 1: Looks at immediate neighbors ----
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # ---- Layer 2: Looks at 2-hop neighbors ----
        # (neighbor's neighbors — ripple effect)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # ---- Layer 3: Deep representation ----
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # ---- Output heads ----
        delay_prob  = torch.sigmoid(self.head_clf(x))   # 0.0 to 1.0
        delay_hours = F.relu(self.head_reg(x))           # Non-negative hours

        return delay_prob, delay_hours


# ============================================
# GRAPH DATA PREPARATION
# ============================================
class GNNDataPreparator:
    """
    Converts our NetworkX graph + feature matrix
    into PyTorch Geometric Data objects for training.
    """

    def __init__(self):
        self.node_scaler  = MinMaxScaler()
        self.node_mapping = {}   # location_id → integer index

    def load_graph(self) -> nx.DiGraph:
        """Loads the graph saved in Phase 4."""
        logger.info(f"Loading graph from {GRAPH_OBJECT_PATH}...")
        with open(GRAPH_OBJECT_PATH, "rb") as f:
            G = pickle.load(f)
        logger.info(
            f"Graph loaded — Nodes: {G.number_of_nodes()}, "
            f"Edges: {G.number_of_edges()}"
        )
        return G

    def load_feature_matrix(self) -> pd.DataFrame:
        """Loads the feature matrix from Phase 4."""
        logger.info(f"Loading feature matrix from {FEATURE_MATRIX_PATH}...")
        df = pd.read_csv(FEATURE_MATRIX_PATH)
        logger.info(f"Feature matrix loaded — Shape: {df.shape}")
        return df

    def prepare_node_features(
        self,
        G:  nx.DiGraph,
        df: pd.DataFrame
    ) -> tuple:
        """
        Extracts node-level features from the graph
        and aggregated statistics from the feature matrix.

        Node features (7 per node):
        [risk_score, norm_capacity, betweenness,
         degree_cent, closeness, in_degree, out_degree]

        Node targets:
        [avg_delay_hours, delay_rate]
        """
        nodes = list(G.nodes())
        self.node_mapping = {node_id: idx for idx, node_id in enumerate(nodes)}

        node_features = []
        node_targets_reg = []
        node_targets_clf = []

        for node_id in nodes:
            attrs = G.nodes[node_id]

            # ---- Node features from graph ----
            risk_score   = attrs.get("risk_score",             0.0)
            norm_cap     = attrs.get("base_capacity", 10000) / 50000.0
            betweenness  = attrs.get("betweenness_centrality",  0.0)
            degree_cent  = attrs.get("degree_centrality",       0.0)
            closeness    = attrs.get("closeness_centrality",    0.0)
            in_degree    = attrs.get("in_degree",               0.0) / 10.0
            out_degree   = attrs.get("out_degree",              0.0) / 10.0

            node_features.append([
                risk_score, norm_cap, betweenness,
                degree_cent, closeness, in_degree, out_degree
            ])

            # ---- Node targets from shipment history ----
            # Average delay for shipments departing from this node
            src_mask = df["source_location_id"] == node_id
            dst_mask = df["dest_location_id"]   == node_id

            node_data = df[src_mask | dst_mask]

            if len(node_data) > 0:
                avg_delay  = node_data["actual_delay_hours"].mean()
                delay_rate = node_data["delay_flag"].mean()
            else:
                # Fallback: derive from risk score
                avg_delay  = risk_score * 48.0 + np.random.normal(5, 2)
                delay_rate = min(risk_score + 0.1, 1.0)

            node_targets_reg.append([max(0.0, avg_delay)])
            node_targets_clf.append([float(delay_rate > 0.5)])

        # Convert to tensors
        X_nodes    = torch.tensor(node_features,    dtype=torch.float)
        y_reg      = torch.tensor(node_targets_reg, dtype=torch.float)
        y_clf      = torch.tensor(node_targets_clf, dtype=torch.float)

        logger.info(
            f"Node features prepared — "
            f"Shape: {X_nodes.shape}, "
            f"Avg risk: {X_nodes[:, 0].mean():.4f}"
        )

        return X_nodes, y_reg, y_clf

    def prepare_edge_index(self, G: nx.DiGraph) -> torch.Tensor:
        """
        Converts NetworkX edges into PyG edge_index format.
        edge_index is a [2, num_edges] tensor where:
        edge_index[0] = source node indices
        edge_index[1] = destination node indices
        """
        edge_list = []
        for src, dst in G.edges():
            if src in self.node_mapping and dst in self.node_mapping:
                edge_list.append([
                    self.node_mapping[src],
                    self.node_mapping[dst]
                ])

        if not edge_list:
            logger.warning("No edges found in graph!")
            return torch.zeros((2, 0), dtype=torch.long)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        logger.info(f"Edge index prepared — Shape: {edge_index.shape}")
        return edge_index

    def build_pyg_data(
        self,
        G:  nx.DiGraph,
        df: pd.DataFrame
    ) -> Data:
        """
        Assembles the complete PyTorch Geometric Data object.
        """
        X_nodes, y_reg, y_clf = self.prepare_node_features(G, df)
        edge_index             = self.prepare_edge_index(G)

        data = Data(
            x          = X_nodes,
            edge_index = edge_index,
            y_reg      = y_reg,
            y_clf      = y_clf
        )

        logger.info(
            f"PyG Data object created — "
            f"Nodes: {data.num_nodes}, "
            f"Edges: {data.num_edges}, "
            f"Node features: {data.num_node_features}"
        )

        return data


# ============================================
# GNN TRAINER
# ============================================
class GNNTrainer:

    def __init__(self):
        self.model     = None
        self.preparator = GNNDataPreparator()
        self.device    = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

    def train(
        self,
        data:        Data,
        epochs:      int   = 300,
        lr:          float = 0.005,
        weight_decay: float = 1e-4
    ) -> dict:
        """
        Full training loop for the GNN.
        Uses a combined loss:
        total_loss = 0.6 * BCE_loss (classification)
                   + 0.4 * MSE_loss (regression)
        """
        data  = data.to(self.device)
        num_features = data.x.shape[1]

        self.model = SupplyChainGNN(
            num_node_features=num_features
        ).to(self.device)

        optimizer     = torch.optim.Adam(
            self.model.parameters(),
            lr           = lr,
            weight_decay = weight_decay
        )
        scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        criterion_clf = nn.BCELoss()
        criterion_reg = nn.MSELoss()

        # ---- Training loop ----
        logger.info(
            f"Starting GNN training — "
            f"Epochs: {epochs}, LR: {lr}, "
            f"Device: {self.device}"
        )

        history = {
            "epoch": [], "total_loss": [],
            "clf_loss": [], "reg_loss": []
        }
        best_loss = float("inf")

        self.model.train()
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            # Forward pass
            delay_prob, delay_hours = self.model(data)

            # Compute losses
            loss_clf = criterion_clf(delay_prob,  data.y_clf)
            loss_reg = criterion_reg(delay_hours, data.y_reg)

            # Combined weighted loss
            total_loss = 0.6 * loss_clf + 0.4 * loss_reg

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()

            # ---- Logging ----
            if epoch % 30 == 0 or epoch == 1:
                logger.info(
                    f"Epoch [{epoch:03d}/{epochs}] | "
                    f"Total Loss: {total_loss.item():.6f} | "
                    f"CLF Loss: {loss_clf.item():.6f} | "
                    f"REG Loss: {loss_reg.item():.6f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )

            history["epoch"].append(epoch)
            history["total_loss"].append(total_loss.item())
            history["clf_loss"].append(loss_clf.item())
            history["reg_loss"].append(loss_reg.item())

            # Save best model
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                torch.save(
                    self.model.state_dict(),
                    GNN_MODEL_PATH
                )

        logger.info(f"Best model saved (loss={best_loss:.6f}) → {GNN_MODEL_PATH}")
        return history

    def evaluate(self, data: Data) -> dict:
        """
        Evaluates the trained GNN on the full graph.
        Returns predictions for every node.
        """
        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            delay_prob, delay_hours = self.model(data)

        delay_prob  = delay_prob.cpu().numpy().flatten()
        delay_hours = delay_hours.cpu().numpy().flatten()
        true_clf    = data.y_clf.cpu().numpy().flatten()
        true_reg    = data.y_reg.cpu().numpy().flatten()

        # ---- Metrics ----
        pred_labels = (delay_prob > 0.5).astype(int)
        f1          = f1_score(
            true_clf.astype(int),
            pred_labels,
            zero_division=0
        )
        mae         = mean_absolute_error(true_reg, delay_hours)

        logger.info("=" * 55)
        logger.info("GNN EVALUATION RESULTS")
        logger.info("=" * 55)
        logger.info(f"F1 Score (Delay Classification): {f1:.4f}")
        logger.info(f"MAE (Delay Hours):               {mae:.4f}")
        logger.info("=" * 55)

        return {
            "delay_prob":  delay_prob,
            "delay_hours": delay_hours,
            "f1_score":    f1,
            "mae":         mae
        }

    def print_node_predictions(
        self,
        G:           nx.DiGraph,
        predictions: dict,
        node_mapping: dict
    ) -> None:
        """
        Prints per-node predictions in a readable format.
        This shows the spatial propagation effect clearly.
        """
        reverse_map = {v: k for k, v in node_mapping.items()}

        logger.info("=" * 75)
        logger.info("  GNN NODE-LEVEL PREDICTIONS (SPATIAL PROPAGATION RESULTS)")
        logger.info("=" * 75)
        logger.info(
            f"{'Port':<30} {'Risk Score':>10} "
            f"{'Delay Prob':>12} {'Pred Hours':>12} {'Risk Level':>12}"
        )
        logger.info("-" * 75)

        for idx in range(len(predictions["delay_prob"])):
            node_id  = reverse_map.get(idx, f"Node_{idx}")
            attrs    = G.nodes.get(node_id, {})
            name     = attrs.get("name", node_id)[:28]
            risk     = attrs.get("risk_score", 0.0)
            d_prob   = predictions["delay_prob"][idx]
            d_hours  = predictions["delay_hours"][idx]

            risk_level = (
                "🔴 CRITICAL" if d_prob > 0.75 else
                "🟠 HIGH"     if d_prob > 0.55 else
                "🟡 MEDIUM"   if d_prob > 0.35 else
                "🟢 LOW"
            )

            logger.info(
                f"{name:<30} {risk:>10.4f} "
                f"{d_prob:>12.4f} {d_hours:>12.4f} {risk_level:>12}"
            )

        logger.info("=" * 75)

    def save_metadata(
        self,
        node_mapping: dict,
        G:            nx.DiGraph,
        history:      dict
    ) -> None:
        """Saves metadata needed for inference in Phase 8."""
        metadata = {
            "node_mapping":    node_mapping,
            "num_nodes":       G.number_of_nodes(),
            "num_edges":       G.number_of_edges(),
            "num_features":    7,
            "training_history": history,
            "node_names": {
                node_id: G.nodes[node_id].get("name", node_id)
                for node_id in G.nodes()
            }
        }
        with open(GNN_METADATA_PATH, "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"GNN metadata saved → {GNN_METADATA_PATH}")


def run_gnn_training() -> None:
    """GNN training pipeline with MLflow tracking."""
    import mlflow

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("supply_chain_delay_prediction")

    logger.info("=" * 55)
    logger.info("  GNN TRAINING PIPELINE — STARTING")
    logger.info("=" * 55)

    preparator = GNNDataPreparator()
    G          = preparator.load_graph()
    df         = preparator.load_feature_matrix()
    data       = preparator.build_pyg_data(G, df)

    with mlflow.start_run(run_name="gnn_supply_chain") as gnn_run:

        mlflow.log_params({
            "model_type":       "GraphConvolutionalNetwork",
            "architecture":     "3-Layer GCN + Dual Output Heads",
            "num_node_features": 7,
            "num_nodes":        G.number_of_nodes(),
            "num_edges":        G.number_of_edges(),
            "epochs":           300,
            "learning_rate":    0.005,
            "weight_decay":     1e-4,
            "dropout_layer1":   0.3,
            "dropout_layer2":   0.2,
            "clf_loss_weight":  0.6,
            "reg_loss_weight":  0.4,
            "optimizer":        "Adam",
            "scheduler":        "CosineAnnealingLR",
            "framework":        "PyTorch_Geometric",
            "device":           "cpu"
        })

        trainer  = GNNTrainer()
        trainer.preparator = preparator
        history  = trainer.train(data, epochs=300, lr=0.005)

        # Log training history metrics
        final_loss = history["total_loss"][-1]
        min_loss   = min(history["total_loss"])
        mlflow.log_metrics({
            "final_total_loss": round(final_loss, 6),
            "best_total_loss":  round(min_loss,   6),
            "final_clf_loss":   round(history["clf_loss"][-1],  6),
            "final_reg_loss":   round(history["reg_loss"][-1],  6)
        })

        # Log loss curve as artifact
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history["epoch"], history["total_loss"],
                label="Total Loss", color="#1f77b4", linewidth=2)
        ax.plot(history["epoch"], history["clf_loss"],
                label="CLF Loss",   color="#e74c3c", linewidth=1.5, alpha=0.7)
        ax.plot(history["epoch"], history["reg_loss"],
                label="REG Loss",   color="#2ecc71", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("GNN Training Loss Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("models/gnn_loss_curve.png", dpi=120)
        plt.close()

        mlflow.log_artifact("models/gnn_loss_curve.png")
        mlflow.log_artifact(GNN_MODEL_PATH)
        mlflow.log_artifact(GNN_METADATA_PATH)

        # Load best and evaluate
        trainer.model.load_state_dict(
            torch.load(GNN_MODEL_PATH, map_location=trainer.device)
        )
        predictions = trainer.evaluate(data)

        mlflow.log_metrics({
            "gnn_f1_score": round(float(predictions["f1_score"]), 4),
            "gnn_mae":      round(float(predictions["mae"]),      4)
        })

        trainer.print_node_predictions(
            G, predictions, preparator.node_mapping
        )
        trainer.save_metadata(
            preparator.node_mapping, G, history
        )

        logger.info(f"GNN run logged to MLflow. Run ID: {gnn_run.info.run_id}")

    logger.info("=" * 55)
    logger.info("  GNN TRAINING COMPLETE")
    logger.info("  View experiments: mlflow ui")
    logger.info("=" * 55)

if __name__ == "__main__":
    run_gnn_training()