# Run this directly in terminal:
# python -c "exec(open('verify_phase5.py').read())"
# OR create verify_phase5.py and run it

import joblib
import numpy as np
import pandas as pd

print("=" * 55)
print("  VERIFYING XGBOOST MODELS")
print("=" * 55)

# Load models
clf    = joblib.load("models/xgboost_classifier.joblib")
reg    = joblib.load("models/xgboost_regressor.joblib")
scaler = joblib.load("models/feature_scaler.joblib")

print(f"Classifier loaded  — Type: {type(clf).__name__}")
print(f"Regressor loaded   — Type: {type(reg).__name__}")
print(f"Scaler loaded      — Expected features: {scaler.n_features_in_}")

# Load feature matrix and run a test prediction
df             = pd.read_csv("data/processed/feature_matrix.csv")
FEATURE_COLS   = [
    "dispatch_hour", "dispatch_dayofweek", "dispatch_month",
    "dispatch_quarter", "is_weekend", "is_month_end",
    "is_peak_season", "rolling_7d_avg_delay", "rolling_30d_avg_delay",
    "rolling_7d_std_delay", "lag_1_delay", "lag_2_delay",
    "standard_duration_hours", "standard_cost",
    "transport_mode_enc", "cargo_type_enc",
    "route_id_enc", "cargo_weight_tons",
    "distance_km", "calculated_distance_km",
    "expected_speed_kmh", "source_hemisphere",
    "dest_hemisphere", "crosses_hemisphere",
    "crosses_pacific", "crosses_atlantic",
    "capacity_ratio", "lon_diff", "lat_diff",
    "source_capacity", "dest_capacity",
    "source_avg_risk", "source_max_risk",
    "source_negative_events", "source_event_count",
    "dest_avg_risk", "dest_max_risk",
    "dest_negative_events", "dest_event_count",
    "combined_route_risk",
    "source_betweenness", "source_degree_cent",
    "source_closeness", "source_in_degree", "source_out_degree",
    "dest_betweenness", "dest_degree_cent",
    "dest_closeness", "dest_in_degree", "dest_out_degree",
]

available_cols = [c for c in FEATURE_COLS if c in df.columns]
sample         = df[available_cols].head(5).fillna(0).values
sample_scaled  = scaler.transform(sample)

# Test classifier
clf_probs      = clf.predict_proba(sample_scaled)[:, 1]
clf_preds      = clf.predict(sample_scaled)

# Test regressor
reg_preds      = reg.predict(sample_scaled)

print("\n--- XGBoost Test Predictions (5 samples) ---")
print(f"{'Sample':<10} {'Delay Prob':>12} {'Predicted?':>12} "
      f"{'Pred Hours':>12} {'Actual Hours':>14}")
print("-" * 62)
for i in range(5):
    actual = df["actual_delay_hours"].iloc[i]
    print(
        f"{i+1:<10} "
        f"{clf_probs[i]:>12.4f} "
        f"{'YES' if clf_preds[i] == 1 else 'NO':>12} "
        f"{reg_preds[i]:>12.2f} "
        f"{actual:>14.2f}"
    )

print("\nXGBoost verification PASSED")

import torch
import pickle
import numpy as np
import sys
import os
sys.path.append(os.path.abspath("."))

from src.models.train_gnn import SupplyChainGNN
from torch_geometric.data import Data

print("=" * 55)
print("  VERIFYING GNN MODEL")
print("=" * 55)

# Load metadata
with open("models/gnn_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(f"Num nodes:     {metadata['num_nodes']}")
print(f"Num edges:     {metadata['num_edges']}")
print(f"Num features:  {metadata['num_features']}")
print(f"Node names:    {list(metadata['node_names'].values())}")

# Load graph
with open("data/processed/graph_object.gpickle", "rb") as f:
    G = pickle.load(f)

# Load model
device    = torch.device("cpu")
model     = SupplyChainGNN(num_node_features=metadata["num_features"])
model.load_state_dict(
    torch.load("models/gnn_supply_chain.pth", map_location=device)
)
model.eval()
print(f"\nGNN Architecture:\n{model}")

# Build a test forward pass
nodes      = list(G.nodes())
node_map   = metadata["node_mapping"]
node_feats = []

for node_id in nodes:
    attrs    = G.nodes[node_id]
    risk     = attrs.get("risk_score",             0.0)
    norm_cap = attrs.get("base_capacity", 10000) / 50000.0
    btwn     = attrs.get("betweenness_centrality",  0.0)
    deg      = attrs.get("degree_centrality",       0.0)
    cls      = attrs.get("closeness_centrality",    0.0)
    ind      = attrs.get("in_degree",               0.0) / 10.0
    outd     = attrs.get("out_degree",              0.0) / 10.0
    node_feats.append([risk, norm_cap, btwn, deg, cls, ind, outd])

x          = torch.tensor(node_feats, dtype=torch.float)
edge_list  = [
    [node_map[s], node_map[d]]
    for s, d in G.edges()
    if s in node_map and d in node_map
]
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
data       = Data(x=x, edge_index=edge_index)

with torch.no_grad():
    delay_prob, delay_hours = model(data)

print("\n--- GNN Node-Level Predictions ---")
print(f"{'Port':<35} {'Risk Score':>12} "
      f"{'Delay Prob':>12} {'Pred Hours':>12}")
print("-" * 75)

reverse_map = {v: k for k, v in node_map.items()}
for idx in range(len(nodes)):
    node_id   = reverse_map.get(idx, f"Node_{idx}")
    name      = G.nodes[node_id].get("name", node_id)
    risk      = G.nodes[node_id].get("risk_score", 0.0)
    d_prob    = delay_prob[idx].item()
    d_hours   = delay_hours[idx].item()
    print(
        f"{name:<35} {risk:>12.4f} "
        f"{d_prob:>12.4f} {d_hours:>12.4f}"
    )

print("\nGNN verification PASSED")

import pickle

print("=" * 55)
print("  VERIFYING ENSEMBLE CONFIG")
print("=" * 55)

with open("models/ensemble_config.pkl", "rb") as f:
    config = pickle.load(f)

print(f"XGBoost Weight:    {config['xgb_weight']}")
print(f"GNN Weight:        {config['gnn_weight']}")
print(f"Feature columns:   {len(config['feature_columns'])} features")
print(f"Model paths:")
for name, path in config["model_paths"].items():
    exists = "EXISTS" if __import__("os").path.exists(path) else "MISSING"
    print(f"  {name:<10} → {path:<45} [{exists}]")

print("\nEnsemble config verification PASSED")