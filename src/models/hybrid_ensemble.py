# src/models/hybrid_ensemble.py
# ============================================
# HYBRID ENSEMBLE MODEL
# Combines XGBoost + GNN predictions
# + Route Recommendation Engine
# ============================================

import os
import sys
import pickle
import joblib
import torch
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.train_gnn import SupplyChainGNN
from src.utils import get_logger

logger = get_logger("HybridEnsemble")

# ============================================
# PATHS
# ============================================
XGBOOST_CLF_PATH  = "models/xgboost_classifier.joblib"
XGBOOST_REG_PATH  = "models/xgboost_regressor.joblib"
SCALER_PATH       = "models/feature_scaler.joblib"
GNN_MODEL_PATH    = "models/gnn_supply_chain.pth"
GNN_METADATA_PATH = "models/gnn_metadata.pkl"
GRAPH_OBJECT_PATH = "data/processed/graph_object.gpickle"
ENSEMBLE_PATH     = "models/ensemble_config.pkl"

# Ensemble weights (tuned so GNN + XGBoost complement each other)
XGB_WEIGHT        = 0.55
GNN_WEIGHT        = 0.45


class HybridEnsemble:
    """
    Production-ready ensemble that combines:
    - XGBoost (tabular features)
    - GNN (spatial graph features)
    - NLP risk score booster

    For inference, given a route_id, it returns:
    {
        'delay_probability': 0.73,
        'predicted_delay_hours': 12.4,
        'risk_level': 'HIGH',
        'confidence': 0.81,
        'recommended_alternatives': [...]
    }
    """

    def __init__(self):
        self.xgb_clf      = None
        self.xgb_reg      = None
        self.scaler       = None
        self.gnn_model    = None
        self.gnn_metadata = None
        self.graph        = None
        self.device       = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._load_all_models()

    # ============================================
    # MODEL LOADING
    # ============================================
    def _load_all_models(self) -> None:
        """Loads all saved model artifacts."""
        logger.info("Loading all model artifacts...")

        try:
            self.xgb_clf = joblib.load(XGBOOST_CLF_PATH)
            self.xgb_reg = joblib.load(XGBOOST_REG_PATH)
            self.scaler  = joblib.load(SCALER_PATH)
            logger.info("XGBoost models loaded.")
        except FileNotFoundError as e:
            logger.error(f"XGBoost models not found: {e}. "
                         f"Run train_xgboost.py first.")
            raise

        try:
            with open(GNN_METADATA_PATH, "rb") as f:
                self.gnn_metadata = pickle.load(f)

            num_features  = self.gnn_metadata.get("num_features", 7)
            self.gnn_model = SupplyChainGNN(
                num_node_features=num_features
            ).to(self.device)
            self.gnn_model.load_state_dict(
                torch.load(GNN_MODEL_PATH, map_location=self.device)
            )
            self.gnn_model.eval()
            logger.info("GNN model loaded.")
        except FileNotFoundError as e:
            logger.error(f"GNN model not found: {e}. "
                         f"Run train_gnn.py first.")
            raise

        try:
            with open(GRAPH_OBJECT_PATH, "rb") as f:
                self.graph = pickle.load(f)
            logger.info(
                f"Graph loaded — "
                f"Nodes: {self.graph.number_of_nodes()}, "
                f"Edges: {self.graph.number_of_edges()}"
            )
        except FileNotFoundError as e:
            logger.error(f"Graph object not found: {e}.")
            raise

        logger.info("All models loaded successfully.")

    # ============================================
    # HYBRID PREDICTION
    # ============================================
    def predict(
        self,
        feature_row:      np.ndarray,
        source_location:  str,
        dest_location:    str,
        nlp_risk_score:   float = 0.0
    ) -> Dict:
        """
        Makes a hybrid prediction for a single shipment.

        Args:
            feature_row:     1D numpy array of tabular features
            source_location: location_id of source port
            dest_location:   location_id of destination port
            nlp_risk_score:  current NLP risk score for the route

        Returns:
            Dictionary with prediction results
        """
        # ---- XGBoost prediction ----
        X_scaled        = self.scaler.transform(
            feature_row.reshape(1, -1)
        )
        xgb_delay_prob  = float(
            self.xgb_clf.predict_proba(X_scaled)[0][1]
        )
        xgb_delay_hours = float(
            self.xgb_reg.predict(X_scaled)[0]
        )

        # ---- GNN prediction ----
        gnn_delay_prob, gnn_delay_hours = self._get_gnn_prediction(
            source_location, dest_location
        )

        # ---- Ensemble combination ----
        ensemble_prob  = (
            XGB_WEIGHT * xgb_delay_prob +
            GNN_WEIGHT * gnn_delay_prob
        )
        ensemble_hours = (
            XGB_WEIGHT * xgb_delay_hours +
            GNN_WEIGHT * gnn_delay_hours
        )

        # ---- NLP risk boost ----
        # If current NLP risk is very high, boost the probability
        if nlp_risk_score > 0.7:
            nlp_boost     = (nlp_risk_score - 0.7) * 0.3
            ensemble_prob = min(ensemble_prob + nlp_boost, 1.0)

        # ---- Confidence score ----
        # How much do XGBoost and GNN agree?
        prob_diff  = abs(xgb_delay_prob - gnn_delay_prob)
        confidence = round(1.0 - prob_diff, 4)

        # ---- Risk level ----
        risk_level = self._get_risk_level(ensemble_prob)

        result = {
            "delay_probability":   round(ensemble_prob, 4),
            "predicted_delay_hours": round(max(0, ensemble_hours), 2),
            "risk_level":          risk_level,
            "confidence":          confidence,
            "xgb_delay_prob":      round(xgb_delay_prob, 4),
            "gnn_delay_prob":      round(gnn_delay_prob, 4),
            "nlp_risk_score":      round(nlp_risk_score, 4)
        }

        return result

    def _get_gnn_prediction(
        self,
        source_location: str,
        dest_location:   str
    ) -> tuple:
        """
        Gets GNN predictions for source and destination nodes.
        Returns average of the two node predictions.
        """
        node_mapping = self.gnn_metadata.get("node_mapping", {})

        # Build PyG data from graph
        from torch_geometric.data import Data
        nodes      = list(self.graph.nodes())
        node_feats = []

        for node_id in nodes:
            attrs    = self.graph.nodes[node_id]
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
            [node_mapping[s], node_mapping[d]]
            for s, d in self.graph.edges()
            if s in node_mapping and d in node_mapping
        ]

        if edge_list:
            edge_index = torch.tensor(
                edge_list, dtype=torch.long
            ).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index).to(self.device)

        with torch.no_grad():
            delay_prob, delay_hours = self.gnn_model(data)

        delay_prob  = delay_prob.cpu().numpy().flatten()
        delay_hours = delay_hours.cpu().numpy().flatten()

        # Get predictions for source and destination nodes
        src_idx     = node_mapping.get(source_location, 0)
        dst_idx     = node_mapping.get(dest_location,   0)

        avg_prob    = float(
            (delay_prob[src_idx] + delay_prob[dst_idx]) / 2
        )
        avg_hours   = float(
            (delay_hours[src_idx] + delay_hours[dst_idx]) / 2
        )

        return avg_prob, avg_hours

    def _get_risk_level(self, probability: float) -> str:
        """Converts probability to business risk label."""
        if probability >= 0.75:
            return "CRITICAL"
        elif probability >= 0.55:
            return "HIGH"
        elif probability >= 0.35:
            return "MEDIUM"
        elif probability >= 0.15:
            return "LOW"
        else:
            return "MINIMAL"


# ============================================
# ROUTE RECOMMENDATION ENGINE
# ============================================
class RouteRecommendationEngine:
    """
    Given a high-risk route, finds the K safest
    alternative paths through the supply chain graph.

    Uses Dijkstra's algorithm with risk-adjusted weights.
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def get_alternative_routes(
        self,
        source:    str,
        target:    str,
        k:         int = 3,
        avoid_nodes: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Finds up to K alternative routes from source to target.

        Args:
            source:       source location_id
            target:       destination location_id
            k:            number of alternatives to return
            avoid_nodes:  list of high-risk nodes to avoid

        Returns:
            List of route alternatives sorted by safety score
        """
        logger.info(
            f"Finding {k} alternative routes: "
            f"{source} → {target}"
        )

        # Build a risk-adjusted copy of the graph
        G_risk = self._build_risk_adjusted_graph(avoid_nodes)

        alternatives = []

        try:
            # Use simple paths algorithm (finds multiple paths)
            all_paths = list(nx.simple_paths.all_simple_paths(
                G_risk,
                source = source,
                target = target,
                cutoff = 5           # Max 5 hops
            ))

            if not all_paths:
                logger.warning(
                    f"No paths found from {source} to {target}"
                )
                return []

            # Score each path
            for path in all_paths[:20]:    # Evaluate top 20 candidates
                route_info = self._score_path(path, G_risk)
                alternatives.append(route_info)

            # Sort by risk score (lowest = safest)
            alternatives = sorted(
                alternatives,
                key=lambda x: x["total_risk_score"]
            )[:k]

            # Add rank
            for i, alt in enumerate(alternatives):
                alt["rank"] = i + 1

            logger.info(f"Found {len(alternatives)} alternative routes.")

        except nx.NetworkXNoPath:
            logger.warning(f"No path exists between {source} and {target}")
        except nx.NodeNotFound as e:
            logger.error(f"Node not found in graph: {e}")

        return alternatives

    def _build_risk_adjusted_graph(
        self,
        avoid_nodes: Optional[List[str]] = None
    ) -> nx.DiGraph:
        """
        Creates a copy of the graph where edge weights
        are adjusted by the risk scores of endpoint nodes.
        Higher risk = higher weight = avoided by Dijkstra.
        """
        G_copy = self.graph.copy()

        # Remove nodes we explicitly want to avoid
        if avoid_nodes:
            for node in avoid_nodes:
                if node in G_copy:
                    G_copy.remove_node(node)
                    logger.info(f"Avoiding high-risk node: {node}")

        # Adjust edge weights by node risk scores
        for u, v, data in G_copy.edges(data=True):
            src_risk   = G_copy.nodes[u].get("risk_score", 0.0)
            dst_risk   = G_copy.nodes[v].get("risk_score", 0.0)
            base_weight = data.get("duration_hours", 100.0)

            # Risk-adjusted weight:
            # high risk routes cost proportionally more
            risk_penalty       = 1.0 + (src_risk + dst_risk)
            data["risk_weight"] = base_weight * risk_penalty

        return G_copy

    def _score_path(
        self,
        path:    List[str],
        G:       nx.DiGraph
    ) -> Dict:
        """
        Scores a path on risk, duration, and cost.
        """
        total_risk     = 0.0
        total_duration = 0.0
        total_cost     = 0.0
        hops           = len(path) - 1
        path_details   = []

        for i in range(len(path) - 1):
            src  = path[i]
            dst  = path[i + 1]
            edge = G.edges.get((src, dst), {})

            src_risk   = G.nodes[src].get("risk_score", 0.0)
            dst_risk   = G.nodes[dst].get("risk_score", 0.0)
            seg_risk   = (src_risk + dst_risk) / 2
            duration   = edge.get("duration_hours", 0.0)
            cost       = edge.get("cost",           0.0)
            mode       = edge.get("transport_mode", "Sea")

            total_risk     += seg_risk
            total_duration += duration
            total_cost     += cost

            path_details.append({
                "from":          G.nodes[src].get("name", src),
                "to":            G.nodes[dst].get("name", dst),
                "transport_mode": mode,
                "duration_hours": duration,
                "cost":           cost,
                "segment_risk":   round(seg_risk, 4)
            })

        avg_risk = total_risk / hops if hops > 0 else 0.0

        return {
            "path":              path,
            "path_names": [
                G.nodes[n].get("name", n) for n in path
            ],
            "hops":              hops,
            "total_risk_score":  round(avg_risk, 4),
            "total_duration_hours": round(total_duration, 2),
            "total_cost":        round(total_cost, 2),
            "risk_level":        self._get_risk_level(avg_risk),
            "segments":          path_details
        }

    def _get_risk_level(self, score: float) -> str:
        if score >= 0.75: return "CRITICAL"
        elif score >= 0.55: return "HIGH"
        elif score >= 0.35: return "MEDIUM"
        elif score >= 0.15: return "LOW"
        else: return "MINIMAL"


def run_ensemble_demo() -> None:
    """
    Demonstrates the full hybrid prediction pipeline
    using a sample shipment from the feature matrix.
    """
    logger.info("=" * 65)
    logger.info("  HYBRID ENSEMBLE DEMO — STARTING")
    logger.info("=" * 65)

    # ---- Load ensemble ----
    ensemble = HybridEnsemble()

    # ---- Load a sample row from feature matrix ----
    df = pd.read_csv("data/processed/feature_matrix.csv")
    df = df.dropna()

    FEATURE_COLS = [
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
    sample_row     = df[available_cols].iloc[0].fillna(0).values

    source_loc     = df["source_location_id"].iloc[0]
    dest_loc       = df["dest_location_id"].iloc[0]
    nlp_risk       = float(df["combined_route_risk"].iloc[0])

    # ---- Make prediction ----
    result = ensemble.predict(
        feature_row     = sample_row,
        source_location = source_loc,
        dest_location   = dest_loc,
        nlp_risk_score  = nlp_risk
    )

    logger.info("=" * 65)
    logger.info("  HYBRID PREDICTION RESULT")
    logger.info("=" * 65)
    logger.info(f"  Route:             {source_loc} → {dest_loc}")
    logger.info(f"  Delay Probability: {result['delay_probability']:.4f}")
    logger.info(f"  Predicted Delay:   {result['predicted_delay_hours']} hours")
    logger.info(f"  Risk Level:        {result['risk_level']}")
    logger.info(f"  Confidence:        {result['confidence']:.4f}")
    logger.info(f"  XGBoost Prob:      {result['xgb_delay_prob']:.4f}")
    logger.info(f"  GNN Prob:          {result['gnn_delay_prob']:.4f}")
    logger.info("=" * 65)

    # ---- Route recommendations ----
    recommender  = RouteRecommendationEngine(ensemble.graph)
    alternatives = recommender.get_alternative_routes(
        source = source_loc,
        target = dest_loc,
        k      = 3
    )

    logger.info("  ALTERNATIVE ROUTE RECOMMENDATIONS")
    logger.info("=" * 65)
    for alt in alternatives:
        logger.info(
            f"  Rank #{alt['rank']} | "
            f"Risk: {alt['total_risk_score']:.4f} [{alt['risk_level']}] | "
            f"Duration: {alt['total_duration_hours']}h | "
            f"Cost: ${alt['total_cost']:,.0f} | "
            f"Path: {' → '.join(alt['path_names'])}"
        )
    logger.info("=" * 65)

    # ---- Save ensemble config ----
    config = {
        "xgb_weight":      XGB_WEIGHT,
        "gnn_weight":      GNN_WEIGHT,
        "feature_columns": available_cols,
        "model_paths": {
            "xgb_clf": XGBOOST_CLF_PATH,
            "xgb_reg": XGBOOST_REG_PATH,
            "scaler":  SCALER_PATH,
            "gnn":     GNN_MODEL_PATH,
            "graph":   GRAPH_OBJECT_PATH
        }
    }
    with open(ENSEMBLE_PATH, "wb") as f:
        pickle.dump(config, f)
    logger.info(f"Ensemble config saved → {ENSEMBLE_PATH}")
    logger.info("  HYBRID ENSEMBLE DEMO COMPLETE")
    logger.info("=" * 65)

if __name__ == "__main__":
    run_ensemble_demo()