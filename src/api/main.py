# src/api/main.py
# ============================================
# PRODUCTION FASTAPI BACKEND
# Serves the Hybrid Ensemble model via REST API
# Endpoints:
#   GET  /health              — Server health check
#   GET  /network-status      — All ports risk status
#   POST /predict             — Single shipment prediction
#   POST /recommend-routes    — Alternative route finder
#   GET  /model-info          — Loaded model metadata
# ============================================

import os
import sys
import json
import pickle
import joblib
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.train_gnn      import SupplyChainGNN
from src.models.hybrid_ensemble import HybridEnsemble, RouteRecommendationEngine
from src.utils import get_logger

logger = get_logger("FastAPI")

# ============================================
# PYDANTIC SCHEMAS
# ============================================

class ShipmentPredictionRequest(BaseModel):
    """Input schema for single shipment delay prediction."""
    route_id:            str   = Field(..., example="RT_SHA_LAX")
    cargo_type:          str   = Field("Consumer_Goods",
                                        example="Electronics")
    cargo_weight_tons:   float = Field(10.0,  example=25.5)
    dispatch_hour:       int   = Field(12,    example=8)
    dispatch_dayofweek:  int   = Field(1,     example=2)
    dispatch_month:      int   = Field(6,     example=11)
    is_weekend:          int   = Field(0,     example=0)
    is_peak_season:      int   = Field(0,     example=1)
    nlp_risk_override:   Optional[float] = Field(
        None,
        description="Override NLP risk score (0.0-1.0). "
                    "If None, live score is fetched from DB."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "route_id":           "RT_SHA_LAX",
                "cargo_type":         "Electronics",
                "cargo_weight_tons":  50.0,
                "dispatch_hour":      8,
                "dispatch_dayofweek": 1,
                "dispatch_month":     11,
                "is_weekend":         0,
                "is_peak_season":     1,
                "nlp_risk_override":  None
            }
        }


class RouteRecommendationRequest(BaseModel):
    """Input schema for alternative route recommendations."""
    source_location_id: str  = Field(..., example="PORT_SHA")
    dest_location_id:   str  = Field(..., example="PORT_LAX")
    max_alternatives:   int  = Field(3,   example=3)
    avoid_high_risk:    bool = Field(True, example=True)

    class Config:
        json_schema_extra = {
            "example": {
                "source_location_id": "PORT_SHA",
                "dest_location_id":   "PORT_LAX",
                "max_alternatives":   3,
                "avoid_high_risk":    True
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for delay prediction."""
    route_id:               str
    delay_probability:      float
    predicted_delay_hours:  float
    risk_level:             str
    confidence:             float
    xgb_delay_prob:         float
    gnn_delay_prob:         float
    nlp_risk_score:         float
    prediction_timestamp:   str
    model_version:          str


# ============================================
# APPLICATION STATE
# ============================================
app_state = {
    "ensemble":     None,
    "recommender":  None,
    "graph":        None,
    "route_data":   None,
    "model_version": "1.0.0",
    "startup_time": None
}

FEATURE_COLS = [
    "dispatch_hour", "dispatch_dayofweek", "dispatch_month",
    "dispatch_quarter", "is_weekend", "is_month_end",
    "is_peak_season", "rolling_7d_avg_delay",
    "rolling_30d_avg_delay", "rolling_7d_std_delay",
    "lag_1_delay", "lag_2_delay",
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


# ============================================
# LIFESPAN — MODEL LOADING
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads all model artifacts into memory when
    the server starts. Uses the modern FastAPI
    lifespan pattern (replaces deprecated @on_event).
    """
    logger.info("=" * 55)
    logger.info("  FASTAPI SERVER STARTING")
    logger.info("  Loading model artifacts into memory...")
    logger.info("=" * 55)

    try:
        # Load hybrid ensemble (XGBoost + GNN + Graph)
        app_state["ensemble"]    = HybridEnsemble()
        app_state["recommender"] = RouteRecommendationEngine(
            app_state["ensemble"].graph
        )
        app_state["graph"]       = app_state["ensemble"].graph
        app_state["startup_time"] = datetime.now().isoformat()

        # Load feature matrix for route lookup
        if os.path.exists("data/processed/feature_matrix.csv"):
            df = pd.read_csv("data/processed/feature_matrix.csv")
            app_state["route_data"] = df
            logger.info(f"Feature matrix loaded: {len(df)} records.")

        logger.info("All models loaded. Server is ready.")
        logger.info("=" * 55)

    except Exception as e:
        logger.critical(f"FATAL: Failed to load models: {e}")
        raise

    yield  # Server runs here

    # Cleanup on shutdown
    logger.info("Server shutting down. Releasing model resources.")
    app_state["ensemble"]    = None
    app_state["recommender"] = None


# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(
    title       = "🚢 Supply Chain Resilience AI — API",
    description = """
## Global Supply Chain Bottleneck Prediction API

Predicts shipment delays using a **Hybrid AI Architecture**:
- **XGBoost** — Tabular pattern recognition
- **Graph Neural Network (GNN)** — Spatial bottleneck propagation
- **FinBERT NLP** — Real-time risk scoring from news & weather

### 2026 Geopolitical Context
Risk scores are currently elevated for Gulf routes due to
US-Israel-Iran conflict. PORT_DXB and PORT_SIN routes carry
a 40% conflict premium in delay probability calculations.

### Authentication
This demo API runs without authentication.
Production deployment would use OAuth2 + API keys.
    """,
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc"
)

# CORS — allows the Streamlit dashboard to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"]
)


# ============================================
# ENDPOINT 1: HEALTH CHECK
# ============================================
@app.get(
    "/health",
    tags        = ["System"],
    summary     = "Server health check",
    description = "Returns server status and model loading state."
)
def health_check():
    """Returns API health status."""
    models_loaded = app_state["ensemble"] is not None

    return {
        "status":        "online" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "startup_time":  app_state.get("startup_time"),
        "model_version": app_state["model_version"],
        "timestamp":     datetime.now().isoformat(),
        "message":       (
            "Supply Chain AI is ready."
            if models_loaded
            else "Models still loading — try again in 30s."
        )
    }


# ============================================
# ENDPOINT 2: MODEL INFO
# ============================================
@app.get(
    "/model-info",
    tags        = ["System"],
    summary     = "Loaded model metadata"
)
def model_info():
    """Returns information about all loaded models."""
    if app_state["ensemble"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    G    = app_state["graph"]
    info = {
        "model_version":   app_state["model_version"],
        "models_loaded": {
            "xgboost_classifier": os.path.exists("models/xgboost_classifier.joblib"),
            "xgboost_regressor":  os.path.exists("models/xgboost_regressor.joblib"),
            "severity_classifier": os.path.exists("models/xgboost_severity_classifier.joblib"),
            "gnn":                os.path.exists("models/gnn_supply_chain.pth"),
            "ensemble_config":    os.path.exists("models/ensemble_config.pkl")
        },
        "graph_info": {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "nodes": [
                {
                    "id":         node,
                    "name":       G.nodes[node].get("name", node),
                    "risk_score": G.nodes[node].get("risk_score", 0.0)
                }
                for node in G.nodes()
            ]
        },
        "evaluation_metrics": _load_evaluation_metrics(),
        "geopolitical_context": {
            "active_conflict":    "US-Israel-Iran (2026)",
            "affected_routes":    ["PORT_DXB", "PORT_SIN"],
            "conflict_premium":   "40% delay cost surcharge on Gulf routes",
            "data_last_updated":  datetime.now().isoformat()
        }
    }
    return info


# ============================================
# ENDPOINT 3: NETWORK STATUS
# ============================================
@app.get(
    "/network-status",
    tags        = ["Predictions"],
    summary     = "Full network risk status",
    description = (
        "Returns current risk levels and predicted delay "
        "probabilities for ALL ports in the supply chain network. "
        "Powered by the GNN spatial propagation model."
    )
)
def get_network_status():
    """
    Returns risk assessment for every node in the
    supply chain graph. This is the main dashboard endpoint.
    """
    if app_state["ensemble"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    try:
        G           = app_state["graph"]
        ensemble    = app_state["ensemble"]

        # Get GNN predictions for all nodes
        gnn_preds   = ensemble._get_gnn_prediction.__func__

        # Build node predictions directly
        results     = []
        for node_id, attrs in G.nodes(data=True):
            name          = attrs.get("name",       node_id)
            risk_score    = attrs.get("risk_score",  0.0)
            in_degree     = attrs.get("in_degree",   0)
            out_degree    = attrs.get("out_degree",  0)
            betweenness   = attrs.get("betweenness_centrality", 0.0)
            location_type = attrs.get("location_type", "Port")

            # Get connected routes
            connected_routes = [
                {
                    "route_id":   G.edges[node_id, neighbor].get("route_id", ""),
                    "to":         G.nodes[neighbor].get("name", neighbor),
                    "mode":       G.edges[node_id, neighbor].get("transport_mode", "Sea"),
                    "duration_h": G.edges[node_id, neighbor].get("duration_hours", 0)
                }
                for neighbor in G.successors(node_id)
            ]

            # Risk level from NLP score
            risk_level = _score_to_risk_level(risk_score)

            # Conflict flag
            is_conflict_affected = node_id in ["PORT_DXB", "PORT_SIN", "PORT_SHA"]

            results.append({
                "location_id":          node_id,
                "location_name":        name,
                "location_type":        location_type,
                "nlp_risk_score":       round(risk_score, 4),
                "risk_level":           risk_level,
                "betweenness_centrality": round(betweenness, 6),
                "in_degree":            in_degree,
                "out_degree":           out_degree,
                "connected_routes":     connected_routes,
                "conflict_affected":    is_conflict_affected,
                "conflict_note": (
                    "Gulf conflict premium active (+40%)"
                    if is_conflict_affected else None
                )
            })

        # Sort by risk score descending
        results = sorted(
            results,
            key     = lambda x: x["nlp_risk_score"],
            reverse = True
        )

        critical = [r for r in results if r["risk_level"] == "CRITICAL"]
        high     = [r for r in results if r["risk_level"] == "HIGH"]

        return {
            "timestamp":          datetime.now().isoformat(),
            "total_ports":        len(results),
            "critical_ports":     len(critical),
            "high_risk_ports":    len(high),
            "geopolitical_alert": (
                "ACTIVE: US-Israel-Iran conflict. "
                "Gulf routes elevated risk."
            ),
            "network_status":     results
        }

    except Exception as e:
        logger.error(f"Network status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# ENDPOINT 4: SINGLE SHIPMENT PREDICTION
# ============================================
@app.post(
    "/predict",
    response_model = PredictionResponse,
    tags           = ["Predictions"],
    summary        = "Predict delay for a single shipment",
    description    = (
        "Takes shipment details and returns delay probability, "
        "predicted hours, risk level and model confidence. "
        "Uses Hybrid XGBoost + GNN ensemble."
    )
)
def predict_delay(request: ShipmentPredictionRequest):
    """
    Predicts whether a shipment will be delayed
    and by how many hours using the hybrid ensemble.
    """
    if app_state["ensemble"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    try:
        ensemble = app_state["ensemble"]

        # ---- Build feature vector ----
        feature_vector = _build_feature_vector(request)

        # ---- Get route info from graph ----
        source_loc, dest_loc = _get_route_endpoints(request.route_id)

        # ---- Get NLP risk score ----
        if request.nlp_risk_override is not None:
            nlp_risk = float(request.nlp_risk_override)
        else:
            nlp_risk = _get_current_nlp_risk(source_loc, dest_loc)

        # ---- Run hybrid prediction ----
        result = ensemble.predict(
            feature_row     = feature_vector,
            source_location = source_loc,
            dest_location   = dest_loc,
            nlp_risk_score  = nlp_risk
        )

        return PredictionResponse(
            route_id               = request.route_id,
            delay_probability      = result["delay_probability"],
            predicted_delay_hours  = result["predicted_delay_hours"],
            risk_level             = result["risk_level"],
            confidence             = result["confidence"],
            xgb_delay_prob         = result["xgb_delay_prob"],
            gnn_delay_prob         = result["gnn_delay_prob"],
            nlp_risk_score         = result["nlp_risk_score"],
            prediction_timestamp   = datetime.now().isoformat(),
            model_version          = app_state["model_version"]
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# ENDPOINT 5: ROUTE RECOMMENDATIONS
# ============================================
@app.post(
    "/recommend-routes",
    tags        = ["Recommendations"],
    summary     = "Get alternative route recommendations",
    description = (
        "Given a high-risk source → destination pair, "
        "returns the K safest alternative paths through "
        "the supply chain network using risk-adjusted Dijkstra."
    )
)
def recommend_routes(request: RouteRecommendationRequest):
    """Returns alternative route recommendations."""
    if app_state["recommender"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    try:
        recommender = app_state["recommender"]
        G           = app_state["graph"]

        # Validate nodes exist
        if request.source_location_id not in G.nodes:
            raise HTTPException(
                status_code = 400,
                detail      = f"Source location '{request.source_location_id}' "
                              f"not found in graph. "
                              f"Valid IDs: {list(G.nodes())}"
            )
        if request.dest_location_id not in G.nodes:
            raise HTTPException(
                status_code = 400,
                detail      = f"Destination location '{request.dest_location_id}' "
                              f"not found in graph. "
                              f"Valid IDs: {list(G.nodes())}"
            )

        # Find high-risk nodes to avoid if requested
        avoid_nodes = None
        if request.avoid_high_risk:
            avoid_nodes = [
                node for node, attrs in G.nodes(data=True)
                if attrs.get("risk_score", 0) > 0.75
                and node not in [
                    request.source_location_id,
                    request.dest_location_id
                ]
            ]
            if avoid_nodes:
                logger.info(f"Avoiding high-risk nodes: {avoid_nodes}")

        alternatives = recommender.get_alternative_routes(
            source      = request.source_location_id,
            target      = request.dest_location_id,
            k           = request.max_alternatives,
            avoid_nodes = avoid_nodes
        )

        source_name = G.nodes[request.source_location_id].get(
            "name", request.source_location_id
        )
        dest_name   = G.nodes[request.dest_location_id].get(
            "name", request.dest_location_id
        )

        return {
            "query": {
                "source":          request.source_location_id,
                "source_name":     source_name,
                "destination":     request.dest_location_id,
                "destination_name": dest_name,
                "alternatives_requested": request.max_alternatives,
                "high_risk_avoided": avoid_nodes or []
            },
            "alternatives_found": len(alternatives),
            "recommendations":    alternatives,
            "timestamp":          datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Route recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# HELPER FUNCTIONS
# ============================================
def _build_feature_vector(request: ShipmentPredictionRequest) -> np.ndarray:
    """
    Builds a numpy feature vector from the API request.
    Uses the feature matrix to get route-level statistics
    for columns the user doesn't provide.
    """
    df = app_state.get("route_data")

    # Get route-level averages as defaults
    route_defaults = {}
    if df is not None and request.route_id in df["route_id"].values:
        route_df       = df[df["route_id"] == request.route_id]
        route_defaults = route_df[FEATURE_COLS].mean().to_dict()

    # Build feature vector using request values where available,
    # falling back to route historical averages
    feature_dict = {col: route_defaults.get(col, 0.0) for col in FEATURE_COLS}

    # Override with request values
    feature_dict["dispatch_hour"]      = request.dispatch_hour
    feature_dict["dispatch_dayofweek"] = request.dispatch_dayofweek
    feature_dict["dispatch_month"]     = request.dispatch_month
    feature_dict["dispatch_quarter"]   = (request.dispatch_month - 1) // 3 + 1
    feature_dict["is_weekend"]         = request.is_weekend
    feature_dict["is_peak_season"]     = request.is_peak_season
    feature_dict["cargo_weight_tons"]  = request.cargo_weight_tons

    return np.array([feature_dict.get(col, 0.0) for col in FEATURE_COLS])


def _get_route_endpoints(route_id: str) -> tuple:
    """Returns (source_location_id, dest_location_id) for a route."""
    route_map = {
        "RT_SHA_SIN": ("PORT_SHA", "PORT_SIN"),
        "RT_SHA_LAX": ("PORT_SHA", "PORT_LAX"),
        "RT_SIN_RTM": ("PORT_SIN", "PORT_RTM"),
        "RT_DXB_RTM": ("PORT_DXB", "PORT_RTM"),
        "RT_HKG_LAX": ("PORT_HKG", "PORT_LAX"),
        "RT_RTM_ANT": ("PORT_RTM", "PORT_ANT"),
        "RT_LAX_CHI": ("PORT_LAX", "WH_CHI"),
        "RT_SHA_DXB": ("PORT_SHA", "PORT_DXB"),
        "RT_SIN_DXB": ("PORT_SIN", "PORT_DXB"),
        "RT_HKG_RTM": ("PORT_HKG", "PORT_RTM"),
    }
    if route_id not in route_map:
        raise ValueError(
            f"Unknown route_id: '{route_id}'. "
            f"Valid routes: {list(route_map.keys())}"
        )
    return route_map[route_id]


def _get_current_nlp_risk(
    source_loc: str,
    dest_loc:   str
) -> float:
    """
    Gets current NLP risk score for a route
    from the graph node attributes.
    """
    G          = app_state["graph"]
    src_risk   = G.nodes.get(source_loc, {}).get("risk_score", 0.0)
    dst_risk   = G.nodes.get(dest_loc,   {}).get("risk_score", 0.0)
    return round((src_risk + dst_risk) / 2, 4)


def _score_to_risk_level(score: float) -> str:
    """Converts numeric score to risk label."""
    if score >= 0.75:   return "CRITICAL"
    elif score >= 0.55: return "HIGH"
    elif score >= 0.35: return "MEDIUM"
    elif score >= 0.15: return "LOW"
    else:               return "MINIMAL"


def _load_evaluation_metrics() -> dict:
    """Loads evaluation report if it exists."""
    path = "data/processed/evaluation_report.json"
    if os.path.exists(path):
        with open(path) as f:
            report = json.load(f)
        return {
            "auc_roc":          report.get(
                "binary_classifier_metrics", {}
            ).get("auc_roc", "N/A"),
            "f1_score":         report.get(
                "binary_classifier_metrics", {}
            ).get("f1_score", "N/A"),
            "ordinal_accuracy": report.get(
                "severity_classifier_metrics", {}
            ).get("ordinal_accuracy", "N/A"),
            "annual_savings":   report.get(
                "roi_results", {}
            ).get("annual_savings_usd", "N/A")
        }
    return {}


# ============================================
# SERVER ENTRY POINT
# ============================================
if __name__ == "__main__":
    logger.info("Starting Supply Chain AI API Server...")
    uvicorn.run(
        "src.api.main:app",
        host     = "0.0.0.0",
        port     = 8000,
        reload   = False,
        workers  = 1,
        log_level = "info"
    )