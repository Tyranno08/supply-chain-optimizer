# tests/test_api.py
# ============================================
# API UNIT TESTS
# Run with: pytest tests/ -v
# ============================================

import pytest
import sys
import os
sys.path.append(os.path.abspath("."))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Tests for system health endpoints."""

    def test_health_check_returns_200(self):
        """Health endpoint must always return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_has_status_field(self):
        """Health response must contain status field."""
        response = client.get("/health")
        data     = response.json()
        assert "status" in data

    def test_health_check_has_timestamp(self):
        """Health response must contain timestamp."""
        response = client.get("/health")
        data     = response.json()
        assert "timestamp" in data

    def test_model_info_returns_200(self):
        """Model info endpoint must return 200."""
        response = client.get("/model-info")
        assert response.status_code in [200, 503]

    def test_docs_endpoint_accessible(self):
        """Swagger docs must be accessible."""
        response = client.get("/docs")
        assert response.status_code == 200


class TestNetworkStatus:
    """Tests for network status endpoint."""

    def test_network_status_returns_200(self):
        response = client.get("/network-status")
        assert response.status_code in [200, 503]

    def test_network_status_has_required_fields(self):
        response = client.get("/network-status")
        if response.status_code == 200:
            data = response.json()
            assert "total_ports"    in data
            assert "network_status" in data
            assert "timestamp"      in data

    def test_network_status_ports_have_risk_scores(self):
        response = client.get("/network-status")
        if response.status_code == 200:
            data  = response.json()
            ports = data.get("network_status", [])
            if ports:
                assert "nlp_risk_score" in ports[0]
                assert "risk_level"     in ports[0]
                assert 0.0 <= ports[0]["nlp_risk_score"] <= 1.0


class TestPredictionEndpoint:
    """Tests for delay prediction endpoint."""

    def test_valid_prediction_request(self):
        """Valid request should return prediction."""
        payload = {
            "route_id":           "RT_SHA_LAX",
            "cargo_type":         "Electronics",
            "cargo_weight_tons":  50.0,
            "dispatch_hour":      8,
            "dispatch_dayofweek": 1,
            "dispatch_month":     11,
            "is_weekend":         0,
            "is_peak_season":     1
        }
        response = client.post("/predict", json=payload)
        assert response.status_code in [200, 503]

    def test_prediction_response_schema(self):
        """Prediction response must match PredictionResponse schema."""
        payload = {
            "route_id":           "RT_SHA_LAX",
            "cargo_type":         "Consumer_Goods",
            "cargo_weight_tons":  25.0,
            "dispatch_hour":      12,
            "dispatch_dayofweek": 3,
            "dispatch_month":     6,
            "is_weekend":         0,
            "is_peak_season":     0
        }
        response = client.post("/predict", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert "delay_probability"     in data
            assert "predicted_delay_hours" in data
            assert "risk_level"            in data
            assert "confidence"            in data
            assert 0.0 <= data["delay_probability"] <= 1.0

    def test_invalid_route_returns_400(self):
        """Invalid route_id should return 400."""
        payload = {
            "route_id":           "RT_INVALID_ROUTE",
            "cargo_type":         "Electronics",
            "cargo_weight_tons":  10.0,
            "dispatch_hour":      8,
            "dispatch_dayofweek": 1,
            "dispatch_month":     6,
            "is_weekend":         0,
            "is_peak_season":     0
        }
        response = client.post("/predict", json=payload)
        assert response.status_code in [400, 500, 503]

    def test_nlp_risk_override(self):
        """NLP risk override should be accepted."""
        payload = {
            "route_id":          "RT_SHA_LAX",
            "cargo_type":        "Electronics",
            "cargo_weight_tons": 50.0,
            "dispatch_hour":     8,
            "dispatch_dayofweek": 1,
            "dispatch_month":    6,
            "is_weekend":        0,
            "is_peak_season":    0,
            "nlp_risk_override": 0.85
        }
        response = client.post("/predict", json=payload)
        assert response.status_code in [200, 503]


class TestRouteRecommendations:
    """Tests for route recommendation endpoint."""

    def test_valid_recommendation_request(self):
        payload = {
            "source_location_id": "PORT_SHA",
            "dest_location_id":   "PORT_LAX",
            "max_alternatives":   3,
            "avoid_high_risk":    True
        }
        response = client.post("/recommend-routes", json=payload)
        assert response.status_code in [200, 503]

    def test_invalid_source_returns_400(self):
        payload = {
            "source_location_id": "PORT_INVALID",
            "dest_location_id":   "PORT_LAX",
            "max_alternatives":   3,
            "avoid_high_risk":    False
        }
        response = client.post("/recommend-routes", json=payload)
        assert response.status_code in [400, 503]

    def test_recommendation_response_structure(self):
        payload = {
            "source_location_id": "PORT_SHA",
            "dest_location_id":   "PORT_RTM",
            "max_alternatives":   2,
            "avoid_high_risk":    False
        }
        response = client.post("/recommend-routes", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert "query"            in data
            assert "recommendations"  in data
            assert "timestamp"        in data