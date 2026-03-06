# src/dashboard/api_client.py
# ============================================
# FASTAPI CLIENT FOR STREAMLIT DASHBOARD
# All API calls go through this module
# ============================================

import requests
import pandas as pd
from typing import Optional, Dict, List
import streamlit as st

API_BASE_URL = "http://localhost:8000"

# ============================================
# REQUEST TIMEOUT CONFIG
# ============================================
TIMEOUT_SHORT  = 5    # seconds — health checks
TIMEOUT_MEDIUM = 15   # seconds — predictions
TIMEOUT_LONG   = 30   # seconds — network status


class APIClient:
    """
    Centralized HTTP client for all FastAPI calls.
    Handles errors gracefully and returns None
    instead of crashing the dashboard.
    """

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url

    def is_server_online(self) -> bool:
        """Quick check if FastAPI server is reachable."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=TIMEOUT_SHORT
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_health(self) -> Optional[Dict]:
        """Fetches server health status."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=TIMEOUT_SHORT
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    def get_model_info(self) -> Optional[Dict]:
        """Fetches model metadata."""
        try:
            response = requests.get(
                f"{self.base_url}/model-info",
                timeout=TIMEOUT_MEDIUM
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    def get_network_status(self) -> Optional[Dict]:
        """Fetches full network risk status for all ports."""
        try:
            response = requests.get(
                f"{self.base_url}/network-status",
                timeout=TIMEOUT_LONG
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    def predict_delay(self, payload: Dict) -> Optional[Dict]:
        """Sends shipment details and gets delay prediction."""
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json    = payload,
                timeout = TIMEOUT_MEDIUM
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": response.json().get("detail", "Unknown error")}
        except Exception as e:
            return {"error": str(e)}

    def get_route_recommendations(self, payload: Dict) -> Optional[Dict]:
        """Fetches alternative route recommendations."""
        try:
            response = requests.post(
                f"{self.base_url}/recommend-routes",
                json    = payload,
                timeout = TIMEOUT_MEDIUM
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": response.json().get("detail", "Unknown error")}
        except Exception as e:
            return {"error": str(e)}

    def get_network_as_dataframe(self) -> Optional[pd.DataFrame]:
        """Converts network status API response to DataFrame."""
        data = self.get_network_status()
        if data and "network_status" in data:
            return pd.DataFrame(data["network_status"])
        return None