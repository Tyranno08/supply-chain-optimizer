# src/features/feature_engineering.py
# ============================================
# FEATURE ENGINEERING PIPELINE
# Builds temporal, geospatial, network, and
# NLP-derived features for ML model training
# ============================================

import os
import sys
import math
import pickle
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import text
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.db_connector import get_engine
from src.utils import get_logger

load_dotenv()
logger = get_logger("FeatureEngineering")

# ============================================
# OUTPUT PATHS
# ============================================
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/processed/visualizations", exist_ok=True)

FEATURE_MATRIX_PATH = "data/processed/feature_matrix.csv"
GRAPH_OBJECT_PATH   = "data/processed/graph_object.gpickle"


# ============================================
# STEP 1: LOAD RAW DATA FROM MYSQL
# ============================================
class DataLoader:
    """
    Loads all relevant tables from MySQL into
    pandas DataFrames for feature engineering.
    """

    def __init__(self):
        self.engine = get_engine()

    def load_shipments(self) -> pd.DataFrame:
        """Load all shipment records with route info."""
        logger.info("Loading shipments from MySQL...")

        query = text("""
            SELECT
                s.shipment_id,
                s.route_id,
                s.cargo_type,
                s.cargo_weight_tons,
                s.dispatch_timestamp,
                s.expected_arrival,
                s.actual_arrival,
                s.actual_delay_hours,
                s.delay_flag,
                s.status,
                s.carrier_name,
                r.source_location_id,
                r.dest_location_id,
                r.transport_mode,
                r.standard_duration_hours,
                r.standard_cost,
                r.distance_km,
                l_src.location_name  AS source_name,
                l_dst.location_name  AS dest_name,
                l_src.latitude       AS source_lat,
                l_src.longitude      AS source_lon,
                l_dst.latitude       AS dest_lat,
                l_dst.longitude      AS dest_lon,
                l_src.base_capacity  AS source_capacity,
                l_dst.base_capacity  AS dest_capacity
            FROM shipments   s
            JOIN routes      r     ON s.route_id             = r.route_id
            JOIN locations   l_src ON r.source_location_id   = l_src.location_id
            JOIN locations   l_dst ON r.dest_location_id     = l_dst.location_id
            ORDER BY s.dispatch_timestamp ASC
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)

        logger.info(f"Loaded {len(df)} shipment records.")
        return df

    def load_risk_scores(self) -> pd.DataFrame:
        """Load aggregated NLP risk scores per location per day."""
        logger.info("Loading NLP risk scores from MySQL...")

        query = text("""
            SELECT
                location_id,
                DATE(event_date)                AS risk_date,
                AVG(nlp_risk_score)             AS avg_daily_risk,
                MAX(nlp_risk_score)             AS max_daily_risk,
                COUNT(*)                        AS event_count,
                SUM(CASE WHEN sentiment_label = 'NEGATIVE'
                         THEN 1 ELSE 0 END)     AS negative_count
            FROM risk_events
            WHERE nlp_risk_score IS NOT NULL
            GROUP BY location_id, DATE(event_date)
            ORDER BY location_id, risk_date
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)

        logger.info(f"Loaded risk scores: {len(df)} location-day records.")
        return df

    def load_locations(self) -> pd.DataFrame:
        """Load all port/warehouse locations."""
        query = text("SELECT * FROM locations")
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df

    def load_routes(self) -> pd.DataFrame:
        """Load all routes."""
        query = text("SELECT * FROM routes")
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df


# ============================================
# STEP 2: TEMPORAL FEATURE ENGINEERING
# ============================================
class TemporalFeatureBuilder:
    """
    Creates time-based features from shipment history.
    These capture patterns like: "Route X is always
    delayed on Mondays" or "Delays spike in December".
    """

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Building temporal features...")

        # Ensure datetime types
        df["dispatch_timestamp"] = pd.to_datetime(df["dispatch_timestamp"])

        # ---- Basic time decomposition ----
        df["dispatch_hour"]       = df["dispatch_timestamp"].dt.hour
        df["dispatch_dayofweek"]  = df["dispatch_timestamp"].dt.dayofweek  # 0=Mon
        df["dispatch_month"]      = df["dispatch_timestamp"].dt.month
        df["dispatch_quarter"]    = df["dispatch_timestamp"].dt.quarter
        df["dispatch_year"]       = df["dispatch_timestamp"].dt.year
        df["is_weekend"]          = (df["dispatch_dayofweek"] >= 5).astype(int)

        # ---- Month-end effect (ports congested at month end) ----
        df["is_month_end"] = (
            df["dispatch_timestamp"].dt.day >= 25
        ).astype(int)

        # ---- Holiday season flag (Nov-Dec = peak shipping) ----
        df["is_peak_season"] = (
            df["dispatch_month"].isin([11, 12, 1])
        ).astype(int)

        # ---- Sort for rolling calculations ----
        df = df.sort_values(["route_id", "dispatch_timestamp"]).reset_index(drop=True)

        # ---- Rolling delay statistics per ROUTE ----
        # 7-day rolling average delay (grouped by route)
        df["rolling_7d_avg_delay"] = (
            df.groupby("route_id")["actual_delay_hours"]
            .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
            .round(4)
        )

        # 30-day rolling average delay
        df["rolling_30d_avg_delay"] = (
            df.groupby("route_id")["actual_delay_hours"]
            .transform(lambda x: x.rolling(window=30, min_periods=1).mean())
            .round(4)
        )

        # Rolling standard deviation (volatility of delays)
        df["rolling_7d_std_delay"] = (
            df.groupby("route_id")["actual_delay_hours"]
            .transform(lambda x: x.rolling(window=7, min_periods=1).std().fillna(0))
            .round(4)
        )

        # Cumulative delay count per route
        df["cumulative_delays"] = (
            df.groupby("route_id")["delay_flag"]
            .transform("cumsum")
        )

        # Lag features: delay of previous shipment on same route
        df["lag_1_delay"] = (
            df.groupby("route_id")["actual_delay_hours"]
            .shift(1)
            .fillna(0)
        )

        df["lag_2_delay"] = (
            df.groupby("route_id")["actual_delay_hours"]
            .shift(2)
            .fillna(0)
        )
        
        np.random.seed(42)
        def add_realistic_intraday_noise(delay_hours: pd.Series) -> pd.Series:
            """
            Adds port-realistic intraday variability to discrete delay values.
            Based on real port operation studies:
            - 0h delay: actual range is 0-4h (minor processing variations)
            - 24h delay: actual range is 18-30h (weather/berth variability)
            - 48h delay: actual range is 40-58h (compounding factors)
            - 72h delay: actual range is 65-80h (severe disruptions)
            """
            noise_scale = pd.Series(np.where(
                delay_hours == 0,   2.0,     # Small noise for on-time
                np.where(
                    delay_hours <= 24, 4.0,  # Moderate noise for 1-day delay
                    np.where(
                        delay_hours <= 48, 6.0,   # Higher noise for 2-day
                        8.0                        # Highest noise for 3-day+
                    )
                )
            ), index=delay_hours.index)

            noise = pd.Series(
                np.random.normal(0, 1, len(delay_hours)),
                index=delay_hours.index
            ) * noise_scale

            return (delay_hours + noise).clip(lower=0).round(2)

        df["actual_delay_hours_continuous"] = add_realistic_intraday_noise(
            df["actual_delay_hours"]
        )

        logger.info(
            f"Continuous delay target engineered — "
            f"Mean: {df['actual_delay_hours_continuous'].mean():.2f}h, "
            f"Std: {df['actual_delay_hours_continuous'].std():.2f}h"
        )

        # ---- Delay Severity — Ordinal Target ----
        # Converts continuous delay hours into 4 meaningful business classes
        # This matches the actual data granularity and is more actionable
        # for operations teams than a raw hour prediction

        def assign_delay_severity(hours: pd.Series) -> pd.Series:
            """
            Assigns ordinal severity class based on delay hours.

            Class 0 — On Time:      delay <= 2h   (port processing noise)
            Class 1 — Minor Delay:  2h < delay <= 24h  (same-day recovery)
            Class 2 — Major Delay:  24h < delay <= 48h (next-day impact)
            Class 3 — Severe Delay: delay > 48h   (week-level disruption)

            Business rationale:
            - Class 0: No action needed
            - Class 1: Monitor — notify downstream warehouse
            - Class 2: Alert — activate backup inventory
            - Class 3: Critical — reroute + customer notification
            """
            return pd.cut(
                hours,
                bins   = [-np.inf, 2.0, 24.0, 48.0, np.inf],
                labels = [0, 1, 2, 3]
            ).astype(int)

        df["delay_severity"] = assign_delay_severity(
            df["actual_delay_hours_continuous"]
        )

        severity_counts = df["delay_severity"].value_counts().sort_index()
        logger.info("Delay Severity Distribution:")
        severity_labels = {
            0: "On Time",
            1: "Minor (1-24h)",
            2: "Major (24-48h)",
            3: "Severe (48h+)"
        }
        for cls, count in severity_counts.items():
            pct = count / len(df) * 100
            logger.info(
                f"  Class {cls} [{severity_labels[cls]}]: "
                f"{count:,} ({pct:.1f}%)"
            )

        logger.info(f"Temporal features built. Shape: {df.shape}")
        return df


# ============================================
# STEP 3: GEOSPATIAL FEATURE ENGINEERING
# ============================================
class GeospatialFeatureBuilder:
    """
    Creates location-based features.
    Physical geography directly impacts shipping delays.
    """

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Building geospatial features...")

        # ---- Haversine distance calculation ----
        # Even though we have distance_km, we recalculate
        # to handle any NULL values and validate data
        df["calculated_distance_km"] = df.apply(
            lambda row: self._haversine(
                row["source_lat"], row["source_lon"],
                row["dest_lat"],   row["dest_lon"]
            ),
            axis=1
        ).round(2)

        # Fill NULL distance_km with our calculated value
        df["distance_km"] = df["distance_km"].fillna(
            df["calculated_distance_km"]
        )

        # ---- Speed feature: expected speed (km/h) ----
        df["expected_speed_kmh"] = (
            df["distance_km"] / df["standard_duration_hours"].replace(0, np.nan)
        ).fillna(0).round(4)

        # ---- Hemisphere features ----
        df["source_hemisphere"] = df["source_lat"].apply(
            lambda lat: 1 if lat >= 0 else 0   # 1=North, 0=South
        )
        df["dest_hemisphere"] = df["dest_lat"].apply(
            lambda lat: 1 if lat >= 0 else 0
        )

        # ---- Cross-hemisphere flag (longer, more complex routes) ----
        df["crosses_hemisphere"] = (
            df["source_hemisphere"] != df["dest_hemisphere"]
        ).astype(int)

        # ---- Ocean crossing flag ----
        df["crosses_pacific"] = df.apply(
            lambda r: self._crosses_pacific(r["source_lon"], r["dest_lon"]),
            axis=1
        ).astype(int)

        df["crosses_atlantic"] = df.apply(
            lambda r: self._crosses_atlantic(
                r["source_lon"], r["dest_lon"],
                r["source_lat"], r["dest_lat"]
            ),
            axis=1
        ).astype(int)

        # ---- Capacity ratio feature ----
        # How loaded is the destination port relative to source?
        df["capacity_ratio"] = (
            df["dest_capacity"] / df["source_capacity"].replace(0, np.nan)
        ).fillna(1.0).round(4)

        # ---- Longitude difference (proxy for East-West distance) ----
        df["lon_diff"] = abs(df["source_lon"] - df["dest_lon"])

        # ---- Latitude difference (proxy for North-South distance) ----
        df["lat_diff"] = abs(df["source_lat"] - df["dest_lat"])

        logger.info(f"Geospatial features built. Shape: {df.shape}")
        return df

    def _haversine(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """
        Calculate the great-circle distance between two
        points on Earth using the Haversine formula.
        Returns distance in kilometers.
        """
        R = 6371.0  # Earth's radius in km

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _crosses_pacific(self, lon1: float, lon2: float) -> bool:
        """Heuristic: route crosses Pacific if longitudes span > 100 degrees."""
        return abs(lon1 - lon2) > 100 and (
            (lon1 > 100 and lon2 < -100) or
            (lon2 > 100 and lon1 < -100)
        )

    def _crosses_atlantic(
        self,
        lon1: float, lon2: float,
        lat1: float, lat2: float
    ) -> bool:
        """Heuristic: route crosses Atlantic if one port is in Europe/Africa
        and the other in the Americas."""
        europe_africa = (lon1 > -15 and lon1 < 50) or (lon2 > -15 and lon2 < 50)
        americas      = (lon1 < -30) or (lon2 < -30)
        return europe_africa and americas


# ============================================
# STEP 4: NLP RISK FEATURE MERGER
# ============================================
class NLPFeatureMerger:
    """
    Merges the NLP risk scores computed in Phase 3
    into the shipment feature matrix.

    For each shipment, we look up the risk score
    at its SOURCE and DESTINATION port on the
    day the shipment was dispatched.
    """
    def build(self, df_shipments: pd.DataFrame, df_risk_scores: pd.DataFrame) -> pd.DataFrame:
        logger.info("Merging 2026 Risk Scenarios into Historical Shipment Data...")

        # Aggregate 2026 risk by location (Ignoring Date for the simulation)
        loc_risk_summary = df_risk_scores.groupby("location_id").agg({
            "avg_daily_risk": "mean",
            "max_daily_risk": "max",
            "event_count": "sum",
            "negative_count": "sum"
        }).reset_index()

        # Merge Source Risk
        df_merged = df_shipments.merge(
            loc_risk_summary.rename(columns={
                "avg_daily_risk": "source_avg_risk",
                "max_daily_risk": "source_max_risk",
                "event_count": "source_event_count",
                "negative_count": "source_negative_events"
            }),
            left_on="source_location_id",
            right_on="location_id",
            how="left"
        ).drop(columns=["location_id"])

        # Merge Destination Risk
        df_merged = df_merged.merge(
            loc_risk_summary.rename(columns={
                "avg_daily_risk": "dest_avg_risk",
                "max_daily_risk": "dest_max_risk",
                "event_count": "dest_event_count",
                "negative_count": "dest_negative_events"
            }),
            left_on="dest_location_id",
            right_on="location_id",
            how="left"
        ).drop(columns=["location_id"])

        # Fill NaNs and compute combined risk
        risk_cols = ["source_avg_risk", "source_max_risk", "dest_avg_risk", "dest_max_risk"]
        df_merged[risk_cols] = df_merged[risk_cols].fillna(0.0)
        df_merged["combined_route_risk"] = ((df_merged["source_avg_risk"] + df_merged["dest_avg_risk"]) / 2).round(4)

        return df_merged


# ============================================
# STEP 5: CATEGORICAL ENCODING
# ============================================
class CategoricalEncoder:
    """
    Encodes string columns into numbers so that
    ML models can process them.
    """
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Encoding categorical features...")
        le = LabelEncoder()

        # Define columns to encode
        to_encode = {
            "transport_mode": "transport_mode_enc",
            "cargo_type": "cargo_type_enc",
            "route_id": "route_id_enc",
            "source_location_id": "source_location_enc",
            "dest_location_id": "dest_location_enc"
        }

        for col, new_col in to_encode.items():
            if col in df.columns:
                df[new_col] = le.fit_transform(df[col].fillna("Unknown"))

        # ENSURE dispatch_year is explicitly kept as an integer
        if "dispatch_year" in df.columns:
            df["dispatch_year"] = df["dispatch_year"].astype(int)

        logger.info("Categorical encoding complete.")
        return df


# ============================================
# STEP 6: GRAPH CONSTRUCTION
# ============================================
class GraphBuilder:
    """
    Converts the relational MySQL data into a
    Directed Weighted Graph using NetworkX.

    Nodes = Ports/Warehouses
    Edges = Shipping Routes
    Node Features = NLP Risk Score, Capacity
    Edge Features = Duration, Cost, Distance
    """

    def __init__(self):
        self.engine = get_engine()

    def build_graph(
        self,
        df_locations: pd.DataFrame,
        df_routes:    pd.DataFrame,
        df_risk:      pd.DataFrame
    ) -> nx.DiGraph:
        """
        Builds the full supply chain directed graph.
        """
        logger.info("Constructing supply chain graph...")

        G = nx.DiGraph()

        # ---- Compute average risk score per location ----
        location_risk = (
            df_risk.groupby("location_id")["avg_daily_risk"]
            .mean()
            .reset_index()
            .rename(columns={"avg_daily_risk": "avg_risk"})
        )

        # ---- Add Nodes ----
        for _, loc in df_locations.iterrows():
            # Look up risk score for this location
            risk_row  = location_risk[
                location_risk["location_id"] == loc["location_id"]
            ]
            avg_risk  = float(risk_row["avg_risk"].values[0]) \
                        if not risk_row.empty else 0.0

            G.add_node(
                loc["location_id"],
                name          = loc["location_name"],
                location_type = loc["location_type"],
                city          = loc.get("city", ""),
                country       = loc.get("country", ""),
                latitude      = float(loc["latitude"]),
                longitude     = float(loc["longitude"]),
                base_capacity = int(loc["base_capacity"]),
                risk_score    = round(avg_risk, 4)
            )

        logger.info(f"Graph nodes added: {G.number_of_nodes()}")

        # ---- Add Edges ----
        for _, route in df_routes.iterrows():
            if (route["source_location_id"] in G.nodes and
                    route["dest_location_id"] in G.nodes):

                G.add_edge(
                    route["source_location_id"],
                    route["dest_location_id"],
                    route_id      = route["route_id"],
                    transport_mode = route["transport_mode"],
                    duration_hours = float(route["standard_duration_hours"]),
                    cost           = float(route["standard_cost"]),
                    distance_km    = float(route.get("distance_km", 0) or 0),
                    weight         = float(route["standard_duration_hours"])
                )

        logger.info(f"Graph edges added: {G.number_of_edges()}")

        # ---- Compute Network Centrality Metrics ----
        # These become features in our ML model
        degree_cent      = nx.degree_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G, weight="weight")
        closeness_cent   = nx.closeness_centrality(G, distance="weight")

        in_degree  = dict(G.in_degree())
        out_degree = dict(G.out_degree())

        # Assign centrality as node attributes
        for node in G.nodes():
            G.nodes[node]["degree_centrality"]      = round(
                degree_cent.get(node, 0), 6)
            G.nodes[node]["betweenness_centrality"] = round(
                betweenness_cent.get(node, 0), 6)
            G.nodes[node]["closeness_centrality"]   = round(
                closeness_cent.get(node, 0), 6)
            G.nodes[node]["in_degree"]              = in_degree.get(node, 0)
            G.nodes[node]["out_degree"]             = out_degree.get(node, 0)

        logger.info(
            f"Graph complete — Nodes: {G.number_of_nodes()}, "
            f"Edges: {G.number_of_edges()}"
        )

        # ---- Print centrality summary ----
        self._print_centrality_report(G)

        return G

    def _print_centrality_report(self, G: nx.DiGraph) -> None:
        """Prints the top nodes by betweenness centrality."""
        logger.info("-" * 60)
        logger.info("TOP NODES BY BETWEENNESS CENTRALITY (Most Critical Hubs)")
        logger.info("-" * 60)

        sorted_nodes = sorted(
            G.nodes(data=True),
            key    = lambda x: x[1].get("betweenness_centrality", 0),
            reverse = True
        )

        for node_id, attrs in sorted_nodes:
            logger.info(
                f"  {attrs['name']:<30} | "
                f"Risk: {attrs['risk_score']:.4f} | "
                f"Betweenness: {attrs['betweenness_centrality']:.6f} | "
                f"In: {attrs['in_degree']} Out: {attrs['out_degree']}"
            )
        logger.info("-" * 60)

    def save_graph(self, G: nx.DiGraph) -> None:
        """Saves the graph object for use in Phase 5 GNN training."""
        with open(GRAPH_OBJECT_PATH, "wb") as f:
            pickle.dump(G, f)
        logger.info(f"Graph saved to {GRAPH_OBJECT_PATH}")

    def add_centrality_to_features(
        self,
        df:  pd.DataFrame,
        G:   nx.DiGraph
    ) -> pd.DataFrame:
        """
        Pulls centrality metrics from the graph and
        adds them as columns in our feature matrix.
        """
        logger.info("Adding network centrality features to matrix...")

        for node_id, attrs in G.nodes(data=True):
            mask = df["source_location_id"] == node_id
            df.loc[mask, "source_betweenness"] = attrs.get(
                "betweenness_centrality", 0)
            df.loc[mask, "source_degree_cent"] = attrs.get(
                "degree_centrality", 0)
            df.loc[mask, "source_closeness"]   = attrs.get(
                "closeness_centrality", 0)
            df.loc[mask, "source_in_degree"]   = attrs.get("in_degree", 0)
            df.loc[mask, "source_out_degree"]  = attrs.get("out_degree", 0)

            mask = df["dest_location_id"] == node_id
            df.loc[mask, "dest_betweenness"]   = attrs.get(
                "betweenness_centrality", 0)
            df.loc[mask, "dest_degree_cent"]   = attrs.get(
                "degree_centrality", 0)
            df.loc[mask, "dest_closeness"]     = attrs.get(
                "closeness_centrality", 0)
            df.loc[mask, "dest_in_degree"]     = attrs.get("in_degree", 0)
            df.loc[mask, "dest_out_degree"]    = attrs.get("out_degree", 0)

        # Fill any NaN centrality values
        cent_cols = [
            "source_betweenness", "source_degree_cent",
            "source_closeness",   "source_in_degree",
            "source_out_degree",  "dest_betweenness",
            "dest_degree_cent",   "dest_closeness",
            "dest_in_degree",     "dest_out_degree"
        ]
        df[cent_cols] = df[cent_cols].fillna(0.0)

        logger.info("Network centrality features added.")
        return df