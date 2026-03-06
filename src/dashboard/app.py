# src/dashboard/app.py
# ============================================
# SUPPLY CHAIN RESILIENCE OPTIMIZER
# MAIN STREAMLIT DASHBOARD
# ============================================

import os
import sys
import json
import time
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image

import streamlit as st
from streamlit_folium import st_folium

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.dashboard.api_client  import APIClient
from src.dashboard.map_builder import build_risk_map

# ============================================
# PAGE CONFIGURATION — Must be first st call
# ============================================
st.set_page_config(
    page_title = "Supply Chain Resilience AI",
    page_icon  = "🚢",
    layout     = "wide",
    initial_sidebar_state = "expanded",
    menu_items = {
        "Get Help":    "https://github.com",
        "Report a bug": "https://github.com",
        "About":       "Supply Chain Resilience Optimizer v1.0"
    }
)

# ============================================
# CUSTOM CSS — Dark professional theme
# ============================================
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
    }

    /* Headers */
    h1, h2, h3 { color: #e6edf3 !important; }
    p, li       { color: #8b949e; }

    /* Dataframe */
    .dataframe { background-color: #161b22 !important; }

    /* Buttons */
    .stButton button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #2ea043;
    }

    /* Alert boxes */
    .stAlert {
        background-color: #161b22;
        border: 1px solid #30363d;
    }

    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #161b22;
        border: 1px solid #30363d;
    }

    /* Risk badge styles */
    .risk-critical { color: #e74c3c; font-weight: bold; }
    .risk-high     { color: #e67e22; font-weight: bold; }
    .risk-medium   { color: #f1c40f; font-weight: bold; }
    .risk-low      { color: #2ecc71; font-weight: bold; }
    .risk-minimal  { color: #3498db; font-weight: bold; }

    /* KPI number style */
    .kpi-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #58a6ff;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# INITIALIZE API CLIENT
# ============================================
client = APIClient()

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0">
        <span style="font-size:2.5rem">🚢</span>
        <h2 style="color:#58a6ff;margin:5px 0">Supply Chain AI</h2>
        <p style="color:#8b949e;font-size:12px">v1.0.0 | 2026</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ---- Server Status ----
    server_online = client.is_server_online()
    if server_online:
        st.success("🟢 API Server Online")
    else:
        st.error("🔴 API Server Offline")
        st.warning(
            "Start the API server:\n"
            "```\npython src/api/main.py\n```"
        )

    st.divider()

    # ---- Navigation ----
    st.markdown("### 📍 Navigation")
    page = st.radio(
        label   = "Select Page",
        options = [
            "🌍 Live Network Map",
            "📊 Analytics & Metrics",
            "🔮 Delay Predictor",
            "🗺️ Route Recommender",
            "💰 Business ROI"
        ],
        label_visibility = "collapsed"
    )

    st.divider()

    # ---- Refresh Button ----
    if st.button("🔄 Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # ---- Model Info ----
    st.markdown("### 🤖 Model Architecture")
    st.markdown("""
    <div style="
        background:#0d1117;
        border:1px solid #30363d;
        border-radius:6px;
        padding:10px;
        font-size:12px;
        color:#8b949e;
    ">
        <b style="color:#58a6ff">Hybrid Ensemble</b><br>
        • XGBoost Classifier<br>
        • XGBoost Regressor<br>
        • Severity Classifier<br>
        • Graph Neural Network<br>
        • FinBERT NLP Engine<br><br>
        <b style="color:#58a6ff">Key Metrics</b><br>
        • AUC-ROC: <b style="color:#2ecc71">0.867</b><br>
        • Ordinal Acc: <b style="color:#2ecc71">84.1%</b><br>
        • Annual Savings: <b style="color:#2ecc71">\$25.1M</b>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ---- Geopolitical Alert ----
    st.markdown("### ⚠️ Active Alerts")
    st.error(
        "**2026 Gulf Conflict Active**\n\n"
        "US-Israel-Iran conflict affecting:\n"
        "- PORT_DXB (Dubai)\n"
        "- PORT_SIN (Singapore)\n"
        "- PORT_SHA (Shanghai)\n\n"
        "40% delay cost premium applied."
    )

    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================
# CACHED DATA FETCHERS
# ============================================
@st.cache_data(ttl=300)   # Cache for 5 minutes
def fetch_network_status():
    return client.get_network_status()


@st.cache_data(ttl=600)   # Cache for 10 minutes
def fetch_model_info():
    return client.get_model_info()


# ============================================
# PAGE 1: LIVE NETWORK MAP
# ============================================
def page_live_network_map():
    st.markdown("## 🌍 Live Supply Chain Network Map")
    st.markdown(
        "Real-time risk levels across global shipping network. "
        "Port colors indicate current NLP-derived risk scores."
    )

    # ---- Fetch data ----
    with st.spinner("Fetching live network status from AI API..."):
        network_data = fetch_network_status()

    if not server_online or network_data is None:
        st.warning(
            "Cannot fetch live data. "
            "Displaying map with graph data only."
        )
        network_df = None
    else:
        network_df = pd.DataFrame(
            network_data.get("network_status", [])
        )

    # ---- Top KPI row ----
    if network_data:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Total Ports Monitored",
                network_data.get("total_ports", "—")
            )
        with col2:
            critical = network_data.get("critical_ports", 0)
            st.metric(
                "Critical Risk Ports",
                critical,
                delta      = f"{critical} require action",
                delta_color = "inverse"
            )
        with col3:
            high = network_data.get("high_risk_ports", 0)
            st.metric(
                "High Risk Ports",
                high,
                delta_color = "inverse"
            )
        with col4:
            if network_df is not None and not network_df.empty:
                avg_risk = round(
                    network_df["nlp_risk_score"].mean(), 4
                )
                st.metric("Avg Network Risk", avg_risk)
        with col5:
            st.metric(
                "Data Timestamp",
                datetime.now().strftime("%H:%M:%S")
            )

    st.divider()

    # ---- Map + Table layout ----
    map_col, table_col = st.columns([3, 1])

    with map_col:
        st.markdown("#### 🗺️ Interactive Risk Heatmap")
        show_routes = st.checkbox("Show Shipping Routes", value=True)

        world_map = build_risk_map(
            network_df  = network_df,
            show_routes = show_routes
        )

        st_folium(
            world_map,
            width  = 900,
            height = 500,
            returned_objects = []
        )

    with table_col:
        st.markdown("#### 📋 Port Risk Table")
        if network_df is not None and not network_df.empty:
            display_df = network_df[[
                "location_name",
                "nlp_risk_score",
                "risk_level"
            ]].copy()
            display_df.columns = ["Port", "Risk Score", "Level"]
            display_df = display_df.sort_values(
                "Risk Score", ascending=False
            )

            # Color the risk level column
            def color_risk(val):
                colors = {
                    "CRITICAL": "color: #e74c3c; font-weight: bold",
                    "HIGH":     "color: #e67e22; font-weight: bold",
                    "MEDIUM":   "color: #f1c40f",
                    "LOW":      "color: #2ecc71",
                    "MINIMAL":  "color: #3498db"
                }
                return colors.get(val, "")

            st.dataframe(
                display_df.style.applymap(
                    color_risk, subset=["Level"]
                ),
                use_container_width = True,
                hide_index          = True,
                height              = 480
            )

    # ---- Geopolitical context banner ----
    if network_data:
        st.info(
            f"⚠️ **Geopolitical Alert:** "
            f"{network_data.get('geopolitical_alert', '')}"
        )

    # ---- Route risk bar chart ----
    if network_df is not None and not network_df.empty:
        st.markdown("#### 📊 Port Risk Score Comparison")

        fig = px.bar(
            network_df.sort_values("nlp_risk_score", ascending=False),
            x              = "location_name",
            y              = "nlp_risk_score",
            color          = "risk_level",
            color_discrete_map = {
                "CRITICAL": "#e74c3c",
                "HIGH":     "#e67e22",
                "MEDIUM":   "#f1c40f",
                "LOW":      "#2ecc71",
                "MINIMAL":  "#3498db"
            },
            title          = "NLP Risk Score by Port (Live Data)",
            labels         = {
                "location_name":  "Port",
                "nlp_risk_score": "NLP Risk Score",
                "risk_level":     "Risk Level"
            },
            text           = "nlp_risk_score"
        )
        fig.update_layout(
            plot_bgcolor  = "#0d1117",
            paper_bgcolor = "#0d1117",
            font_color    = "#e6edf3",
            xaxis_tickangle = -30,
            showlegend    = True
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


# ============================================
# PAGE 2: ANALYTICS & METRICS
# ============================================
def page_analytics():
    st.markdown("## 📊 Model Analytics & Evaluation Metrics")

    # ---- Load evaluation report ----
    report_path = "data/processed/evaluation_report.json"
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
    else:
        st.warning("Evaluation report not found. Run Phase 6 first.")
        return

    clf_metrics = report.get("binary_classifier_metrics", {})
    reg_metrics = report.get("regressor_metrics", {})
    sev_metrics = report.get("severity_classifier_metrics", {})
    roi_results = report.get("roi_results", {})

    # ---- Headline KPIs ----
    st.markdown("### 🎯 Model Performance KPIs")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "AUC-ROC Score",
            f"{clf_metrics.get('auc_roc', 0):.4f}",
            delta = "Primary metric",
        )
    with col2:
        st.metric(
            "F1 Score",
            f"{clf_metrics.get('f1_score', 0):.4f}"
        )
    with col3:
        st.metric(
            "Ordinal Accuracy",
            f"{sev_metrics.get('ordinal_accuracy', 0):.4f}",
            delta = "Severity classifier"
        )
    with col4:
        st.metric(
            "Annual ROI",
            f"${roi_results.get('annual_savings_usd', 0):,.0f}"
        )

    st.divider()

    # ---- Classifier metrics gauge charts ----
    st.markdown("### 📈 Classifier Performance")
    g1, g2, g3, g4, g5 = st.columns(5)

    metrics_to_gauge = [
        ("Accuracy",  clf_metrics.get("accuracy",  0), g1),
        ("F1 Score",  clf_metrics.get("f1_score",  0), g2),
        ("Precision", clf_metrics.get("precision", 0), g3),
        ("Recall",    clf_metrics.get("recall",    0), g4),
        ("AUC-ROC",   clf_metrics.get("auc_roc",   0), g5),
    ]

    for label, value, col in metrics_to_gauge:
        with col:
            fig = go.Figure(go.Indicator(
                mode   = "gauge+number",
                value  = value * 100,
                title  = {"text": label, "font": {"color": "#e6edf3", "size": 13}},
                number = {"suffix": "%", "font": {"color": "#58a6ff", "size": 20}},
                gauge  = {
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#8b949e"
                    },
                    "bar":  {"color": "#238636"},
                    "bgcolor": "#161b22",
                    "bordercolor": "#30363d",
                    "steps": [
                        {"range": [0,  50], "color": "#21262d"},
                        {"range": [50, 75], "color": "#1f3a24"},
                        {"range": [75, 100], "color": "#1a4e2e"}
                    ],
                    "threshold": {
                        "line": {"color": "#58a6ff", "width": 3},
                        "thickness": 0.75,
                        "value": value * 100
                    }
                }
            ))
            fig.update_layout(
                height        = 200,
                margin        = dict(l=10, r=10, t=40, b=10),
                paper_bgcolor = "#0d1117",
                font_color    = "#e6edf3"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---- Severity classifier breakdown ----
    st.markdown("### 🎚️ Severity Classifier Performance")
    sev_col1, sev_col2 = st.columns(2)

    with sev_col1:
        severity_data = {
            "Metric":  ["Weighted F1", "Macro F1", "Exact Accuracy", "Ordinal Accuracy"],
            "Value":   [
                sev_metrics.get("weighted_f1",      0),
                sev_metrics.get("macro_f1",          0),
                sev_metrics.get("exact_accuracy",    0),
                sev_metrics.get("ordinal_accuracy",  0)
            ],
            "Note": [
                "Affected by class imbalance",
                "Unweighted per-class average",
                "Exact class match",
                "Within 1 severity class ✅"
            ]
        }
        sev_df = pd.DataFrame(severity_data)
        st.dataframe(sev_df, use_container_width=True, hide_index=True)

        st.info(
            "**Key Insight:** Ordinal Accuracy (84.1%) is the primary metric. "
            "An operations team cares whether a delay is Minor vs Severe, "
            "not whether it is exactly Class 1 vs Class 2."
        )

    with sev_col2:
        fig_sev = go.Figure(go.Bar(
            x           = sev_df["Metric"],
            y           = [v * 100 for v in sev_df["Value"]],
            marker_color = ["#e67e22", "#e74c3c", "#f1c40f", "#2ecc71"],
            text        = [f"{v*100:.1f}%" for v in sev_df["Value"]],
            textposition = "outside"
        ))
        fig_sev.update_layout(
            title         = "Severity Classifier Metrics (%)",
            paper_bgcolor = "#0d1117",
            plot_bgcolor  = "#0d1117",
            font_color    = "#e6edf3",
            yaxis_range   = [0, 110],
            showlegend    = False
        )
        st.plotly_chart(fig_sev, use_container_width=True)

    st.divider()

    # ---- Saved visualization images ----
    st.markdown("### 🖼️ Evaluation Visualizations")
    viz_dir = "data/processed/visualizations"

    viz_tabs = st.tabs([
        "Confusion Matrix",
        "ROC Curve",
        "SHAP Summary",
        "SHAP Bar",
        "GNN Predictions",
        "ROI Dashboard"
    ])

    viz_files = [
        ("04_confusion_matrix.png",    viz_tabs[0]),
        ("05_roc_curve.png",           viz_tabs[1]),
        ("09_shap_summary.png",        viz_tabs[2]),
        ("10_shap_bar.png",            viz_tabs[3]),
        ("12_gnn_predictions.png",     viz_tabs[4]),
        ("13_roi_dashboard.png",       viz_tabs[5]),
    ]

    for filename, tab in viz_files:
        with tab:
            filepath = os.path.join(viz_dir, filename)
            if os.path.exists(filepath):
                st.image(filepath, use_column_width=True)
            else:
                st.warning(f"Image not found: {filename}. Run Phase 6 evaluation.")

    # ---- Data notes ----
    st.divider()
    st.markdown("### 📝 Methodology Notes")
    notes = report.get("notes", {})
    for key, note in notes.items():
        with st.expander(f"📌 {key.replace('_', ' ').title()}"):
            st.markdown(note)


# ============================================
# PAGE 3: DELAY PREDICTOR
# ============================================
def page_delay_predictor():
    st.markdown("## 🔮 Shipment Delay Predictor")
    st.markdown(
        "Enter shipment details to get an instant AI-powered "
        "delay probability using the Hybrid XGBoost + GNN ensemble."
    )

    if not server_online:
        st.error(
            "❌ API server is offline. "
            "Start it with: `python src/api/main.py`"
        )
        return

    # ---- Input form ----
    with st.form("prediction_form"):
        st.markdown("### 📦 Shipment Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            route_id = st.selectbox(
                "Select Route",
                options = [
                    "RT_SHA_LAX", "RT_SHA_SIN", "RT_SHA_DXB",
                    "RT_SIN_RTM", "RT_SIN_DXB", "RT_DXB_RTM",
                    "RT_HKG_LAX", "RT_HKG_RTM", "RT_RTM_ANT",
                    "RT_LAX_CHI"
                ],
                index   = 0,
                help    = "Select the shipping route"
            )
            cargo_type = st.selectbox(
                "Cargo Type",
                options = [
                    "Electronics", "Consumer_Goods",
                    "Raw_Materials", "Perishables", "Machinery"
                ]
            )
            cargo_weight = st.number_input(
                "Cargo Weight (tons)",
                min_value = 0.1,
                max_value = 500.0,
                value     = 25.0,
                step      = 1.0
            )

        with col2:
            dispatch_hour = st.slider(
                "Dispatch Hour (0-23)",
                min_value = 0,
                max_value = 23,
                value     = 8
            )
            dispatch_dow = st.selectbox(
                "Day of Week",
                options      = [0, 1, 2, 3, 4, 5, 6],
                format_func  = lambda x: [
                    "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday", "Sunday"
                ][x]
            )
            dispatch_month = st.selectbox(
                "Dispatch Month",
                options      = list(range(1, 13)),
                format_func  = lambda x: [
                    "", "January", "February", "March", "April",
                    "May", "June", "July", "August", "September",
                    "October", "November", "December"
                ][x],
                index        = 10
            )

        with col3:
            is_weekend = st.toggle(
                "Weekend Dispatch",
                value = dispatch_dow >= 5
            )
            is_peak = st.toggle(
                "Peak Season (Nov-Jan)",
                value = dispatch_month in [11, 12, 1]
            )
            nlp_override = st.slider(
                "NLP Risk Override (0=Auto)",
                min_value = 0.0,
                max_value = 1.0,
                value     = 0.0,
                step      = 0.01,
                help      = "Set to 0 for automatic risk from live API data"
            )

        submitted = st.form_submit_button(
            "🚀 Predict Delay Probability",
            use_container_width = True,
            type                = "primary"
        )

    # ---- Prediction result ----
    if submitted:
        payload = {
            "route_id":           route_id,
            "cargo_type":         cargo_type,
            "cargo_weight_tons":  cargo_weight,
            "dispatch_hour":      dispatch_hour,
            "dispatch_dayofweek": dispatch_dow,
            "dispatch_month":     dispatch_month,
            "is_weekend":         int(is_weekend),
            "is_peak_season":     int(is_peak),
            "nlp_risk_override":  nlp_override if nlp_override > 0 else None
        }

        with st.spinner("Running hybrid AI prediction..."):
            result = client.predict_delay(payload)

        if result and "error" not in result:
            _render_prediction_result(result, route_id)
        else:
            st.error(
                f"Prediction failed: "
                f"{result.get('error', 'Unknown error') if result else 'No response'}"
            )


def _render_prediction_result(result: dict, route_id: str):
    """Renders the prediction result cards."""
    delay_prob   = result["delay_probability"]
    delay_hours  = result["predicted_delay_hours"]
    risk_level   = result["risk_level"]
    confidence   = result["confidence"]
    xgb_prob     = result["xgb_delay_prob"]
    gnn_prob     = result["gnn_delay_prob"]
    nlp_risk     = result["nlp_risk_score"]

    risk_colors  = {
        "CRITICAL": "#e74c3c",
        "HIGH":     "#e67e22",
        "MEDIUM":   "#f1c40f",
        "LOW":      "#2ecc71",
        "MINIMAL":  "#3498db"
    }
    color = risk_colors.get(risk_level, "#3498db")

    st.divider()
    st.markdown("### 🎯 Prediction Result")

    # ---- Main result banner ----
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 2px solid {color};
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin: 10px 0;
    ">
        <div style="font-size:3rem;font-weight:bold;color:{color}">
            {delay_prob:.1%}
        </div>
        <div style="color:#8b949e;font-size:1.1rem;margin-top:4px">
            Delay Probability
        </div>
        <div style="
            display:inline-block;
            background:{color};
            color:white;
            padding:4px 16px;
            border-radius:20px;
            font-weight:bold;
            margin-top:10px;
        ">
            {risk_level} RISK
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Detail metrics ----
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Delay",  f"{delay_hours:.1f}h")
    m2.metric("Confidence",       f"{confidence:.1%}")
    m3.metric("XGBoost Prob",     f"{xgb_prob:.3f}")
    m4.metric("GNN Prob",         f"{gnn_prob:.3f}")

    # ---- Gauge chart ----
    fig = go.Figure(go.Indicator(
        mode   = "gauge+number+delta",
        value  = delay_prob * 100,
        title  = {
            "text": f"Route: {route_id}",
            "font": {"color": "#e6edf3", "size": 14}
        },
        number = {
            "suffix": "%",
            "font":   {"color": color, "size": 28}
        },
        delta  = {
            "reference": 43,
            "suffix":    "%",
            "relative":  False,
            "font":      {"size": 14}
        },
        gauge  = {
            "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
            "bar":  {"color": color},
            "bgcolor": "#161b22",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0,  15],  "color": "#21262d"},
                {"range": [15, 35],  "color": "#1a3a1a"},
                {"range": [35, 55],  "color": "#3a3a1a"},
                {"range": [55, 75],  "color": "#3a2a1a"},
                {"range": [75, 100], "color": "#3a1a1a"}
            ],
            "threshold": {
                "line":      {"color": "#ffffff", "width": 3},
                "thickness": 0.85,
                "value":     delay_prob * 100
            }
        }
    ))
    fig.update_layout(
        height        = 280,
        margin        = dict(l=20, r=20, t=60, b=20),
        paper_bgcolor = "#0d1117",
        font_color    = "#e6edf3"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Action recommendation ----
    st.markdown("### 💡 Recommended Action")
    if risk_level == "CRITICAL":
        st.error(
            "🚨 **IMMEDIATE ACTION REQUIRED**\n\n"
            f"Delay probability of {delay_prob:.1%} is CRITICAL. "
            "Recommend:\n"
            "1. Immediately reroute to alternative port\n"
            "2. Notify all downstream warehouses\n"
            "3. Activate backup inventory protocol\n"
            "4. Contact carrier for emergency diversion"
        )
    elif risk_level == "HIGH":
        st.warning(
            "⚠️ **HIGH RISK — Proactive Action Recommended**\n\n"
            f"Delay probability of {delay_prob:.1%}. Recommend:\n"
            "1. Book alternative route as contingency\n"
            "2. Alert downstream warehouse to expect delay\n"
            "3. Check NLP risk feed for latest updates"
        )
    elif risk_level == "MEDIUM":
        st.info(
            "📋 **MEDIUM RISK — Monitor Closely**\n\n"
            f"Delay probability of {delay_prob:.1%}. Recommend:\n"
            "1. Monitor route for next 24 hours\n"
            "2. Prepare contingency routing plan\n"
            "3. Ensure buffer stock at destination"
        )
    else:
        st.success(
            f"✅ **LOW RISK — Normal Operations**\n\n"
            f"Delay probability of {delay_prob:.1%}. "
            "Shipment expected to arrive on schedule."
        )

    # ---- Model breakdown ----
    with st.expander("🔬 Model Breakdown Details"):
        st.markdown(f"""
        | Component | Score | Weight |
        |---|---|---|
        | XGBoost Classifier | {xgb_prob:.4f} | 55% |
        | Graph Neural Network | {gnn_prob:.4f} | 45% |
        | **Ensemble Combined** | **{delay_prob:.4f}** | **Final** |
        | NLP Risk Score | {nlp_risk:.4f} | Booster |
        | Confidence (Agreement) | {confidence:.4f} | — |
        """)
        st.caption(
            "Confidence = 1 - |XGBoost_prob - GNN_prob|. "
            "Higher confidence = both models agree."
        )


# ============================================
# PAGE 4: ROUTE RECOMMENDER
# ============================================
def page_route_recommender():
    st.markdown("## 🗺️ Alternative Route Recommender")
    st.markdown(
        "Find the safest alternative shipping routes using "
        "risk-adjusted Dijkstra's algorithm on the supply chain graph."
    )

    if not server_online:
        st.error("❌ API server offline.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 🎛️ Route Query")

        location_options = {
            "PORT_SHA": "Shanghai Port",
            "PORT_LAX": "Port of Los Angeles",
            "PORT_RTM": "Port of Rotterdam",
            "PORT_SIN": "Port of Singapore",
            "PORT_DXB": "Jebel Ali Port (Dubai)",
            "PORT_HKG": "Port of Hong Kong",
            "PORT_ANT": "Port of Antwerp",
            "WH_CHI":   "Chicago Distribution Center"
        }

        source = st.selectbox(
            "Source Port",
            options      = list(location_options.keys()),
            format_func  = lambda x: f"{location_options[x]} ({x})",
            index        = 0
        )
        dest = st.selectbox(
            "Destination Port",
            options      = list(location_options.keys()),
            format_func  = lambda x: f"{location_options[x]} ({x})",
            index        = 1
        )
        max_alts = st.slider(
            "Max Alternatives",
            min_value = 1,
            max_value = 5,
            value     = 3
        )
        avoid_risk = st.toggle(
            "Avoid High-Risk Ports",
            value = True,
            help  = "Excludes ports with risk score > 0.75"
        )

        search_btn = st.button(
            "🔍 Find Safest Routes",
            use_container_width = True,
            type                = "primary",
            disabled            = source == dest
        )

        if source == dest:
            st.warning("Source and destination must be different.")

    with col2:
        if search_btn and source != dest:
            payload = {
                "source_location_id": source,
                "dest_location_id":   dest,
                "max_alternatives":   max_alts,
                "avoid_high_risk":    avoid_risk
            }

            with st.spinner("Calculating optimal routes..."):
                result = client.get_route_recommendations(payload)

            if result and "error" not in result:
                _render_route_recommendations(result, location_options)
            else:
                err = result.get("error", "Unknown") if result else "No response"
                st.error(f"Route search failed: {err}")
        else:
            st.markdown("#### 👈 Configure query and click Search")
            st.info(
                "The Route Recommender uses **risk-adjusted Dijkstra's algorithm** "
                "on the supply chain graph.\n\n"
                "Routes are scored by:\n"
                "- Average NLP risk score of endpoint ports\n"
                "- Route duration (hours)\n"
                "- Estimated cost (USD)\n"
                "- Number of hops\n\n"
                "**Enable 'Avoid High-Risk Ports'** to automatically "
                "exclude nodes with risk score > 0.75 from path calculation."
            )


def _render_route_recommendations(result: dict, location_options: dict):
    """Renders route recommendation results."""
    alternatives = result.get("recommendations", [])
    query        = result.get("query", {})

    src_name = query.get("source_name", query.get("source", ""))
    dst_name = query.get("destination_name", query.get("destination", ""))

    st.markdown(
        f"#### Routes: **{src_name}** → **{dst_name}**"
    )
    st.caption(
        f"Found {result.get('alternatives_found', 0)} alternatives "
        f"| Avoided: {query.get('high_risk_avoided', [])}"
    )

    if not alternatives:
        st.warning(
            "No alternative routes found. "
            "Try disabling 'Avoid High-Risk Ports'."
        )
        return

    # ---- Route comparison chart ----
    chart_data = pd.DataFrame([{
        "Rank":       f"Route #{a['rank']}",
        "Risk Score": a["total_risk_score"],
        "Duration":   a["total_duration_hours"],
        "Cost":       a["total_cost"],
        "Risk Level": a["risk_level"],
        "Path":       " → ".join(a["path_names"])
    } for a in alternatives])

    fig = go.Figure()
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#9b59b6"]

    for i, row in chart_data.iterrows():
        fig.add_trace(go.Bar(
            name  = row["Rank"],
            x     = ["Risk Score", "Duration (÷100h)", "Cost (÷\$10k)"],
            y     = [
                row["Risk Score"],
                row["Duration"] / 100,
                row["Cost"] / 10000
            ],
            marker_color = colors[i % len(colors)],
            text  = [
                f"{row['Risk Score']:.3f}",
                f"{row['Duration']:.0f}h",
                f"${row['Cost']:,.0f}"
            ],
            textposition = "outside"
        ))

    fig.update_layout(
        barmode       = "group",
        title         = "Route Comparison (lower = better for all metrics)",
        paper_bgcolor = "#0d1117",
        plot_bgcolor  = "#0d1117",
        font_color    = "#e6edf3",
        legend        = dict(
            bgcolor    = "#161b22",
            bordercolor = "#30363d"
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Individual route cards ----
    st.markdown("#### 📋 Route Details")
    for alt in alternatives:
        risk_colors_map = {
            "CRITICAL": "#e74c3c",
            "HIGH":     "#e67e22",
            "MEDIUM":   "#f1c40f",
            "LOW":      "#2ecc71",
            "MINIMAL":  "#3498db"
        }
        card_color = risk_colors_map.get(alt["risk_level"], "#3498db")

        with st.expander(
            f"🏆 Rank #{alt['rank']} — Risk: {alt['total_risk_score']:.4f} "
            f"[{alt['risk_level']}] | "
            f"Duration: {alt['total_duration_hours']:.0f}h | "
            f"Cost: ${alt['total_cost']:,.0f}"
        ):
            st.markdown(
                f"**Full Path:** "
                f"{' → '.join(alt['path_names'])}"
            )
            st.markdown(f"**Hops:** {alt['hops']}")

            # Segment breakdown
            if alt.get("segments"):
                seg_df = pd.DataFrame(alt["segments"])
                st.dataframe(
                    seg_df,
                    use_container_width = True,
                    hide_index          = True
                )


# ============================================
# PAGE 5: BUSINESS ROI
# ============================================
def page_business_roi():
    st.markdown("## 💰 Business ROI Calculator")
    st.markdown(
        "Quantify the financial impact of deploying the "
        "Supply Chain AI system in your organization."
    )

    # ---- Load existing ROI results ----
    report_path = "data/processed/evaluation_report.json"
    saved_roi   = {}
    if os.path.exists(report_path):
        with open(report_path) as f:
            report    = json.load(f)
        saved_roi     = report.get("roi_results", {})

    st.markdown("### 🏭 Your Organization Parameters")

    col1, col2 = st.columns(2)

    with col1:
        monthly_vol = st.number_input(
            "Monthly Shipment Volume",
            min_value = 50,
            max_value = 10000,
            value     = int(saved_roi.get("monthly_shipments_estimated", 833)),
            step      = 50,
            help      = "Number of shipments per month"
        )
        delay_cost_hr = st.number_input(
            "Delay Cost per Hour (USD)",
            min_value = 500,
            max_value = 10000,
            value     = 2016,
            step      = 100,
            help      = "Industry average: \$1,500-\$3,500/hour (Gartner 2023)"
        )
        gulf_pct = st.slider(
            "% Routes Through Gulf Region",
            min_value = 0,
            max_value = 100,
            value     = 30,
            help      = "Routes affected by 2026 conflict premium"
        )

    with col2:
        delay_rate = st.slider(
            "Estimated Delay Rate (%)",
            min_value = 10,
            max_value = 90,
            value     = 43,
            help      = "What % of shipments get delayed? Dataset: 81%"
        )
        model_recall = st.slider(
            "Model Recall (from evaluation)",
            min_value = 0.50,
            max_value = 1.00,
            value     = float(
                report.get("binary_classifier_metrics", {}).get("recall", 0.894)
            ),
            step      = 0.01
        )
        reroute_cost = st.number_input(
            "Cost to Reroute One Shipment (USD)",
            min_value = 1000,
            max_value = 50000,
            value     = 6500,
            step      = 500
        )

    calculate = st.button(
        "💰 Calculate ROI",
        use_container_width = True,
        type                = "primary"
    )

    if calculate or saved_roi:
        # ---- ROI Calculation ----
        conflict_premium     = 1 + (gulf_pct / 100) * 0.40
        eff_delay_cost       = delay_cost_hr * conflict_premium
        estimated_delayed    = int(monthly_vol * delay_rate / 100)
        true_positives       = int(estimated_delayed * model_recall)
        precision            = float(
            report.get("binary_classifier_metrics", {}).get("precision", 0.927)
        )
        false_positives      = int(
            true_positives * (1 - precision) / max(precision, 0.01)
        )
        avg_delay_hrs        = 10.0
        hours_prevented      = avg_delay_hrs * 0.35

        cost_without         = estimated_delayed * avg_delay_hrs * eff_delay_cost
        cost_with            = (
            (estimated_delayed - true_positives) * avg_delay_hrs * eff_delay_cost +
            true_positives * (avg_delay_hrs - hours_prevented) * eff_delay_cost +
            false_positives * reroute_cost
        )
        monthly_savings      = max(0, cost_without - cost_with)
        annual_savings       = monthly_savings * 12

        st.divider()
        st.markdown("### 📊 ROI Results")

        # ---- Headline numbers ----
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Monthly Savings",     f"${monthly_savings:,.0f}")
        r2.metric("Annual Savings",      f"${annual_savings:,.0f}")
        r3.metric("Shipments Protected", f"{true_positives:,}/mo")
        r4.metric("False Alarms",        f"{false_positives:,}/mo")

        st.divider()

        # ---- Cost comparison chart ----
        cost_col, breakdown_col = st.columns(2)

        with cost_col:
            fig_cost = go.Figure(go.Bar(
                x             = ["Without AI Model", "With AI Model"],
                y             = [cost_without, cost_with],
                marker_color  = ["#e74c3c", "#2ecc71"],
                text          = [
                    f"${cost_without:,.0f}",
                    f"${cost_with:,.0f}"
                ],
                textposition  = "outside",
                textfont      = {"color": "white", "size": 13}
            ))
            fig_cost.add_annotation(
                x          = 1,
                y          = max(cost_without, cost_with) * 0.5,
                text       = f"Savings:<br><b>${monthly_savings:,.0f}/mo</b>",
                showarrow  = False,
                font       = {"color": "#2ecc71", "size": 14},
                bgcolor    = "rgba(22,27,34,0.8)",
                bordercolor = "#2ecc71",
                borderwidth = 1
            )
            fig_cost.update_layout(
                title         = "Monthly Delay Cost: With vs Without AI",
                paper_bgcolor = "#0d1117",
                plot_bgcolor  = "#0d1117",
                font_color    = "#e6edf3",
                yaxis_title   = "Monthly Cost (USD)",
                showlegend    = False,
                yaxis         = dict(tickformat="$,.0f")
            )
            st.plotly_chart(fig_cost, use_container_width=True)

        with breakdown_col:
            # ---- Annual projection pie ----
            fig_pie = go.Figure(go.Pie(
                labels = [
                    "Remaining Delay Costs",
                    "Savings from AI",
                    "False Alarm Costs"
                ],
                values = [
                    max(0, cost_with - false_positives * reroute_cost),
                    monthly_savings,
                    false_positives * reroute_cost
                ],
                marker_colors = ["#e74c3c", "#2ecc71", "#f1c40f"],
                hole          = 0.4,
                textinfo      = "label+percent",
                textfont      = {"color": "white", "size": 11}
            ))
            fig_pie.update_layout(
                title         = "Monthly Cost Breakdown",
                paper_bgcolor = "#0d1117",
                font_color    = "#e6edf3",
                legend        = dict(
                    bgcolor     = "#161b22",
                    bordercolor = "#30363d",
                    font        = dict(color="white")
                )
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ---- 12-month projection chart ----
        st.markdown("### 📈 12-Month Cumulative Savings Projection")

        months        = list(range(1, 13))
        cumulative    = [monthly_savings * m for m in months]
        month_labels  = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]

        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(
            x           = month_labels,
            y           = cumulative,
            mode        = "lines+markers+text",
            line        = dict(color="#2ecc71", width=3),
            marker      = dict(size=8, color="#2ecc71"),
            fill        = "tozeroy",
            fillcolor   = "rgba(46,204,113,0.15)",
            text        = [f"${v:,.0f}" for v in cumulative],
            textposition = "top center",
            textfont    = dict(color="white", size=10),
            name        = "Cumulative Savings"
        ))
        fig_proj.add_hline(
            y           = annual_savings / 2,
            line_dash   = "dash",
            line_color  = "#58a6ff",
            annotation_text = f"Mid-Year: ${annual_savings/2:,.0f}",
            annotation_font_color = "#58a6ff"
        )
        fig_proj.update_layout(
            paper_bgcolor = "#0d1117",
            plot_bgcolor  = "#0d1117",
            font_color    = "#e6edf3",
            xaxis_title   = "Month",
            yaxis_title   = "Cumulative Savings (USD)",
            yaxis         = dict(tickformat="$,.0f"),
            showlegend    = False,
            height        = 350
        )
        st.plotly_chart(fig_proj, use_container_width=True)

        # ---- Detailed breakdown table ----
        st.markdown("### 📋 Detailed ROI Breakdown")

        breakdown_data = {
            "Category": [
                "Monthly Shipment Volume",
                "Estimated Delayed Shipments",
                "Correctly Flagged (True Positives)",
                "False Alarms (False Positives)",
                "Hours Prevented per Shipment",
                "Gulf Conflict Route Premium",
                "Effective Delay Cost per Hour",
                "Cost WITHOUT AI Model (monthly)",
                "Cost WITH AI Model (monthly)",
                "Monthly Net Savings",
                "Annual Projected Savings"
            ],
            "Value": [
                f"{monthly_vol:,} shipments",
                f"{estimated_delayed:,} shipments ({delay_rate}%)",
                f"{true_positives:,} shipments",
                f"{false_positives:,} shipments",
                f"{hours_prevented:.2f} hours",
                f"{gulf_pct}% routes × 40% surcharge",
                f"${eff_delay_cost:,.2f}/hour",
                f"${cost_without:,.2f}",
                f"${cost_with:,.2f}",
                f"${monthly_savings:,.2f}",
                f"${annual_savings:,.2f}"
            ],
            "Notes": [
                "Your input",
                f"Based on {delay_rate}% delay rate",
                f"Model recall = {model_recall:.1%}",
                f"Model precision = {precision:.1%}",
                "30% delay hours prevented by early warning",
                "2026 US-Israel-Iran conflict active",
                f"Base ${delay_cost_hr:,} × {conflict_premium:.2f} premium",
                "All delayed shipments × avg 10h × cost",
                "Reduced delays + rerouting costs",
                "Cost reduction from AI deployment",
                "Monthly savings × 12"
            ]
        }

        st.dataframe(
            pd.DataFrame(breakdown_data),
            use_container_width = True,
            hide_index          = True
        )

        # ---- Interview-ready summary box ----
        st.divider()
        st.markdown("### 🎤 Interview-Ready Summary")
        st.success(f"""
                        **Copy this into your resume / use in interviews:**

                        > *"The Supply Chain Resilience Optimizer achieves an AUC-ROC of 0.867
                        > and correctly flags {model_recall:.1%} of delayed shipments before they occur.
                        > For an operator running {monthly_vol:,} shipments per month with {gulf_pct}%
                        > of routes through the Gulf region (currently under 2026 conflict premium),
                        > conservative ROI modeling projects **${annual_savings:,.0f} in annual savings**
                        > using a \$2,016/hour delay cost benchmark from Gartner 2023.
                        > The 84.1% ordinal accuracy of the severity classifier directly maps to
                        > operations team response protocols: monitor, alert, or reroute."*
                                """)

# ============================================
# MAIN PAGE ROUTER
# ============================================
def main():
    """Routes to the correct page based on sidebar selection."""

    if page == "🌍 Live Network Map":
        page_live_network_map()

    elif page == "📊 Analytics & Metrics":
        page_analytics()

    elif page == "🔮 Delay Predictor":
        page_delay_predictor()

    elif page == "🗺️ Route Recommender":
        page_route_recommender()

    elif page == "💰 Business ROI":
        page_business_roi()

    # ---- Footer ----
    st.divider()
    st.markdown("""
    <div style="
        text-align: center;
        color: #484f58;
        font-size: 12px;
        padding: 10px;
    ">
        🚢 Supply Chain Resilience Optimizer v1.0.0 |
        Built with FastAPI + PyTorch GNN + XGBoost + FinBERT |
        2026 Gulf Conflict Risk Premium Active
    </div>
    """, unsafe_allow_html=True)


# ============================================
# ENTRY POINT
# ============================================
if __name__ == "__main__":
    main()