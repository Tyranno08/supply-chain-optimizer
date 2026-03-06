# src/dashboard/map_builder.py
# ============================================
# FOLIUM WORLD MAP BUILDER
# Creates interactive geographic visualizations
# of the supply chain network
# ============================================

import folium
import pandas as pd
import pickle
import os
from folium.plugins import MarkerCluster, MiniMap, Fullscreen
from typing import Optional

GRAPH_OBJECT_PATH = "data/processed/graph_object.gpickle"

# Risk level colors
RISK_COLORS = {
    "CRITICAL": "#e74c3c",
    "HIGH":     "#e67e22",
    "MEDIUM":   "#f1c40f",
    "LOW":      "#2ecc71",
    "MINIMAL":  "#3498db"
}

RISK_ICONS = {
    "CRITICAL": "exclamation-sign",
    "HIGH":     "warning-sign",
    "MEDIUM":   "info-sign",
    "LOW":      "ok-sign",
    "MINIMAL":  "ok-circle"
}


def build_risk_map(
    network_df:     Optional[pd.DataFrame] = None,
    show_routes:    bool = True
) -> folium.Map:
    """
    Builds an interactive Folium world map showing:
    - Port locations as colored markers
    - Risk level indicated by marker color
    - Shipping routes as lines between ports
    - Popup info on click: name, risk score, routes

    Args:
        network_df:  DataFrame from API /network-status
        show_routes: Whether to draw route lines

    Returns:
        folium.Map object ready for streamlit_folium
    """

    # ---- Initialize map centered on global trade center ----
    world_map = folium.Map(
        location        = [20, 60],
        zoom_start      = 3,
        tiles           = "CartoDB dark_matter",
        prefer_canvas   = True
    )

    # ---- Add map controls ----
    Fullscreen(position="topright").add_to(world_map)
    MiniMap(toggle_display=True).add_to(world_map)

    # ---- Load graph for coordinates ----
    graph = None
    if os.path.exists(GRAPH_OBJECT_PATH):
        with open(GRAPH_OBJECT_PATH, "rb") as f:
            graph = pickle.load(f)

    if graph is None and network_df is None:
        return world_map

    # ---- Add port markers ----
    added_nodes = set()

    if network_df is not None and not network_df.empty:
        for _, row in network_df.iterrows():
            node_id    = row.get("location_id",   "")
            name       = row.get("location_name", node_id)
            risk_score = float(row.get("nlp_risk_score", 0.0))
            risk_level = row.get("risk_level",    "LOW")
            conflict   = row.get("conflict_affected", False)

            # Get coordinates from graph
            lat, lon = _get_coordinates(graph, node_id)
            if lat is None:
                continue

            # Build popup HTML
            popup_html = _build_popup_html(row, risk_score, conflict)

            # Add marker
            folium.Marker(
                location  = [lat, lon],
                popup     = folium.Popup(popup_html, max_width=300),
                tooltip   = f"{name} | Risk: {risk_level}",
                icon      = folium.Icon(
                    color  = _risk_to_folium_color(risk_level),
                    icon   = RISK_ICONS.get(risk_level, "info-sign"),
                    prefix = "glyphicon"
                )
            ).add_to(world_map)

            # Add risk score circle
            folium.CircleMarker(
                location   = [lat, lon],
                radius     = 8 + risk_score * 20,
                color      = RISK_COLORS.get(risk_level, "#3498db"),
                fill       = True,
                fill_color = RISK_COLORS.get(risk_level, "#3498db"),
                fill_opacity = 0.35,
                weight     = 2
            ).add_to(world_map)

            added_nodes.add(node_id)

    elif graph is not None:
        # Fallback: use graph directly if API is unavailable
        for node_id, attrs in graph.nodes(data=True):
            lat  = attrs.get("latitude",   None)
            lon  = attrs.get("longitude",  None)
            name = attrs.get("name",       node_id)
            risk = attrs.get("risk_score", 0.0)

            if lat is None:
                continue

            risk_level = _score_to_risk_level(risk)
            folium.Marker(
                location = [lat, lon],
                tooltip  = f"{name} | Risk: {risk:.3f}",
                icon     = folium.Icon(
                    color  = _risk_to_folium_color(risk_level),
                    icon   = "ship",
                    prefix = "fa"
                )
            ).add_to(world_map)
            added_nodes.add(node_id)

    # ---- Draw shipping routes ----
    if show_routes and graph is not None:
        for src, dst, edge_data in graph.edges(data=True):
            src_lat, src_lon = _get_coordinates(graph, src)
            dst_lat, dst_lon = _get_coordinates(graph, dst)

            if src_lat is None or dst_lat is None:
                continue

            mode     = edge_data.get("transport_mode", "Sea")
            duration = edge_data.get("duration_hours",  100)
            cost     = edge_data.get("cost",            0)

            line_color = (
                "#3498db" if mode == "Sea"  else
                "#e67e22" if mode == "Air"  else
                "#2ecc71"
            )
            line_dash  = (
                "5,5"     if mode == "Air"  else
                "10,5"    if mode == "Land" else
                None
            )

            line = folium.PolyLine(
                locations   = [[src_lat, src_lon], [dst_lat, dst_lon]],
                color       = line_color,
                weight      = 2,
                opacity     = 0.6,
                tooltip     = (
                    f"{mode} Route | "
                    f"Duration: {duration}h | "
                    f"Cost: ${cost:,.0f}"
                ),
                dash_array  = line_dash
            )
            line.add_to(world_map)

            # Arrow at midpoint
            mid_lat = (src_lat + dst_lat) / 2
            mid_lon = (src_lon + dst_lon) / 2
            folium.Marker(
                location = [mid_lat, mid_lon],
                icon     = folium.DivIcon(
                    html = f"""
                        <div style="
                            color:{line_color};
                            font-size:10px;
                            font-weight:bold;
                            background:rgba(0,0,0,0.6);
                            padding:2px 4px;
                            border-radius:3px;
                        ">{mode}</div>
                    """,
                    icon_size   = (40, 15),
                    icon_anchor = (20, 7)
                )
            ).add_to(world_map)

    # ---- Add legend ----
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px; right: 15px;
        background-color: rgba(13,17,23,0.9);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: monospace;
        font-size: 12px;
        color: white;
        z-index: 9999;
    ">
        <b>🚢 Risk Level Legend</b><br><br>
        <span style="color:#e74c3c">●</span> CRITICAL (>0.75)<br>
        <span style="color:#e67e22">●</span> HIGH     (>0.55)<br>
        <span style="color:#f1c40f">●</span> MEDIUM   (>0.35)<br>
        <span style="color:#2ecc71">●</span> LOW      (>0.15)<br>
        <span style="color:#3498db">●</span> MINIMAL  (<0.15)<br>
        <br>
        <b>Route Lines:</b><br>
        <span style="color:#3498db">━━</span> Sea<br>
        <span style="color:#e67e22">- -</span> Air<br>
        <span style="color:#2ecc71">╌╌</span> Land<br>
        <br>
        <small>⚠️ Gulf conflict active 2026</small>
    </div>
    """
    world_map.get_root().html.add_child(folium.Element(legend_html))

    return world_map


def _get_coordinates(graph, node_id: str):
    """Safely extracts lat/lon from graph node."""
    if graph is None or node_id not in graph.nodes:
        return None, None
    attrs = graph.nodes[node_id]
    return attrs.get("latitude"), attrs.get("longitude")


def _build_popup_html(row: pd.Series, risk_score: float, conflict: bool) -> str:
    """Builds styled HTML popup for a port marker."""
    name      = row.get("location_name",  "Unknown")
    risk_lvl  = row.get("risk_level",     "LOW")
    in_deg    = row.get("in_degree",      0)
    out_deg   = row.get("out_degree",     0)
    btwn      = row.get("betweenness_centrality", 0)
    color     = RISK_COLORS.get(risk_lvl, "#3498db")

    conflict_badge = (
        '<br><span style="background:#e74c3c;padding:2px 6px;'
        'border-radius:3px;font-size:10px;">⚠️ CONFLICT ZONE</span>'
        if conflict else ""
    )

    return f"""
    <div style="
        font-family: monospace;
        background: #0d1117;
        color: white;
        padding: 10px;
        border-radius: 6px;
        min-width: 200px;
    ">
        <b style="font-size:14px">{name}</b>
        {conflict_badge}
        <hr style="border-color:#30363d;margin:6px 0">
        <table style="width:100%;font-size:12px">
            <tr>
                <td>Risk Score:</td>
                <td><b style="color:{color}">{risk_score:.4f}</b></td>
            </tr>
            <tr>
                <td>Risk Level:</td>
                <td><b style="color:{color}">{risk_lvl}</b></td>
            </tr>
            <tr>
                <td>Incoming Routes:</td>
                <td>{in_deg}</td>
            </tr>
            <tr>
                <td>Outgoing Routes:</td>
                <td>{out_deg}</td>
            </tr>
            <tr>
                <td>Betweenness:</td>
                <td>{btwn:.4f}</td>
            </tr>
        </table>
    </div>
    """


def _risk_to_folium_color(risk_level: str) -> str:
    """Converts risk level to Folium marker color name."""
    mapping = {
        "CRITICAL": "red",
        "HIGH":     "orange",
        "MEDIUM":   "beige",
        "LOW":      "green",
        "MINIMAL":  "blue"
    }
    return mapping.get(risk_level, "blue")


def _score_to_risk_level(score: float) -> str:
    if score >= 0.75:   return "CRITICAL"
    elif score >= 0.55: return "HIGH"
    elif score >= 0.35: return "MEDIUM"
    elif score >= 0.15: return "LOW"
    else:               return "MINIMAL"