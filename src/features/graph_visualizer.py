# src/features/graph_visualizer.py
# ============================================
# SUPPLY CHAIN GRAPH VISUALIZER
# Generates portfolio-ready network visualizations
# ============================================

import os
import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import get_logger

logger = get_logger("GraphVisualizer")

VIZ_DIR = "data/processed/visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)


class SupplyChainVisualizer:

    def __init__(self, G: nx.DiGraph):
        self.G = G

    # ============================================
    # VISUALIZATION 1: RISK HEATMAP NETWORK
    # ============================================
    def plot_risk_heatmap(self) -> None:
        """
        Main network visualization where:
        - Node COLOR represents NLP Risk Score
          (Red = High Risk, Green = Low Risk)
        - Node SIZE represents Port Capacity
        - Edge THICKNESS represents route duration
        - Edge COLOR represents transport mode
        """
        logger.info("Generating Risk Heatmap Network visualization...")

        fig, ax = plt.subplots(figsize=(18, 12))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        # ---- Use geographic positions ----
        pos = {
            node: (
                attrs["longitude"],
                attrs["latitude"]
            )
            for node, attrs in self.G.nodes(data=True)
        }

        # ---- Node colors from risk score ----
        risk_scores  = [
            self.G.nodes[n].get("risk_score", 0.0)
            for n in self.G.nodes()
        ]
        node_colors  = risk_scores

        # ---- Node sizes from capacity ----
        capacities   = [
            self.G.nodes[n].get("base_capacity", 1000)
            for n in self.G.nodes()
        ]
        max_cap      = max(capacities) if capacities else 1
        node_sizes   = [
            (cap / max_cap) * 3000 + 500
            for cap in capacities
        ]

        # ---- Edge colors by transport mode ----
        edge_colors  = []
        edge_widths  = []
        for u, v, data in self.G.edges(data=True):
            mode = data.get("transport_mode", "Sea")
            if mode == "Sea":
                edge_colors.append("#1f77b4")   # Blue
            elif mode == "Air":
                edge_colors.append("#ff7f0e")   # Orange
            else:
                edge_colors.append("#2ca02c")   # Green (Land)

            duration = data.get("duration_hours", 100)
            edge_widths.append(max(0.5, 5.0 - duration / 100))

        # ---- Draw edges ----
        nx.draw_networkx_edges(
            self.G, pos,
            edge_color   = edge_colors,
            width        = edge_widths,
            arrowsize    = 20,
            arrowstyle   = "->",
            ax           = ax,
            connectionstyle = "arc3,rad=0.1"
        )

        # ---- Draw nodes ----
        cmap = plt.cm.RdYlGn_r
        nodes = nx.draw_networkx_nodes(
            self.G, pos,
            node_color   = node_colors,
            node_size    = node_sizes,
            cmap         = cmap,
            vmin         = 0.0,
            vmax         = 1.0,
            ax           = ax,
            edgecolors   = "white",
            linewidths   = 1.5
        )

        # ---- Draw labels ----
        labels = {
            node: f"{attrs['name'].split()[2] if len(attrs['name'].split()) > 2 else attrs['name']}\n"
                  f"Risk: {attrs.get('risk_score', 0):.2f}"
            for node, attrs in self.G.nodes(data=True)
        }
        nx.draw_networkx_labels(
            self.G, pos,
            labels    = labels,
            font_size = 8,
            font_color = "white",
            font_weight = "bold",
            ax        = ax
        )

        # ---- Colorbar ----
        sm = plt.cm.ScalarMappable(
            cmap  = cmap,
            norm  = plt.Normalize(vmin=0.0, vmax=1.0)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(
            "NLP Risk Score  (0 = Safe  |  1 = Critical)",
            color     = "white",
            fontsize  = 11
        )
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        # ---- Legend for transport modes ----
        legend_elements = [
            Line2D([0], [0], color="#1f77b4", linewidth=3,
                   label="Sea Route"),
            Line2D([0], [0], color="#ff7f0e", linewidth=3,
                   label="Air Route"),
            Line2D([0], [0], color="#2ca02c", linewidth=3,
                   label="Land Route"),
        ]
        ax.legend(
            handles    = legend_elements,
            loc        = "lower left",
            facecolor  = "#161b22",
            edgecolor  = "white",
            labelcolor = "white",
            fontsize   = 10
        )

        ax.set_title(
            "Global Supply Chain Network — NLP Risk Heatmap\n"
            "Node Color = Risk Level  |  Node Size = Port Capacity  |"
            "  Edge Color = Transport Mode",
            color    = "white",
            fontsize = 14,
            pad      = 20
        )
        ax.axis("off")

        output_path = f"{VIZ_DIR}/01_risk_heatmap_network.png"
        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi             = 150,
            bbox_inches     = "tight",
            facecolor       = "#0d1117"
        )
        plt.close()
        logger.info(f"Risk Heatmap saved : {output_path}")

    # ============================================
    # VISUALIZATION 2: CENTRALITY ANALYSIS
    # ============================================
    def plot_centrality_analysis(self) -> None:
        """
        Bar charts showing which ports are most
        critical to the network by centrality metrics.
        """
        logger.info("Generating Centrality Analysis visualization...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor("#0d1117")

        centrality_metrics = {
            "Betweenness Centrality\n(Critical Connectors)": "betweenness_centrality",
            "Degree Centrality\n(Most Connected)":          "degree_centrality",
            "Closeness Centrality\n(Fastest to Reach All)": "closeness_centrality"
        }

        for ax, (title, metric) in zip(axes, centrality_metrics.items()):
            ax.set_facecolor("#161b22")

            node_names  = [
                self.G.nodes[n].get("name", n)
                for n in self.G.nodes()
            ]
            values      = [
                self.G.nodes[n].get(metric, 0)
                for n in self.G.nodes()
            ]
            risk_scores = [
                self.G.nodes[n].get("risk_score", 0)
                for n in self.G.nodes()
            ]

            # Sort by metric value
            sorted_pairs = sorted(
                zip(node_names, values, risk_scores),
                key=lambda x: x[1],
                reverse=True
            )
            names, vals, risks = zip(*sorted_pairs)

            # Color bars by risk score
            bar_colors = [
                plt.cm.RdYlGn_r(r) for r in risks
            ]

            bars = ax.barh(
                range(len(names)),
                vals,
                color  = bar_colors,
                edgecolor = "white",
                linewidth = 0.5
            )

            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(
                [n.replace(" Port", "").replace("Port of ", "")
                 for n in names],
                color    = "white",
                fontsize = 9
            )
            ax.set_xlabel(metric.split("\n")[0], color="white", fontsize=10)
            ax.set_title(title, color="white", fontsize=11, pad=10)
            ax.tick_params(colors="white")
            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(
            "Supply Chain Network — Port Criticality Analysis",
            color    = "white",
            fontsize = 15,
            y        = 1.02
        )

        output_path = f"{VIZ_DIR}/02_centrality_analysis.png"
        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi         = 150,
            bbox_inches = "tight",
            facecolor   = "#0d1117"
        )
        plt.close()
        logger.info(f"Centrality Analysis saved : {output_path}")

    # ============================================
    # VISUALIZATION 3: DELAY DISTRIBUTION
    # ============================================
    def plot_delay_distribution(self, df: pd.DataFrame) -> None:
        """
        Shows delay patterns in the shipment data.
        """
        logger.info("Generating Delay Distribution visualization...")

        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor("#0d1117")

        # Plot 1: Delay distribution by transport mode
        ax1 = axes[0, 0]
        ax1.set_facecolor("#161b22")
        for mode in df["transport_mode"].unique():
            subset = df[df["transport_mode"] == mode]["actual_delay_hours"]
            ax1.hist(
                subset.clip(0, 100),
                bins      = 30,
                alpha     = 0.7,
                label     = mode,
                edgecolor = "white"
            )
        ax1.set_title("Delay Distribution by Transport Mode",
                      color="white", fontsize=12)
        ax1.set_xlabel("Delay Hours", color="white")
        ax1.set_ylabel("Frequency",   color="white")
        ax1.tick_params(colors="white")
        ax1.legend(facecolor="#0d1117", labelcolor="white")
        ax1.spines["bottom"].set_color("white")
        ax1.spines["left"].set_color("white")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # Plot 2: Monthly delay trend
        ax2 = axes[0, 1]
        ax2.set_facecolor("#161b22")
        monthly = (
            df.groupby("dispatch_month")["actual_delay_hours"]
            .mean()
            .reset_index()
        )
        ax2.plot(
            monthly["dispatch_month"],
            monthly["actual_delay_hours"],
            color     = "#1f77b4",
            linewidth = 2,
            marker    = "o",
            markersize = 6
        )
        ax2.fill_between(
            monthly["dispatch_month"],
            monthly["actual_delay_hours"],
            alpha = 0.3,
            color = "#1f77b4"
        )
        ax2.set_title("Average Monthly Delay Trend",
                      color="white", fontsize=12)
        ax2.set_xlabel("Month",          color="white")
        ax2.set_ylabel("Avg Delay Hours", color="white")
        ax2.tick_params(colors="white")
        ax2.spines["bottom"].set_color("white")
        ax2.spines["left"].set_color("white")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # Plot 3: Delay rate by route
        ax3 = axes[1, 0]
        ax3.set_facecolor("#161b22")
        route_delay = (
            df.groupby("route_id")["delay_flag"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        colors_route = [
            "#e74c3c" if v > 0.5 else
            "#f39c12" if v > 0.3 else
            "#2ecc71"
            for v in route_delay["delay_flag"]
        ]
        ax3.bar(
            range(len(route_delay)),
            route_delay["delay_flag"],
            color     = colors_route,
            edgecolor = "white",
            linewidth = 0.5
        )
        ax3.set_xticks(range(len(route_delay)))
        ax3.set_xticklabels(
            route_delay["route_id"],
            rotation  = 45,
            ha        = "right",
            color     = "white",
            fontsize  = 8
        )
        ax3.set_title("Delay Rate by Route (Red > 50%)",
                      color="white", fontsize=12)
        ax3.set_ylabel("Delay Rate", color="white")
        ax3.tick_params(colors="white")
        ax3.spines["bottom"].set_color("white")
        ax3.spines["left"].set_color("white")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        # Plot 4: Risk score vs delay correlation
        ax4 = axes[1, 1]
        ax4.set_facecolor("#161b22")
        scatter_df = df[df["combined_route_risk"] > 0].copy()
        if not scatter_df.empty:
            sc = ax4.scatter(
                scatter_df["combined_route_risk"],
                scatter_df["actual_delay_hours"].clip(0, 100),
                c         = scatter_df["combined_route_risk"],
                cmap      = plt.cm.RdYlGn_r,
                alpha     = 0.6,
                s         = 20,
                edgecolors = "none"
            )
            plt.colorbar(sc, ax=ax4).ax.yaxis.set_tick_params(color="white")
        ax4.set_title("NLP Risk Score vs Actual Delay",
                      color="white", fontsize=12)
        ax4.set_xlabel("Combined Route Risk Score", color="white")
        ax4.set_ylabel("Actual Delay Hours",        color="white")
        ax4.tick_params(colors="white")
        ax4.spines["bottom"].set_color("white")
        ax4.spines["left"].set_color("white")
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)

        output_path = f"{VIZ_DIR}/03_delay_distribution.png"
        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi         = 150,
            bbox_inches = "tight",
            facecolor   = "#0d1117"
        )
        plt.close()
        logger.info(f"Delay Distribution saved : {output_path}")