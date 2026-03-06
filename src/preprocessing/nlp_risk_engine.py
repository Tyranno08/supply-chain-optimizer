# src/preprocessing/nlp_risk_engine.py
# ============================================
# NLP RISK SCORING ENGINE
# Uses Hugging Face FinBERT to convert raw
# text headlines into numerical risk scores
# ============================================

import os
import sys
import time
import torch
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import text
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.db_connector import get_engine
from src.utils import get_logger

load_dotenv()
logger = get_logger("NLPRiskEngine")

# ============================================
# CONFIGURATION
# ============================================
# FinBERT: BERT fine-tuned on financial news
# Perfect for supply chain & logistics text
MODEL_NAME  = "ProsusAI/finbert"
BATCH_SIZE  = 16      # Process 16 headlines at a time
MAX_TOKENS  = 512     # BERT's maximum token limit


# ============================================
# RISK SCORE LOGIC
# ============================================
# FinBERT outputs 3 labels: positive, negative, neutral
# We convert these to a 0.0 - 1.0 risk score

LABEL_TO_RISK = {
    "negative": 1.0,   # Base multiplier for negative sentiment
    "neutral":  0.5,   # Base multiplier for neutral sentiment
    "positive": 0.0    # Base multiplier for positive sentiment
}

# Supply chain specific high-risk keywords
# If these appear in the headline, we boost the risk score
HIGH_RISK_KEYWORDS = [
    "strike", "storm", "hurricane", "typhoon", "flood",
    "blockage", "closure", "delay", "congestion", "shortage",
    "disruption", "halt", "suspend", "ban", "sanction",
    "accident", "fire", "explosion", "earthquake", "war",
    "conflict", "protest", "lockdown", "pandemic"
]

LOW_RISK_KEYWORDS = [
    "normal", "clear", "smooth", "record", "efficient",
    "improved", "resuming", "open", "operating", "stable"
]


class NLPRiskEngine:
    """
    Production-grade NLP pipeline for supply chain
    risk scoring using Hugging Face FinBERT.
    """

    def __init__(self):
        self.engine    = get_engine()
        self.model     = None
        self.tokenizer = None
        self.nlp_pipe  = None
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================
    # MODEL LOADING
    # ============================================
    def load_model(self) -> None:
        """
        Downloads and loads FinBERT from Hugging Face.
        Model is cached locally after first download
        so subsequent runs are instant.
        """
        logger.info(f"Loading FinBERT model: {MODEL_NAME}")
        logger.info(f"Using device: {self.device.upper()}")
        logger.info("First run will download ~440MB model. "
                    "Subsequent runs will use local cache...")

        start_time = time.time()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model     = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME
            )

            # Create the Hugging Face pipeline
            self.nlp_pipe = pipeline(
                task            = "text-classification",
                model           = self.model,
                tokenizer       = self.tokenizer,
                device          = 0 if self.device == "cuda" else -1,
                return_all_scores = True   # We want scores for ALL 3 labels
            )

            elapsed = round(time.time() - start_time, 2)
            logger.info(f"FinBERT model loaded successfully in {elapsed}s")

        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            logger.info("Falling back to DistilBERT general sentiment model...")
            self._load_fallback_model()

    def _load_fallback_model(self) -> None:
        """
        Fallback to DistilBERT if FinBERT fails.
        Lighter model, still effective for our purpose.
        """
        FALLBACK_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
        logger.info(f"Loading fallback model: {FALLBACK_MODEL}")

        self.nlp_pipe = pipeline(
            task      = "text-classification",
            model     = FALLBACK_MODEL,
            device    = 0 if self.device == "cuda" else -1,
            return_all_scores = True
        )
        logger.info("Fallback model loaded successfully.")

    # ============================================
    # RISK SCORE CALCULATION
    # ============================================
    def calculate_risk_score(
        self,
        headline:       str,
        model_output:   list
    ) -> tuple:
        """
        Converts FinBERT output into a final Risk Score (0.0 - 1.0).

        FinBERT returns scores like:
        [
            {'label': 'positive', 'score': 0.02},
            {'label': 'negative', 'score': 0.94},
            {'label': 'neutral',  'score': 0.04}
        ]

        Our formula:
        base_score  = weighted sum using LABEL_TO_RISK multipliers
        keyword_adj = boost/reduce based on supply chain keywords
        final_score = clip(base_score + keyword_adj, 0.0, 1.0)

        Returns: (risk_score, sentiment_label)
        """
        # Step 1: Convert model output to a dict for easy access
        scores = {item["label"].lower(): item["score"] for item in model_output}

        # Step 2: Calculate base weighted risk score
        base_score = (
            scores.get("negative", 0.0) * LABEL_TO_RISK["negative"] +
            scores.get("neutral",  0.0) * LABEL_TO_RISK["neutral"]  +
            scores.get("positive", 0.0) * LABEL_TO_RISK["positive"]
        )

        # Step 3: Keyword-based adjustment
        headline_lower  = headline.lower()
        keyword_boost   = 0.0

        for keyword in HIGH_RISK_KEYWORDS:
            if keyword in headline_lower:
                keyword_boost += 0.08   # Each high-risk keyword adds 8%
                logger.debug(f"High-risk keyword found: '{keyword}' "
                             f"(+0.08 boost)")

        for keyword in LOW_RISK_KEYWORDS:
            if keyword in headline_lower:
                keyword_boost -= 0.05   # Each low-risk keyword reduces 5%
                logger.debug(f"Low-risk keyword found: '{keyword}' "
                             f"(-0.05 reduction)")

        # Step 4: Final score — clipped to valid range [0.0, 1.0]
        final_score = round(
            min(max(base_score + keyword_boost, 0.0), 1.0),
            4
        )

        # Step 5: Determine primary sentiment label
        dominant_label = max(scores, key=scores.get).upper()

        return final_score, dominant_label

    # ============================================
    # FETCH UNSCORED EVENTS FROM MYSQL
    # ============================================
    def fetch_unscored_events(self) -> pd.DataFrame:
        """
        Fetches all risk_events where nlp_risk_score IS NULL.
        These are events fetched in Phase 2 that haven't
        been processed by the NLP model yet.
        """
        logger.info("Fetching unscored events from MySQL...")

        query = text("""
            SELECT
                re.event_id,
                re.location_id,
                re.headline,
                re.event_type,
                re.source_api,
                l.location_name
            FROM risk_events re
            JOIN locations   l  ON re.location_id = l.location_id
            WHERE re.nlp_risk_score IS NULL
            ORDER BY re.event_date DESC
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query)
            rows   = result.fetchall()
            cols   = result.keys()

        df = pd.DataFrame(rows, columns=cols)
        logger.info(f"Found {len(df)} unscored events to process.")
        return df

    # ============================================
    # PROCESS EVENTS IN BATCHES
    # ============================================
    def process_events(self) -> None:
        """
        Main processing function.
        Fetches unscored events, runs FinBERT on them
        in batches, and updates the MySQL database.
        """
        # Fetch events that need scoring
        df = self.fetch_unscored_events()

        if df.empty:
            logger.info("No unscored events found. "
                        "All risk_events already have NLP scores.")
            return

        # Process in batches
        total       = len(df)
        processed   = 0
        failed      = 0

        logger.info(f"Starting NLP scoring for {total} events "
                    f"in batches of {BATCH_SIZE}...")

        for batch_start in range(0, total, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total)
            batch_df  = df.iloc[batch_start:batch_end]

            # Truncate headlines to MAX_TOKENS characters
            headlines = [
                str(h)[:MAX_TOKENS]
                for h in batch_df["headline"].tolist()
            ]

            try:
                # Run FinBERT on entire batch at once (much faster than one-by-one)
                batch_outputs = self.nlp_pipe(headlines)

                # Process each result in the batch
                for idx, (_, row) in enumerate(batch_df.iterrows()):
                    try:
                        model_output = batch_outputs[idx]
                        risk_score, sentiment_label = self.calculate_risk_score(
                            headline     = row["headline"],
                            model_output = model_output
                        )

                        # Update database
                        self._update_risk_score(
                            event_id        = row["event_id"],
                            risk_score      = risk_score,
                            sentiment_label = sentiment_label
                        )

                        processed += 1
                        logger.info(
                            f"[{processed}/{total}] "
                            f"Location: {row['location_name'][:20]:<20} | "
                            f"Type: {row['event_type']:<15} | "
                            f"Risk: {risk_score:.4f} | "
                            f"Sentiment: {sentiment_label} | "
                            f"Headline: {str(row['headline'])[:50]}..."
                        )

                    except Exception as row_err:
                        logger.warning(
                            f"Failed to process event_id "
                            f"{row['event_id']}: {row_err}"
                        )
                        failed += 1

            except Exception as batch_err:
                logger.error(f"Batch {batch_start}-{batch_end} failed: {batch_err}")
                failed += BATCH_SIZE

        logger.info(
            f"NLP Scoring complete. "
            f"Processed: {processed}, Failed: {failed}, Total: {total}"
        )

    # ============================================
    # UPDATE MYSQL WITH RISK SCORES
    # ============================================
    def _update_risk_score(
        self,
        event_id:        int,
        risk_score:      float,
        sentiment_label: str
    ) -> None:
        """
        Updates a single risk_event record with its
        computed NLP risk score and sentiment label.
        """
        with self.engine.connect() as conn:
            query = text("""
                UPDATE risk_events
                SET
                    nlp_risk_score  = :risk_score,
                    sentiment_label = :sentiment_label
                WHERE event_id = :event_id
            """)
            conn.execute(query, {
                "risk_score":      risk_score,
                "sentiment_label": sentiment_label,
                "event_id":        event_id
            })
            conn.commit()


# ============================================
# ROUTE RISK AGGREGATOR
# ============================================
class RouteRiskAggregator:
    """
    After individual events are scored, this class
    aggregates those scores at the ROUTE level.

    Logic:
    - Each route connects source → destination ports
    - Risk events at EITHER port affect the route
    - We compute a weighted average of recent event scores
    - Recent events are weighted MORE than older events
    """

    def __init__(self):
        self.engine = get_engine()

    def compute_route_risk_index(self) -> pd.DataFrame:
        """
        Computes a Risk Index for every route in our database
        based on NLP scores of events at the connected ports.

        Returns a DataFrame with route-level risk scores.
        """
        logger.info("Computing Route Risk Index from NLP scores...")

        query = text("""
            SELECT
                r.route_id,
                r.source_location_id,
                r.dest_location_id,
                r.transport_mode,
                l_src.location_name  AS source_name,
                l_dst.location_name  AS dest_name,
                re.event_id,
                re.event_date,
                re.event_type,
                re.nlp_risk_score,
                re.sentiment_label,
                -- Recency weight: events in last 7 days count double
                CASE
                    WHEN re.event_date >= NOW() - INTERVAL 7 DAY THEN 2.0
                    WHEN re.event_date >= NOW() - INTERVAL 30 DAY THEN 1.5
                    ELSE 1.0
                END AS recency_weight
            FROM routes        r
            JOIN locations     l_src ON r.source_location_id = l_src.location_id
            JOIN locations     l_dst ON r.dest_location_id   = l_dst.location_id
            JOIN risk_events   re    ON (
                re.location_id = r.source_location_id OR
                re.location_id = r.dest_location_id
            )
            WHERE re.nlp_risk_score IS NOT NULL
            ORDER BY r.route_id, re.event_date DESC
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query)
            rows   = result.fetchall()
            cols   = result.keys()

        if not rows:
            logger.warning("No scored events found. "
                           "Run NLPRiskEngine first.")
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=cols)

        # Compute weighted average risk score per route
        df["weighted_score"] = df["nlp_risk_score"] * df["recency_weight"]

        route_risk = (
            df.groupby([
                "route_id",
                "source_name",
                "dest_name",
                "transport_mode"
            ])
            .agg(
                total_events        = ("event_id",       "count"),
                avg_risk_score      = ("nlp_risk_score", "mean"),
                weighted_risk_score = ("weighted_score", "mean"),
                max_risk_score      = ("nlp_risk_score", "max"),
                negative_events     = ("sentiment_label",
                                       lambda x: (x == "NEGATIVE").sum()),
                latest_event_date   = ("event_date",     "max")
            )
            .reset_index()
        )

        # Normalize weighted score back to 0-1 range
        route_risk["weighted_risk_score"] = (
            pd.to_numeric(route_risk["weighted_risk_score"], errors='coerce') # Force to numbers
            .fillna(0.0)                                                     # Replace NaNs with 0
            .clip(0.0, 1.0)
            .round(4)
        )

        # Assign Risk Category for easy interpretation
        route_risk["risk_category"] = route_risk["weighted_risk_score"].apply(
            self._assign_risk_category
        )

        logger.info(
            f"Route Risk Index computed for "
            f"{len(route_risk)} routes."
        )

        return route_risk

    def _assign_risk_category(self, score: float) -> str:
        """
        Converts a numeric score to a business-readable label.
        """
        if score >= 0.75:
            return "CRITICAL"
        elif score >= 0.55:
            return "HIGH"
        elif score >= 0.35:
            return "MEDIUM"
        elif score >= 0.15:
            return "LOW"
        else:
            return "MINIMAL"

    def save_route_risk_to_csv(self, df: pd.DataFrame) -> None:
        """
        Saves the Route Risk Index to a CSV file in data/processed/.
        This will be used as input in Phase 4 Feature Engineering.
        """
        os.makedirs("data/processed", exist_ok=True)
        output_path = "data/processed/route_risk_index.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Route Risk Index saved to {output_path}")

    def print_risk_report(self, df: pd.DataFrame) -> None:
        """
        Prints a formatted risk summary report to the console.
        Shows which routes are most at risk right now.
        """
        logger.info("=" * 70)
        logger.info("       SUPPLY CHAIN ROUTE RISK REPORT")
        logger.info("=" * 70)

        # Sort by risk score descending
        df_sorted = df.sort_values(
            "weighted_risk_score",
            ascending=False
        )

        for _, row in df_sorted.iterrows():
            risk_bar = self._generate_risk_bar(row["weighted_risk_score"])
            logger.info(
                f"Route: {row['route_id']:<15} | "
                f"{row['source_name'][:15]:<15} -> "
                f"{row['dest_name'][:15]:<15} | "
                f"Risk: {row['weighted_risk_score']:.4f} "
                f"{risk_bar} [{row['risk_category']}] | "
                f"Events: {row['total_events']}"
            )

        logger.info("=" * 70)

        # Summary statistics
        critical_routes = len(df[df["risk_category"] == "CRITICAL"])
        high_routes     = len(df[df["risk_category"] == "HIGH"])
        logger.info(
            f"SUMMARY: {critical_routes} CRITICAL routes, "
            f"{high_routes} HIGH risk routes out of {len(df)} total."
        )
        logger.info("=" * 70)
        
    def _generate_risk_bar(self, score: float) -> str:
            """
            Generates a simple ASCII progress bar for risk visualization.
            Standard characters used to avoid Unicode errors on Windows.
            """
            filled = int(score * 10)
            # Using # and - ensures it works on all Windows consoles
            bar    = "#" * filled + "-" * (10 - filled)
            return f"[{bar}]"


# ============================================
# MAIN RUNNER
# ============================================
def run_nlp_pipeline() -> None:
    """
    Orchestrates the complete NLP Phase:
    1. Score all unscored events with FinBERT
    2. Aggregate scores at the route level
    3. Save the Route Risk Index for Phase 4
    """
    logger.info("=" * 60)
    logger.info("  NLP RISK SCORING ENGINE — STARTING")
    logger.info("=" * 60)

    # Step 1: Score individual events
    scorer = NLPRiskEngine()
    scorer.load_model()
    scorer.process_events()

    # Step 2: Aggregate scores per route
    aggregator  = RouteRiskAggregator()
    route_risk  = aggregator.compute_route_risk_index()

    if not route_risk.empty:
        # Step 3: Save and report
        aggregator.save_route_risk_to_csv(route_risk)
        aggregator.print_risk_report(route_risk)
    else:
        logger.warning(
            "Route Risk Index is empty. "
            "Check that risk_events table has scored records."
        )

    logger.info("=" * 60)
    logger.info("  NLP PHASE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_nlp_pipeline()