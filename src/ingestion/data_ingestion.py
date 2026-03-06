# src/ingestion/data_ingestion.py
# ============================================
# MASTER DATA INGESTION PIPELINE
# Fetches from: OpenWeatherMap, NewsAPI, Kaggle CSV
# Loads into:   MySQL via SQLAlchemy
# ============================================
'''
import os
import sys
import uuid
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import text

# Add project root to path so we can import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.db_connector import get_engine
from src.utils import get_logger

load_dotenv()
logger = get_logger("DataIngestion")

# ============================================
# CONFIGURATION
# ============================================
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
NEWS_API_KEY    = os.getenv("NEWS_API_KEY")
KAGGLE_CSV_PATH = "data/raw/supply_chain_raw.csv"

# ============================================
# REAL GLOBAL PORTS — Our Graph Nodes
# ============================================
MAJOR_PORTS = [
    {
        "location_id":   "PORT_SHA",
        "location_name": "Shanghai Port",
        "location_type": "Port",
        "city":          "Shanghai",
        "country":       "China",
        "latitude":      31.2222,
        "longitude":     121.4581,
        "base_capacity": 40000
    },
    {
        "location_id":   "PORT_LAX",
        "location_name": "Port of Los Angeles",
        "location_type": "Port",
        "city":          "Los Angeles",
        "country":       "USA",
        "latitude":      33.7292,
        "longitude":     -118.2620,
        "base_capacity": 20000
    },
    {
        "location_id":   "PORT_RTM",
        "location_name": "Port of Rotterdam",
        "location_type": "Port",
        "city":          "Rotterdam",
        "country":       "Netherlands",
        "latitude":      51.9496,
        "longitude":     4.1444,
        "base_capacity": 30000
    },
    {
        "location_id":   "PORT_SIN",
        "location_name": "Port of Singapore",
        "location_type": "Port",
        "city":          "Singapore",
        "country":       "Singapore",
        "latitude":      1.2640,
        "longitude":     103.8400,
        "base_capacity": 35000
    },
    {
        "location_id":   "PORT_DXB",
        "location_name": "Jebel Ali Port Dubai",
        "location_type": "Port",
        "city":          "Dubai",
        "country":       "UAE",
        "latitude":      24.9857,
        "longitude":     55.0273,
        "base_capacity": 25000
    },
    {
        "location_id":   "PORT_HKG",
        "location_name": "Port of Hong Kong",
        "location_type": "Port",
        "city":          "Hong Kong",
        "country":       "China",
        "latitude":      22.3193,
        "longitude":     114.1694,
        "base_capacity": 22000
    },
    {
        "location_id":   "PORT_ANT",
        "location_name": "Port of Antwerp",
        "location_type": "Port",
        "city":          "Antwerp",
        "country":       "Belgium",
        "latitude":      51.2213,
        "longitude":     4.4051,
        "base_capacity": 18000
    },
    {
        "location_id":   "WH_CHI",
        "location_name": "Chicago Distribution Center",
        "location_type": "Warehouse",
        "city":          "Chicago",
        "country":       "USA",
        "latitude":      41.8781,
        "longitude":     -87.6298,
        "base_capacity": 8000
    }
]

# ============================================
# ROUTES — Our Graph Edges
# ============================================
ROUTES = [
    {
        "route_id":               "RT_SHA_SIN",
        "source_location_id":     "PORT_SHA",
        "dest_location_id":       "PORT_SIN",
        "transport_mode":         "Sea",
        "standard_duration_hours": 72.0,
        "standard_cost":          2500.00,
        "distance_km":            4680.0
    },
    {
        "route_id":               "RT_SHA_LAX",
        "source_location_id":     "PORT_SHA",
        "dest_location_id":       "PORT_LAX",
        "transport_mode":         "Sea",
        "standard_duration_hours": 336.0,
        "standard_cost":          4800.00,
        "distance_km":            11200.0
    },
    {
        "route_id":               "RT_SIN_RTM",
        "source_location_id":     "PORT_SIN",
        "dest_location_id":       "PORT_RTM",
        "transport_mode":         "Sea",
        "standard_duration_hours": 504.0,
        "standard_cost":          6200.00,
        "distance_km":            16800.0
    },
    {
        "route_id":               "RT_DXB_RTM",
        "source_location_id":     "PORT_DXB",
        "dest_location_id":       "PORT_RTM",
        "transport_mode":         "Sea",
        "standard_duration_hours": 288.0,
        "standard_cost":          3900.00,
        "distance_km":            9600.0
    },
    {
        "route_id":               "RT_HKG_LAX",
        "source_location_id":     "PORT_HKG",
        "dest_location_id":       "PORT_LAX",
        "transport_mode":         "Sea",
        "standard_duration_hours": 312.0,
        "standard_cost":          4500.00,
        "distance_km":            11600.0
    },
    {
        "route_id":               "RT_RTM_ANT",
        "source_location_id":     "PORT_RTM",
        "dest_location_id":       "PORT_ANT",
        "transport_mode":         "Land",
        "standard_duration_hours": 3.0,
        "standard_cost":          450.00,
        "distance_km":            78.0
    },
    {
        "route_id":               "RT_LAX_CHI",
        "source_location_id":     "PORT_LAX",
        "dest_location_id":       "WH_CHI",
        "transport_mode":         "Land",
        "standard_duration_hours": 48.0,
        "standard_cost":          1200.00,
        "distance_km":            3200.0
    },
    {
        "route_id":               "RT_SHA_DXB",
        "source_location_id":     "PORT_SHA",
        "dest_location_id":       "PORT_DXB",
        "transport_mode":         "Sea",
        "standard_duration_hours": 168.0,
        "standard_cost":          3100.00,
        "distance_km":            7200.0
    },
    {
        "route_id":               "RT_SIN_DXB",
        "source_location_id":     "PORT_SIN",
        "dest_location_id":       "PORT_DXB",
        "transport_mode":         "Sea",
        "standard_duration_hours": 144.0,
        "standard_cost":          2800.00,
        "distance_km":            6400.0
    },
    {
        "route_id":               "RT_HKG_RTM",
        "source_location_id":     "PORT_HKG",
        "dest_location_id":       "PORT_RTM",
        "transport_mode":         "Air",
        "standard_duration_hours": 14.0,
        "standard_cost":          18000.00,
        "distance_km":            9400.0
    }
]


# ============================================
# STEP 1: SEED LOCATIONS INTO MYSQL
# ============================================
def seed_locations(engine) -> None:
    """
    Inserts our real global ports and warehouses
    into the locations table. Uses INSERT IGNORE
    so it's safe to run multiple times.
    """
    logger.info("Seeding locations into MySQL...")
    inserted = 0

    with engine.connect() as conn:
        for port in MAJOR_PORTS:
            query = text("""
                INSERT IGNORE INTO locations
                    (location_id, location_name, location_type,
                     city, country, latitude, longitude, base_capacity)
                VALUES
                    (:location_id, :location_name, :location_type,
                     :city, :country, :latitude, :longitude, :base_capacity)
            """)
            result = conn.execute(query, port)
            if result.rowcount > 0:
                inserted += 1
        conn.commit()

    logger.info(f"Locations seeded. {inserted} new records inserted, "
                f"{len(MAJOR_PORTS) - inserted} already existed.")


# ============================================
# STEP 2: SEED ROUTES INTO MYSQL
# ============================================
def seed_routes(engine) -> None:
    """
    Inserts our supply chain routes (graph edges)
    into the routes table.
    """
    logger.info("Seeding routes into MySQL...")
    inserted = 0

    with engine.connect() as conn:
        for route in ROUTES:
            query = text("""
                INSERT IGNORE INTO routes
                    (route_id, source_location_id, dest_location_id,
                     transport_mode, standard_duration_hours,
                     standard_cost, distance_km)
                VALUES
                    (:route_id, :source_location_id, :dest_location_id,
                     :transport_mode, :standard_duration_hours,
                     :standard_cost, :distance_km)
            """)
            result = conn.execute(query, route)
            if result.rowcount > 0:
                inserted += 1
        conn.commit()

    logger.info(f"Routes seeded. {inserted} new records inserted.")


# ============================================
# STEP 3: FETCH WEATHER DATA (OpenWeatherMap)
# ============================================
def fetch_weather_data(engine) -> None:
    """
    Hits the OpenWeatherMap Current Weather API for
    every port location and stores results in risk_events.
    Weather conditions are treated as operational risk factors.
    """
    logger.info("Fetching live weather data from OpenWeatherMap...")
    success_count = 0
    fail_count    = 0

    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

    for port in MAJOR_PORTS:
        try:
            params = {
                "lat":   port["latitude"],
                "lon":   port["longitude"],
                "appid": WEATHER_API_KEY,
                "units": "metric"
            }
            response = requests.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Extract weather attributes
            weather_main  = data["weather"][0]["main"]
            weather_desc  = data["weather"][0]["description"]
            wind_speed    = data["wind"]["speed"]         # m/s
            temperature   = data["main"]["temp"]          # Celsius
            humidity      = data["main"]["humidity"]      # %
            visibility    = data.get("visibility", 10000) # meters

            # Build a structured headline for NLP processing in Phase 3
            headline = (
                f"Weather at {port['location_name']}: {weather_desc}. "
                f"Wind: {wind_speed} m/s, Temp: {temperature}C, "
                f"Humidity: {humidity}%, Visibility: {visibility}m."
            )

            # Insert into risk_events
            _insert_risk_event(
                engine       = engine,
                location_id  = port["location_id"],
                headline     = headline,
                event_type   = f"Weather_{weather_main}",
                source_api   = "OpenWeatherMap"
            )
            success_count += 1
            logger.info(f"Weather fetched for {port['location_name']}: "
                        f"{weather_desc}, wind={wind_speed}m/s")

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {port['location_name']}: {e}")
            fail_count += 1
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection failed for {port['location_name']}. "
                         f"Check internet connection.")
            fail_count += 1
        except KeyError as e:
            logger.error(f"Unexpected API response structure for "
                         f"{port['location_name']}: {e}")
            fail_count += 1

    logger.info(f"Weather ingestion complete. "
                f"Success: {success_count}, Failed: {fail_count}")


# ============================================
# STEP 4: FETCH NEWS DATA (NewsAPI)
# ============================================
def fetch_news_data(engine) -> None:
    """
    Hits the NewsAPI Everything endpoint searching for
    supply chain disruption news for each major port.
    Top 3 articles per port are stored as risk events.
    """
    logger.info("Fetching live supply chain news from NewsAPI...")
    success_count = 0
    fail_count    = 0

    BASE_URL = "https://newsapi.org/v2/everything"

    # Search terms specifically targeting supply chain disruptions
    SEARCH_TERMS = [
        "shipping delay",
        "port strike",
        "supply chain disruption",
        "cargo bottleneck",
        "logistics disruption"
    ]

    for port in MAJOR_PORTS:
        # Combine port name with supply chain keywords
        query = f'"{port["city"]}" AND (shipping OR port OR cargo OR logistics OR delay OR strike)'

        try:
            params = {
                "q":        query,
                "language": "en",
                "sortBy":   "publishedAt",
                "pageSize": 3,          # Top 3 most recent articles
                "apiKey":   NEWS_API_KEY
            }
            response = requests.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json().get("articles", [])

            if not articles:
                logger.warning(f"No news articles found for {port['location_name']}")
                continue

            for article in articles:
                title       = article.get("title", "No title")
                description = article.get("description", "")
                published   = article.get("publishedAt", datetime.now().isoformat())

                # Build a rich headline combining title + description for better NLP
                headline = f"{title}. {description}" if description else title

                # Parse the published date from the API
                try:
                    event_date = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    event_date = datetime.now()

                _insert_risk_event(
                    engine      = engine,
                    location_id = port["location_id"],
                    headline    = headline,
                    event_type  = "News",
                    source_api  = "NewsAPI",
                    event_date  = event_date
                )
                success_count += 1

            logger.info(f"News fetched for {port['location_name']}: "
                        f"{len(articles)} articles stored.")

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching news for "
                         f"{port['location_name']}: {e}")
            fail_count += 1
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection failed for NewsAPI. "
                         f"Check internet connection.")
            fail_count += 1

    logger.info(f"News ingestion complete. "
                f"Success: {success_count} articles, Failed: {fail_count}")


# ============================================
# HELPER: INSERT RISK EVENT INTO MYSQL
# ============================================
def _insert_risk_event(
    engine,
    location_id: str,
    headline:    str,
    event_type:  str,
    source_api:  str,
    event_date:  datetime = None
) -> None:
    """
    Private helper function to insert a single
    risk event record into the risk_events table.
    """
    if event_date is None:
        event_date = datetime.now()

    with engine.connect() as conn:
        query = text("""
            INSERT INTO risk_events
                (location_id, event_date, headline, event_type, source_api)
            VALUES
                (:location_id, :event_date, :headline, :event_type, :source_api)
        """)
        conn.execute(query, {
            "location_id": location_id,
            "event_date":  event_date,
            "headline":    headline,
            "event_type":  event_type,
            "source_api":  source_api
        })
        conn.commit()


# ============================================
# STEP 5: LOAD KAGGLE HISTORICAL SHIPMENT DATA
# ============================================
def load_kaggle_shipments(engine) -> None:
    """
    Reads the DataCo Supply Chain CSV from Kaggle,
    maps it to our schema, and loads into MySQL.
    This gives us real historical delay patterns to train on.
    """
    logger.info(f"Loading Kaggle dataset from {KAGGLE_CSV_PATH}...")

    if not os.path.exists(KAGGLE_CSV_PATH):
        logger.error(
            f"Kaggle CSV not found at {KAGGLE_CSV_PATH}. "
            f"Please download it manually from Kaggle and place it there."
        )
        return

    try:
        # Load the raw CSV
        df = pd.read_csv(KAGGLE_CSV_PATH, encoding='unicode_escape')
        logger.info(f"Raw Kaggle data loaded: {df.shape[0]} rows, "
                    f"{df.shape[1]} columns")

        # ---- Map Kaggle columns to our schema ----
        # The DataCo dataset has columns like:
        # 'Days for shipping (real)', 'Days for shipment (scheduled)',
        # 'Late_delivery_risk', 'Shipping Mode', 'Order Id', etc.

        # Select and rename relevant columns
        df_mapped = pd.DataFrame()

        # Generate unique shipment IDs
        df_mapped["shipment_id"] = [
            f"SHP_{str(uuid.uuid4())[:8].upper()}"
            for _ in range(len(df))
        ]

        # Map shipping modes to our route IDs (simplified mapping)
        SHIPPING_MODE_TO_ROUTE = {
            "Sea":            "RT_DXB_RTM",  # Changed to Dubai route to capture risk!
            "Air":            "RT_HKG_RTM",
            "First Class":    "RT_SHA_DXB",  # Changed to Dubai route
            "Second Class":   "RT_SIN_DXB",  # Changed to Dubai route
            "Standard Class": "RT_SHA_LAX",
            "Same Day":       "RT_HKG_RTM"
        }

        if "Shipping Mode" in df.columns:
            df_mapped["route_id"] = (
                df["Shipping Mode"]
                .map(SHIPPING_MODE_TO_ROUTE)
                .fillna("RT_DXB_RTM")
            )
        else:
            df_mapped["route_id"] = "RT_DXB_RTM"

        # Cargo type from category name if available
        CATEGORY_MAP = {
            "Electronics":   "Electronics",
            "Clothing":      "Consumer_Goods",
            "Furniture":     "Raw_Materials",
            "Books":         "Consumer_Goods",
            "Sports":        "Consumer_Goods"
        }
        if "Category Name" in df.columns:
            df_mapped["cargo_type"] = (
                df["Category Name"]
                .apply(lambda x: next(
                    (v for k, v in CATEGORY_MAP.items() if k in str(x)), 
                    "Consumer_Goods"
                ))
            )
        else:
            df_mapped["cargo_type"] = "Consumer_Goods"

        # Use order weight or default
        if "Order Item Total" in df.columns:
            df_mapped["cargo_weight_tons"] = (
                df["Order Item Total"] / 1000
            ).clip(0.1, 500.0).round(2)
        else:
            df_mapped["cargo_weight_tons"] = 10.0

        # Build timestamps from order date if available
        if "order date (DateOrders)" in df.columns:
            df_mapped["dispatch_timestamp"] = pd.to_datetime(
                df["order date (DateOrders)"], 
                errors='coerce'
            ).fillna(datetime(2022, 1, 1))
        else:
            df_mapped["dispatch_timestamp"] = datetime(2022, 1, 1)

        # Scheduled shipping days
        if "Days for shipment (scheduled)" in df.columns:
            scheduled_hours = df["Days for shipment (scheduled)"] * 24
        else:
            scheduled_hours = 72

        df_mapped["expected_arrival"] = (
            pd.to_datetime(df_mapped["dispatch_timestamp"]) +
            pd.to_timedelta(scheduled_hours, unit='h')
        )

        # Real shipping days to calculate actual delay
        if "Days for shipping (real)" in df.columns:
            real_hours = df["Days for shipping (real)"] * 24
            df_mapped["actual_arrival"] = (
                pd.to_datetime(df_mapped["dispatch_timestamp"]) +
                pd.to_timedelta(real_hours, unit='h')
            )
            df_mapped["actual_delay_hours"] = (
                (real_hours - scheduled_hours)
                .clip(lower=0)
                .round(2)
            )
        else:
            df_mapped["actual_arrival"]    = df_mapped["expected_arrival"]
            df_mapped["actual_delay_hours"] = 0.0

        # Delay flag: TRUE if delay > 2 hours
        df_mapped["delay_flag"] = df_mapped["actual_delay_hours"] > 2.0

        # Late delivery risk column maps directly
        if "Late_delivery_risk" in df.columns:
            df_mapped["status"] = df["Late_delivery_risk"].apply(
                lambda x: "Delayed" if x == 1 else "Delivered"
            )
        else:
            df_mapped["status"] = "Delivered"

        # Carrier
        df_mapped["carrier_name"] = "DataCo Logistics"

        # ---- Take a clean subset (max 5000 rows for speed) ----
        df_final = df_mapped.head(5000).copy()
        df_final = df_final.dropna(subset=["dispatch_timestamp", "route_id"])

        logger.info(f"Mapped dataset ready: {len(df_final)} shipment records.")

        # ---- Batch insert into MySQL ----
        inserted = 0
        batch_size = 100

        with engine.connect() as conn:
            for i in range(0, len(df_final), batch_size):
                batch = df_final.iloc[i : i + batch_size]
                for _, row in batch.iterrows():
                    try:
                        query = text("""
                            INSERT IGNORE INTO shipments
                                (shipment_id, route_id, cargo_type,
                                 cargo_weight_tons, dispatch_timestamp,
                                 expected_arrival, actual_arrival,
                                 actual_delay_hours, delay_flag,
                                 status, carrier_name)
                            VALUES
                                (:shipment_id, :route_id, :cargo_type,
                                 :cargo_weight_tons, :dispatch_timestamp,
                                 :expected_arrival, :actual_arrival,
                                 :actual_delay_hours, :delay_flag,
                                 :status, :carrier_name)
                        """)
                        conn.execute(query, {
                            "shipment_id":        row["shipment_id"],
                            "route_id":           row["route_id"],
                            "cargo_type":         row["cargo_type"],
                            "cargo_weight_tons":  float(row["cargo_weight_tons"]),
                            "dispatch_timestamp": row["dispatch_timestamp"],
                            "expected_arrival":   row["expected_arrival"],
                            "actual_arrival":     row["actual_arrival"],
                            "actual_delay_hours": float(row["actual_delay_hours"]),
                            "delay_flag":         bool(row["delay_flag"]),
                            "status":             row["status"],
                            "carrier_name":       row["carrier_name"]
                        })
                        inserted += 1
                    except Exception as row_error:
                        logger.warning(f"Skipping row due to error: {row_error}")

                conn.commit()
                logger.info(f"Batch {i // batch_size + 1} committed. "
                            f"Total inserted so far: {inserted}")

        logger.info(f"Kaggle shipment data loaded successfully. "
                    f"Total records inserted: {inserted}")

    except Exception as e:
        logger.error(f"Failed to load Kaggle data: {e}")
        raise


# ============================================
# MAIN PIPELINE RUNNER
# ============================================
def run_pipeline() -> None:
    """
    Orchestrates the full ingestion pipeline.
    Run this script directly to populate your database.
    """
    logger.info("=" * 60)
    logger.info("  SUPPLY CHAIN OPTIMIZER — DATA INGESTION PIPELINE")
    logger.info("=" * 60)

    # Get database connection
    engine = get_engine()

    # Execute all ingestion steps in order
    seed_locations(engine)
    seed_routes(engine)
    fetch_weather_data(engine)
    fetch_news_data(engine)
    load_kaggle_shipments(engine)

    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE — All data loaded into MySQL")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
    '''

# src/ingestion/data_ingestion.py
# ============================================
# MASTER DATA INGESTION PIPELINE
# Fetches from: OpenWeatherMap, NewsAPI, Kaggle CSV
# Loads into:   MySQL via SQLAlchemy
# ============================================

import os
import sys
import uuid
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import text

# Add project root to path so we can import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.db_connector import get_engine
from src.utils import get_logger

load_dotenv()
logger = get_logger("DataIngestion")

# ============================================
# CONFIGURATION
# ============================================
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
NEWS_API_KEY    = os.getenv("NEWS_API_KEY")
KAGGLE_CSV_PATH = "data/raw/supply_chain_raw.csv"

# ============================================
# REAL GLOBAL PORTS — Our Graph Nodes
# ============================================
MAJOR_PORTS = [
    {
        "location_id":   "PORT_SHA",
        "location_name": "Shanghai Port",
        "location_type": "Port",
        "city":          "Shanghai",
        "country":       "China",
        "latitude":      31.2222,
        "longitude":     121.4581,
        "base_capacity": 40000
    },
    {
        "location_id":   "PORT_LAX",
        "location_name": "Port of Los Angeles",
        "location_type": "Port",
        "city":          "Los Angeles",
        "country":       "USA",
        "latitude":      33.7292,
        "longitude":     -118.2620,
        "base_capacity": 20000
    },
    {
        "location_id":   "PORT_RTM",
        "location_name": "Port of Rotterdam",
        "location_type": "Port",
        "city":          "Rotterdam",
        "country":       "Netherlands",
        "latitude":      51.9496,
        "longitude":     4.1444,
        "base_capacity": 30000
    },
    {
        "location_id":   "PORT_SIN",
        "location_name": "Port of Singapore",
        "location_type": "Port",
        "city":          "Singapore",
        "country":       "Singapore",
        "latitude":      1.2640,
        "longitude":     103.8400,
        "base_capacity": 35000
    },
    {
        "location_id":   "PORT_DXB",
        "location_name": "Jebel Ali Port Dubai",
        "location_type": "Port",
        "city":          "Dubai",
        "country":       "UAE",
        "latitude":      24.9857,
        "longitude":     55.0273,
        "base_capacity": 25000
    },
    {
        "location_id":   "PORT_HKG",
        "location_name": "Port of Hong Kong",
        "location_type": "Port",
        "city":          "Hong Kong",
        "country":       "China",
        "latitude":      22.3193,
        "longitude":     114.1694,
        "base_capacity": 22000
    },
    {
        "location_id":   "PORT_ANT",
        "location_name": "Port of Antwerp",
        "location_type": "Port",
        "city":          "Antwerp",
        "country":       "Belgium",
        "latitude":      51.2213,
        "longitude":     4.4051,
        "base_capacity": 18000
    },
    {
        "location_id":   "WH_CHI",
        "location_name": "Chicago Distribution Center",
        "location_type": "Warehouse",
        "city":          "Chicago",
        "country":       "USA",
        "latitude":      41.8781,
        "longitude":     -87.6298,
        "base_capacity": 8000
    }
]

# ============================================
# ROUTES — Our Graph Edges
# ============================================
ROUTES = [
    {
        "route_id":               "RT_SHA_SIN",
        "source_location_id":     "PORT_SHA",
        "dest_location_id":       "PORT_SIN",
        "transport_mode":         "Sea",
        "standard_duration_hours": 72.0,
        "standard_cost":          2500.00,
        "distance_km":            4680.0
    },
    {
        "route_id":               "RT_SHA_LAX",
        "source_location_id":     "PORT_SHA",
        "dest_location_id":       "PORT_LAX",
        "transport_mode":         "Sea",
        "standard_duration_hours": 336.0,
        "standard_cost":          4800.00,
        "distance_km":            11200.0
    },
    {
        "route_id":               "RT_SIN_RTM",
        "source_location_id":     "PORT_SIN",
        "dest_location_id":       "PORT_RTM",
        "transport_mode":         "Sea",
        "standard_duration_hours": 504.0,
        "standard_cost":          6200.00,
        "distance_km":            16800.0
    },
    {
        "route_id":               "RT_DXB_RTM",
        "source_location_id":     "PORT_DXB",
        "dest_location_id":       "PORT_RTM",
        "transport_mode":         "Sea",
        "standard_duration_hours": 288.0,
        "standard_cost":          3900.00,
        "distance_km":            9600.0
    },
    {
        "route_id":               "RT_HKG_LAX",
        "source_location_id":     "PORT_HKG",
        "dest_location_id":       "PORT_LAX",
        "transport_mode":         "Sea",
        "standard_duration_hours": 312.0,
        "standard_cost":          4500.00,
        "distance_km":            11600.0
    },
    {
        "route_id":               "RT_RTM_ANT",
        "source_location_id":     "PORT_RTM",
        "dest_location_id":       "PORT_ANT",
        "transport_mode":         "Land",
        "standard_duration_hours": 3.0,
        "standard_cost":          450.00,
        "distance_km":            78.0
    },
    {
        "route_id":               "RT_LAX_CHI",
        "source_location_id":     "PORT_LAX",
        "dest_location_id":       "WH_CHI",
        "transport_mode":         "Land",
        "standard_duration_hours": 48.0,
        "standard_cost":          1200.00,
        "distance_km":            3200.0
    },
    {
        "route_id":               "RT_SHA_DXB",
        "source_location_id":     "PORT_SHA",
        "dest_location_id":       "PORT_DXB",
        "transport_mode":         "Sea",
        "standard_duration_hours": 168.0,
        "standard_cost":          3100.00,
        "distance_km":            7200.0
    },
    {
        "route_id":               "RT_SIN_DXB",
        "source_location_id":     "PORT_SIN",
        "dest_location_id":       "PORT_DXB",
        "transport_mode":         "Sea",
        "standard_duration_hours": 144.0,
        "standard_cost":          2800.00,
        "distance_km":            6400.0
    },
    {
        "route_id":               "RT_HKG_RTM",
        "source_location_id":     "PORT_HKG",
        "dest_location_id":       "PORT_RTM",
        "transport_mode":         "Air",
        "standard_duration_hours": 14.0,
        "standard_cost":          18000.00,
        "distance_km":            9400.0
    }
]


# ============================================
# STEP 1: SEED LOCATIONS INTO MYSQL
# ============================================
def seed_locations(engine) -> None:
    """
    Inserts our real global ports and warehouses
    into the locations table. Uses INSERT IGNORE
    so it's safe to run multiple times.
    """
    logger.info("Seeding locations into MySQL...")
    inserted = 0

    with engine.connect() as conn:
        for port in MAJOR_PORTS:
            query = text("""
                INSERT IGNORE INTO locations
                    (location_id, location_name, location_type,
                     city, country, latitude, longitude, base_capacity)
                VALUES
                    (:location_id, :location_name, :location_type,
                     :city, :country, :latitude, :longitude, :base_capacity)
            """)
            result = conn.execute(query, port)
            if result.rowcount > 0:
                inserted += 1
        conn.commit()

    logger.info(f"Locations seeded. {inserted} new records inserted, "
                f"{len(MAJOR_PORTS) - inserted} already existed.")


# ============================================
# STEP 2: SEED ROUTES INTO MYSQL
# ============================================
def seed_routes(engine) -> None:
    """
    Inserts our supply chain routes (graph edges)
    into the routes table.
    """
    logger.info("Seeding routes into MySQL...")
    inserted = 0

    with engine.connect() as conn:
        for route in ROUTES:
            query = text("""
                INSERT IGNORE INTO routes
                    (route_id, source_location_id, dest_location_id,
                     transport_mode, standard_duration_hours,
                     standard_cost, distance_km)
                VALUES
                    (:route_id, :source_location_id, :dest_location_id,
                     :transport_mode, :standard_duration_hours,
                     :standard_cost, :distance_km)
            """)
            result = conn.execute(query, route)
            if result.rowcount > 0:
                inserted += 1
        conn.commit()

    logger.info(f"Routes seeded. {inserted} new records inserted.")


# ============================================
# STEP 3: FETCH WEATHER DATA (OpenWeatherMap)
# ============================================
def fetch_weather_data(engine) -> None:
    """
    Hits the OpenWeatherMap Current Weather API for
    every port location and stores results in risk_events.
    Weather conditions are treated as operational risk factors.
    """
    logger.info("Fetching live weather data from OpenWeatherMap...")
    success_count = 0
    fail_count    = 0

    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

    for port in MAJOR_PORTS:
        try:
            params = {
                "lat":   port["latitude"],
                "lon":   port["longitude"],
                "appid": WEATHER_API_KEY,
                "units": "metric"
            }
            response = requests.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Extract weather attributes
            weather_main  = data["weather"][0]["main"]
            weather_desc  = data["weather"][0]["description"]
            wind_speed    = data["wind"]["speed"]         # m/s
            temperature   = data["main"]["temp"]          # Celsius
            humidity      = data["main"]["humidity"]      # %
            visibility    = data.get("visibility", 10000) # meters

            # Build a structured headline for NLP processing in Phase 3
            headline = (
                f"Weather at {port['location_name']}: {weather_desc}. "
                f"Wind: {wind_speed} m/s, Temp: {temperature}C, "
                f"Humidity: {humidity}%, Visibility: {visibility}m."
            )

            # Insert into risk_events
            _insert_risk_event(
                engine       = engine,
                location_id  = port["location_id"],
                headline     = headline,
                event_type   = f"Weather_{weather_main}",
                source_api   = "OpenWeatherMap"
            )
            success_count += 1
            logger.info(f"Weather fetched for {port['location_name']}: "
                        f"{weather_desc}, wind={wind_speed}m/s")

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {port['location_name']}: {e}")
            fail_count += 1
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection failed for {port['location_name']}. "
                         f"Check internet connection.")
            fail_count += 1
        except KeyError as e:
            logger.error(f"Unexpected API response structure for "
                         f"{port['location_name']}: {e}")
            fail_count += 1

    logger.info(f"Weather ingestion complete. "
                f"Success: {success_count}, Failed: {fail_count}")


# ============================================
# STEP 4: FETCH NEWS DATA (NewsAPI)
# ============================================
def fetch_news_data(engine) -> None:
    """
    Hits the NewsAPI Everything endpoint searching for
    supply chain disruption news for each major port.
    Top 3 articles per port are stored as risk events.
    """
    logger.info("Fetching live supply chain news from NewsAPI...")
    success_count = 0
    fail_count    = 0

    BASE_URL = "https://newsapi.org/v2/everything"

    for port in MAJOR_PORTS:
        # Combine port name with supply chain keywords
        query = f'"{port["city"]}" AND (shipping OR port OR cargo OR logistics OR delay OR strike)'

        try:
            params = {
                "q":        query,
                "language": "en",
                "sortBy":   "publishedAt",
                "pageSize": 3,          # Top 3 most recent articles
                "apiKey":   NEWS_API_KEY
            }
            response = requests.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json().get("articles", [])

            if not articles:
                logger.warning(f"No news articles found for {port['location_name']}")
                continue

            for article in articles:
                title       = article.get("title", "No title")
                description = article.get("description", "")
                published   = article.get("publishedAt", datetime.now().isoformat())

                # Build a rich headline combining title + description for better NLP
                headline = f"{title}. {description}" if description else title

                # Parse the published date from the API
                try:
                    event_date = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    event_date = datetime.now()

                _insert_risk_event(
                    engine      = engine,
                    location_id = port["location_id"],
                    headline    = headline,
                    event_type  = "News",
                    source_api  = "NewsAPI",
                    event_date  = event_date
                )
                success_count += 1

            logger.info(f"News fetched for {port['location_name']}: "
                        f"{len(articles)} articles stored.")

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching news for "
                         f"{port['location_name']}: {e}")
            fail_count += 1
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection failed for NewsAPI. "
                         f"Check internet connection.")
            fail_count += 1

    logger.info(f"News ingestion complete. "
                f"Success: {success_count} articles, Failed: {fail_count}")


# ============================================
# HELPER: INSERT RISK EVENT INTO MYSQL
# ============================================
def _insert_risk_event(
    engine,
    location_id: str,
    headline:    str,
    event_type:  str,
    source_api:  str,
    event_date:  datetime = None
) -> None:
    """
    Private helper function to insert a single
    risk event record into the risk_events table.
    """
    if event_date is None:
        event_date = datetime.now()

    with engine.connect() as conn:
        query = text("""
            INSERT INTO risk_events
                (location_id, event_date, headline, event_type, source_api)
            VALUES
                (:location_id, :event_date, :headline, :event_type, :source_api)
        """)
        conn.execute(query, {
            "location_id": location_id,
            "event_date":  event_date,
            "headline":    headline,
            "event_type":  event_type,
            "source_api":  source_api
        })
        conn.commit()


# ============================================
# STEP 5: LOAD KAGGLE HISTORICAL SHIPMENT DATA
# ============================================
def load_kaggle_shipments(engine) -> None:
    """
    Reads the DataCo Supply Chain CSV from Kaggle,
    maps it to our schema, and loads into MySQL.
    This gives us real historical delay patterns to train on.
    """
    logger.info(f"Loading Kaggle dataset from {KAGGLE_CSV_PATH}...")

    if not os.path.exists(KAGGLE_CSV_PATH):
        logger.error(
            f"Kaggle CSV not found at {KAGGLE_CSV_PATH}. "
            f"Please download it manually from Kaggle and place it there."
        )
        return

    try:
        # Load the raw CSV
        df = pd.read_csv(KAGGLE_CSV_PATH, encoding='unicode_escape')
        logger.info(f"Raw Kaggle data loaded: {df.shape[0]} rows, "
                    f"{df.shape[1]} columns")

        # ---- Map Kaggle columns to our schema ----
        df_mapped = pd.DataFrame()

        # Generate unique shipment IDs
        df_mapped["shipment_id"] = [
            f"SHP_{str(uuid.uuid4())[:8].upper()}"
            for _ in range(len(df))
        ]

        # Map shipping modes to our route IDs
        SHIPPING_MODE_TO_ROUTE = {
            "Sea":            "RT_DXB_RTM",
            "Air":            "RT_HKG_RTM",
            "First Class":    "RT_SHA_DXB",
            "Second Class":   "RT_SIN_DXB",
            "Standard Class": "RT_SHA_LAX",
            "Same Day":       "RT_HKG_RTM"
        }

        if "Shipping Mode" in df.columns:
            df_mapped["route_id"] = (
                df["Shipping Mode"]
                .map(SHIPPING_MODE_TO_ROUTE)
                .fillna("RT_DXB_RTM")
            )
        else:
            df_mapped["route_id"] = "RT_DXB_RTM"

        # Cargo type from category name if available
        CATEGORY_MAP = {
            "Electronics":   "Electronics",
            "Clothing":      "Consumer_Goods",
            "Furniture":     "Raw_Materials",
            "Books":         "Consumer_Goods",
            "Sports":        "Consumer_Goods"
        }
        if "Category Name" in df.columns:
            df_mapped["cargo_type"] = (
                df["Category Name"]
                .apply(lambda x: next(
                    (v for k, v in CATEGORY_MAP.items() if k in str(x)),
                    "Consumer_Goods"
                ))
            )
        else:
            df_mapped["cargo_type"] = "Consumer_Goods"

        # Use order weight or default
        if "Order Item Total" in df.columns:
            df_mapped["cargo_weight_tons"] = (
                df["Order Item Total"] / 1000
            ).clip(0.1, 500.0).round(2)
        else:
            df_mapped["cargo_weight_tons"] = 10.0

        # Build timestamps from order date if available
        if "order date (DateOrders)" in df.columns:
            df_mapped["dispatch_timestamp"] = pd.to_datetime(
                df["order date (DateOrders)"],
                errors='coerce'
            ).fillna(datetime(2022, 1, 1))
        else:
            df_mapped["dispatch_timestamp"] = datetime(2022, 1, 1)

        # Scheduled shipping days
        if "Days for shipment (scheduled)" in df.columns:
            scheduled_hours = df["Days for shipment (scheduled)"] * 24
        else:
            scheduled_hours = 72

        df_mapped["expected_arrival"] = (
            pd.to_datetime(df_mapped["dispatch_timestamp"]) +
            pd.to_timedelta(scheduled_hours, unit='h')
        )

        # ---- UPDATED: Real shipping days to calculate actual delay ----
        if "Days for shipping (real)" in df.columns:
            real_hours = df["Days for shipping (real)"] * 24
            scheduled_hours_series = (
                df["Days for shipment (scheduled)"] * 24
                if "Days for shipment (scheduled)" in df.columns
                else pd.Series([72] * len(df))
            )
            raw_delay = (real_hours - scheduled_hours_series).clip(lower=0)

            # KEY FIX: Cap delay at realistic max
            # Real-world shipment delays rarely exceed 72 hours (3 days)
            # Values above this in the Kaggle dataset are data artifacts
            # from the day-level granularity of the source data
            df_mapped["actual_delay_hours"] = raw_delay.clip(
                lower=0,
                upper=72            # Max 72 hours = 3 days realistic cap
            ).round(2)

            df_mapped["actual_arrival"] = (
                pd.to_datetime(df_mapped["dispatch_timestamp"]) +
                pd.to_timedelta(real_hours, unit="h")
            )
        else:
            df_mapped["actual_arrival"]     = df_mapped["expected_arrival"]
            df_mapped["actual_delay_hours"] = 0.0

        # Delay flag: TRUE if delay > 6 hours
        # Using 6 hours as threshold for more meaningful classification
        df_mapped["delay_flag"] = df_mapped["actual_delay_hours"] > 6.0

        # Late delivery risk column maps directly
        if "Late_delivery_risk" in df.columns:
            df_mapped["status"] = df["Late_delivery_risk"].apply(
                lambda x: "Delayed" if x == 1 else "Delivered"
            )
        else:
            df_mapped["status"] = "Delivered"

        # Carrier
        df_mapped["carrier_name"] = "DataCo Logistics"

        # ---- Take a clean subset (max 5000 rows for speed) ----
        df_final = df_mapped.head(5000).copy()
        df_final = df_final.dropna(subset=["dispatch_timestamp", "route_id"])

        logger.info(f"Mapped dataset ready: {len(df_final)} shipment records.")

        # ---- Batch insert into MySQL ----
        inserted  = 0
        batch_size = 100

        with engine.connect() as conn:
            for i in range(0, len(df_final), batch_size):
                batch = df_final.iloc[i : i + batch_size]
                for _, row in batch.iterrows():
                    try:
                        query = text("""
                            INSERT IGNORE INTO shipments
                                (shipment_id, route_id, cargo_type,
                                 cargo_weight_tons, dispatch_timestamp,
                                 expected_arrival, actual_arrival,
                                 actual_delay_hours, delay_flag,
                                 status, carrier_name)
                            VALUES
                                (:shipment_id, :route_id, :cargo_type,
                                 :cargo_weight_tons, :dispatch_timestamp,
                                 :expected_arrival, :actual_arrival,
                                 :actual_delay_hours, :delay_flag,
                                 :status, :carrier_name)
                        """)
                        conn.execute(query, {
                            "shipment_id":        row["shipment_id"],
                            "route_id":           row["route_id"],
                            "cargo_type":         row["cargo_type"],
                            "cargo_weight_tons":  float(row["cargo_weight_tons"]),
                            "dispatch_timestamp": row["dispatch_timestamp"],
                            "expected_arrival":   row["expected_arrival"],
                            "actual_arrival":     row["actual_arrival"],
                            "actual_delay_hours": float(row["actual_delay_hours"]),
                            "delay_flag":         bool(row["delay_flag"]),
                            "status":             row["status"],
                            "carrier_name":       row["carrier_name"]
                        })
                        inserted += 1
                    except Exception as row_error:
                        logger.warning(f"Skipping row due to error: {row_error}")

                conn.commit()
                logger.info(f"Batch {i // batch_size + 1} committed. "
                            f"Total inserted so far: {inserted}")

        logger.info(f"Kaggle shipment data loaded successfully. "
                    f"Total records inserted: {inserted}")

    except Exception as e:
        logger.error(f"Failed to load Kaggle data: {e}")
        raise


# ============================================
# MAIN PIPELINE RUNNER
# ============================================
def run_pipeline() -> None:
    """
    Orchestrates the full ingestion pipeline.
    Run this script directly to populate your database.
    """
    logger.info("=" * 60)
    logger.info("  SUPPLY CHAIN OPTIMIZER — DATA INGESTION PIPELINE")
    logger.info("=" * 60)

    # Get database connection
    engine = get_engine()

    # Execute all ingestion steps in order
    seed_locations(engine)
    seed_routes(engine)
    fetch_weather_data(engine)
    fetch_news_data(engine)
    load_kaggle_shipments(engine)

    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE — All data loaded into MySQL")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()