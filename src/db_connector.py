# src/db_connector.py
# ============================================
# CENTRALIZED DATABASE CONNECTION MANAGER
# ============================================

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from src.utils import get_logger

load_dotenv()
logger = get_logger("DBConnector")


def get_engine():
    """
    Creates and returns a SQLAlchemy engine connected to MySQL.
    Raises a clear error if connection fails.
    """
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASS")
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")

    connection_url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"

    try:
        engine = create_engine(
            connection_url,
            pool_pre_ping=True,       # Checks connection health before using it
            pool_recycle=3600,        # Recycles connections every hour
            echo=False                # Set True to see raw SQL queries in terminal
        )
        # Test the connection immediately
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection established successfully.")
        return engine

    except OperationalError as e:
        logger.critical(f"FATAL: Could not connect to MySQL database. Error: {e}")
        raise