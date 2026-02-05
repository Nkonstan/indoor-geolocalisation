import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import pickle
import logging
import sys
import os
from time import sleep

try:
    from app.utils.logging import setup_logging
    setup_logging()
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

try:
    # Database Connection
    MONGODB_URI = os.environ['MONGODB_URI']
    MONGODB_DATABASE = os.environ['MONGODB_DATABASE']
    
    # Data File Paths
    SEGMENTATION_DB_DHN = os.environ['SEGMENTATION_DB_DHN']
    DB_BINARY_PATH = os.environ['DB_BINARY_PATH']
    DB_LABELS_PATH = os.environ['DB_LABELS_PATH']
    
    MAX_RETRIES = int(os.environ.get('MAX_RETRIES', '3'))
    RETRY_DELAY = int(os.environ.get('RETRY_DELAY', '5'))

except KeyError as e:
    # Print a blocking error message describing exactly what is missing
    print(f"\n{'='*60}")
    print(f"CRITICAL CONFIGURATION ERROR: Missing environment variable")
    print(f"variable name: {e}")
    print(f"{'='*60}\n")
    # Exit with status code 1 (indicates failure to Docker/Orchestrator)
    sys.exit(1)


def check_mongodb_connection(client):
    """Test if MongoDB server is responsive"""
    try:
        client.admin.command('ismaster')
        return True
    except ConnectionFailure:
        return False


def verify_input_files():
    """Verify all required input files exist"""
    required_files = [
        SEGMENTATION_DB_DHN,
        DB_BINARY_PATH,
        DB_LABELS_PATH
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")


def migrate_segmentation_data(db, file_path, collection_name):
    """Migrate segmentation data to specified collection"""
    logger.info(f"Migrating {collection_name} data from {file_path}...")
    df = pd.read_csv(file_path)

    # Insert in batches to handle large datasets
    batch_size = 1000
    total_records = len(df)

    for i in range(0, total_records, batch_size):
        batch = df.iloc[i:i + batch_size]
        db[collection_name].insert_many(batch.to_dict('records'))
        logger.info(f"Inserted {min(i + batch_size, total_records)}/{total_records} records in {collection_name}")


def migrate_binary_data(db):
    """Migrate binary codes and labels"""
    logger.info("Migrating binary codes and labels...")

    db_binary = np.load(DB_BINARY_PATH)
    with open(DB_LABELS_PATH, 'rb') as f:
        db_paths = pickle.load(f)
    db_labels = [path.split('/')[2] for path in db_paths]

    # Create and insert documents in batches
    batch_size = 1000
    total_records = len(db_binary)

    for i in range(0, total_records, batch_size):
        batch_documents = []
        for j in range(i, min(i + batch_size, total_records)):
            batch_documents.append({
                'binary_code': db_binary[j].tolist(),
                'path': db_paths[j],
                'label': db_labels[j]
            })
        db.binary_codes.insert_many(batch_documents)
        logger.info(f"Inserted {min(i + batch_size, total_records)}/{total_records} binary code records")


def migrate_data():
    """Main migration function with retry logic"""
    verify_input_files()

    for attempt in range(MAX_RETRIES):
        try:
            client = MongoClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=5000,  # Shorter timeout for faster feedback
                connectTimeoutMS=5000
            )

            if not check_mongodb_connection(client):
                raise ConnectionFailure("MongoDB server is not available")

            db = client[MONGODB_DATABASE]

            # Clear existing collections
            db.segmentation_dhn.drop()
            db.segmentation_hashnet.drop()
            db.binary_codes.drop()

            # Perform migrations
            migrate_segmentation_data(db, SEGMENTATION_DB_DHN, 'segmentation_dhn')
            migrate_binary_data(db)

            logger.info("Migration completed successfully!")
            return True

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                sleep(RETRY_DELAY)
            else:
                logger.error("Max retries reached. Please check if MongoDB is running.")
                raise
        except Exception as e:
            logger.error(f"Unexpected error during migration: {str(e)}")
            raise
        finally:
            client.close()


if __name__ == "__main__":
    try:
        migrate_data()
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        sys.exit(1)