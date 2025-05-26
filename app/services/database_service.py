import logging
import numpy as np
from pymongo import MongoClient
import gc
import itertools
import torch

logger = logging.getLogger(__name__)


class DatabaseService:
    """Handles database operations for MongoDB"""

    def __init__(self, config):
        self.config = config
        self.mongo_client = MongoClient(self.config.MONGODB_URI)
        self.db = self.mongo_client[self.config.MONGODB_DATABASE]
        self._initialize_mongodb_data()
        # torch.set_grad_enabled(False)

    def _initialize_mongodb_data(self):
        """Initialize MongoDB data with indexes for on-demand loading"""
        try:
            # Initialize target elements
            self.target_elements = ['floor', 'ceiling', 'door', 'window']

            # Initialize available classes
            self.available_classes = [
                'wall', 'door', 'chair', 'table', 'window', 'floor',
                'ceiling', 'bed', 'sofa', 'fridge', 'microwave',
                'painting', 'plant', 'sink', 'toilet', 'book',
                'carpet', 'curtain'
            ]

            # Add safety checks before creating indexes
            for collection_name in [
                self.config.MONGODB_DHN_COLLECTION,
                self.config.MONGODB_BINARY_COLLECTION
            ]:
                try:
                    # Check if collection exists before trying to create an index
                    if collection_name in self.db.list_collection_names():
                        if collection_name == self.config.MONGODB_DHN_COLLECTION:
                            # Create compound index for segment and binary code
                            self.db[collection_name].create_index([
                                ("Segment", 1),
                                ("Binary Code", 1)
                            ], background=True)  # Use background indexing
                            logger.info(f"Created index for collection {collection_name}")
                        else:
                            # Create index for binary codes collection
                            self.db[collection_name].create_index([("label", 1)], background=True)
                            logger.info(f"Created index for collection {collection_name}")
                    else:
                        logger.warning(f"Collection {collection_name} does not exist, skipping index creation")
                except Exception as e:
                    # Log but continue if an index creation fails
                    logger.warning(f"Error creating index for {collection_name}: {str(e)}")
                    continue

            # Verify collections are accessible
            for collection_name in [
                self.config.MONGODB_DHN_COLLECTION,
                self.config.MONGODB_BINARY_COLLECTION
            ]:
                if collection_name in self.db.list_collection_names():
                    count = self.db[collection_name].count_documents({})
                    logger.info(f"Collection {collection_name} contains {count} documents")
                else:
                    logger.warning(f"Collection {collection_name} does not exist")

        except Exception as e:
            logger.error(f"Error initializing MongoDB data: {str(e)}")
            # Don't raise the exception - allow the application to continue even if index creation fails
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_segmentation_data(self, collection_name: str, segment: str = None, batch_size: int = 1000):
        """Get segmentation data in batches when needed"""
        try:
            query = {"Segment": segment} if segment else {}
            cursor = self.db[collection_name].find(
                query,
                {"Binary Code": 1, "Image Name": 1, "Segment": 1, "Country": 1},
                batch_size=batch_size
            )

            # Process in batches
            while True:
                batch = list(itertools.islice(cursor, batch_size))
                if not batch:
                    break

                # Process binary codes in batch
                for doc in batch:
                    if isinstance(doc['Binary Code'], str):
                        doc['Binary Code'] = np.fromstring(
                            doc['Binary Code'].strip('[]'),
                            sep=' ',
                            dtype=np.float32
                        )

                yield batch

        except Exception as e:
            logger.error(f"Error retrieving segmentation data: {str(e)}")
            raise
        finally:
            if 'cursor' in locals():
                cursor.close()

def cleanup(self):
    """Close MongoDB connection"""
    try:
        self.mongo_client.close()
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {str(e)}")