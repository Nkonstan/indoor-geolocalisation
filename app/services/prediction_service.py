import logging
import numpy as np
import json
import gc
import torch
import time
import faiss

logger = logging.getLogger(__name__)

class PredictionService:
    """Handles predictions and analysis based on feature vectors"""
    def __init__(self, model_service):
        self.model_service = model_service
        self.config = model_service.config
        self.device = model_service.device
        # Initialize geo index manager for Faiss
        self.geo_index_manager = {
            'index': None,
            'metadata': [],
            'is_built': False
        }

    def process_database_predictions(self, query_vec):
        """Process database predictions using original distance metric"""
        logger.info("Processing database predictions with optimized approach")
        try:
            # Handle query_vec shape once at the start
            query_vec = query_vec.squeeze()
            if len(query_vec.shape) == 1:
                query_vec = query_vec.reshape(1, -1)

            # Constants
            CHUNK_SIZE = 10000
            TOP_K = 1000

            # Pre-allocate arrays with numpy for better performance
            top_k_distances = np.full(TOP_K, np.inf, dtype=np.float32)
            top_k_labels = np.array([None] * TOP_K, dtype=object)

            # Get cursor with optimized projection and batch size
            cursor = self.model_service.db_service.db[self.config.MONGODB_BINARY_COLLECTION].find(
                projection={'binary_code': 1, 'label': 1, '_id': 0},
                batch_size=CHUNK_SIZE
            )

            # Pre-allocate chunk buffer
            current_chunk = np.zeros((CHUNK_SIZE, query_vec.shape[1]), dtype=np.float32)
            chunk_labels = []
            current_idx = 0

            # Process in batches with minimal memory allocations
            for doc in cursor:
                try:
                    # Convert binary code with minimal copying
                    if isinstance(doc['binary_code'], str):
                        binary_code = np.fromstring(doc['binary_code'].strip('[]'), sep=' ', dtype=np.float32)
                    else:
                        binary_code = np.asarray(doc['binary_code'], dtype=np.float32)

                    # Reshape if needed
                    if binary_code.shape != (query_vec.shape[1],):
                        binary_code = binary_code.reshape(query_vec.shape[1])

                    # Add to current chunk
                    current_chunk[current_idx] = binary_code
                    chunk_labels.append(doc['label'])
                    current_idx += 1

                    # Process chunk when it's full
                    if current_idx == CHUNK_SIZE:
                        self.process_chunk(
                            current_chunk,
                            chunk_labels,
                            query_vec,
                            top_k_distances,
                            top_k_labels
                        )
                        # Reset chunk
                        current_idx = 0
                        chunk_labels = []
                        current_chunk.fill(0)  # Reuse array instead of creating new
                        # Force cleanup after each large chunk
                        gc.collect()

                except Exception as e:
                    logger.error(f"Error processing document: {str(e)}")
                    continue

            # Process remaining documents in last chunk
            if current_idx > 0:
                self.process_chunk(
                    current_chunk[:current_idx],
                    chunk_labels,
                    query_vec,
                    top_k_distances,
                    top_k_labels
                )

            # Count labels using numpy operations for efficiency
            label_counts = {}
            for label in top_k_labels:
                if label is not None:
                    label_counts[label] = label_counts.get(label, 0) + 1

            return self.format_predictions(label_counts)

        except Exception as e:
            logger.error(f"Error processing database predictions: {str(e)}")
            raise

        finally:
            # Thorough cleanup
            try:
                if 'cursor' in locals() and cursor is not None:
                    cursor.close()

                cleanup_vars = ['current_chunk', 'binary_code', 'chunk_labels']
                for var in cleanup_vars:
                    if var in locals():
                        del locals()[var]

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {str(cleanup_error)}")

    def process_database_predictions_custom_faiss(self, query_vec):
        """Process database predictions using Faiss with custom distance calculation"""
        logger.info("Processing database predictions with custom Faiss approach")
        try:
            # Handle query_vec shape
            query_vec = query_vec.squeeze()
            if len(query_vec.shape) == 1:
                query_vec = query_vec.reshape(1, -1)

            # Constants
            TOP_K = 1000

            # Get or initialize the index manager
            if not hasattr(self, 'custom_index_manager'):
                self.custom_index_manager = {
                    'index': None,
                    'vectors': None,
                    'metadata': [],
                    'is_built': False
                }

            # Build index if needed
            if not self.custom_index_manager['is_built']:
                logger.info("Building custom Faiss index...")
                start_time = time.time()

                # We'll use FlatL2 index for storing original vectors
                # (but will apply custom distance calculation during search)
                dimension = query_vec.shape[1]
                index = faiss.IndexFlatL2(dimension)

                # We'll store all vectors to apply custom distance later
                all_vectors = []
                all_labels = []

                # Get all vectors from MongoDB
                cursor = self.model_service.db_service.db[self.config.MONGODB_BINARY_COLLECTION].find(
                    projection={'binary_code': 1, 'label': 1, '_id': 0}
                )

                for doc in cursor:
                    try:
                        # Process vector format
                        if isinstance(doc['binary_code'], str):
                            vector = np.fromstring(doc['binary_code'].strip('[]'), sep=' ', dtype=np.float32)
                        else:
                            vector = np.asarray(doc['binary_code'], dtype=np.float32)

                        # Reshape if needed
                        if vector.shape != (dimension,):
                            vector = vector.reshape(dimension)

                        # Keep original vector (without -1 to 0 conversion)
                        all_vectors.append(vector)
                        all_labels.append(doc['label'])
                    except Exception as e:
                        logger.error(f"Error processing document: {str(e)}")
                        continue

                # Convert to numpy array
                vectors_array = np.vstack(all_vectors).astype(np.float32)

                # Add to standard L2 index
                index.add(vectors_array)

                # Store everything
                self.custom_index_manager['index'] = index
                self.custom_index_manager['vectors'] = vectors_array
                self.custom_index_manager['metadata'] = all_labels
                self.custom_index_manager['is_built'] = True

                logger.info(f"Built custom index with {len(all_labels)} vectors in {time.time() - start_time:.2f}s")

            # Now perform custom distance calculation
            # First get a larger pool of candidates with standard L2 distance
            CANDIDATE_POOL = TOP_K * 10  # Get 10x more candidates
            start_time = time.time()

            # Get candidates using standard L2 distance
            distances, indices = self.custom_index_manager['index'].search(query_vec, CANDIDATE_POOL)

            # Now apply your custom distance to these candidates
            custom_distances = []
            for idx in indices[0]:
                # Get the original vector
                candidate_vec = self.custom_index_manager['vectors'][idx]

                # Apply your custom distance calculation
                # 1. Convert -1 to 0 (as in your original algorithm)
                candidate_vec_converted = np.where(candidate_vec == -1, 0, candidate_vec)

                # 2. Calculate distance using != operator (as in your original algorithm)
                distance = np.sum(query_vec != candidate_vec_converted)

                custom_distances.append((idx, distance))

            # Sort by custom distance
            custom_distances.sort(key=lambda x: x[1])

            # Get the top K
            top_indices = [idx for idx, _ in custom_distances[:TOP_K]]

            # Count label occurrences
            label_counts = {}
            for idx in top_indices:
                label = self.custom_index_manager['metadata'][idx]
                label_counts[label] = label_counts.get(label, 0) + 1

            logger.info(f"Custom search completed in {time.time() - start_time:.2f}s")

            return self.format_predictions(label_counts)

        except Exception as e:
            logger.error(f"Error in custom Faiss implementation: {str(e)}")
            logger.exception(e)
            # Fall back to original method
            return self.process_database_predictions(query_vec)

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def process_chunk(self, chunk_data, chunk_labels, query_vec, top_k_distances, top_k_labels):
        """Process a single chunk of binary codes and update top K results"""
        # Convert -1 to 0
        chunk_data = np.where(chunk_data == -1, 0, chunk_data)

        # Calculate distances for this chunk
        distances = np.sum(query_vec != chunk_data, axis=1)

        # Find potential top K in this chunk
        for i, (dist, label) in enumerate(zip(distances, chunk_labels)):
            # Find position to insert
            insert_idx = np.searchsorted(top_k_distances, dist)

            if insert_idx < len(top_k_distances):
                # Shift everything down
                top_k_distances[insert_idx + 1:] = top_k_distances[insert_idx:-1]
                top_k_labels[insert_idx + 1:] = top_k_labels[insert_idx:-1]

                # Insert new value
                top_k_distances[insert_idx] = dist
                top_k_labels[insert_idx] = label

        # Clean up
        del distances
        gc.collect()

    def format_predictions(self, label_counts):
        """Format prediction results as JSON with validation."""
        try:
            # Input validation
            if label_counts is None:
                raise ValueError("Label counts is None")
            if not label_counts:
                raise ValueError("Label counts is empty")

            logger.info("Formatting prediction results")
            logger.debug(f"Number of labels: {len(label_counts)}")

            # Calculate total count and validate
            total_count = sum(label_counts.values())
            if total_count == 0:
                raise ValueError("Total count is zero")

            try:
                # Calculate percentages
                percentages = {
                    label: round((count / total_count) * 100, 2)
                    for label, count in label_counts.items()
                }

                # Sort percentages
                sorted_percentages = dict(sorted(
                    percentages.items(),
                    key=lambda x: x[1],
                    reverse=True
                ))

                # Validate results
                if not sorted_percentages:
                    raise ValueError("No valid percentages calculated")

                # Convert to JSON
                result = json.dumps(sorted_percentages, indent=2)
                logger.debug(f"Top prediction: {next(iter(sorted_percentages.items()))}")
                return result

            except (TypeError, ValueError) as e:
                logger.error(f"Error in percentage calculation: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Error formatting predictions: {str(e)}")
            logger.error(f"Label counts: {label_counts}")
            raise