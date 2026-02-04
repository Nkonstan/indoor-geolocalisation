import os
import time
import logging
import threading
import pickle
import numpy as np
import faiss

logger = logging.getLogger(__name__)

class SegmentIndexManager:
    """Manages Faiss indexes for fast similarity search."""
    
    def __init__(self, model_service):
        self.model_service = model_service
        self.indexes = {}  # Dictionary of {segment_type: faiss_index}
        self.metadata = {}  # Dictionary of {segment_type: list_of_metadata}
        self.index_ready = {}  # Track which indexes are ready
        self.index_lock = threading.RLock()  # Thread safety
        self.index_dir = model_service.config.FAISS_INDEX_DIR
        os.makedirs(self.index_dir, exist_ok=True)
    
    def get_or_build_index(self, segment_type):
        """
        Get an existing index or build it if needed.
        
        Args:
            segment_type: Type of segment (e.g., 'floor', 'ceiling')
            
        Returns:
            tuple: (faiss_index, metadata_list)
        """
        with self.index_lock:
            # Check if index exists and is ready
            if (segment_type in self.indexes and 
                self.index_ready.get(segment_type, False)):
                return self.indexes[segment_type], self.metadata[segment_type]
            
            # Try to load from disk first
            index_path = os.path.join(self.index_dir, f"{segment_type}_index.bin")
            meta_path = os.path.join(self.index_dir, f"{segment_type}_metadata.pkl")
            
            if os.path.exists(index_path) and os.path.exists(meta_path):
                try:
                    # Load index and metadata
                    self.indexes[segment_type] = faiss.read_index(index_path)
                    with open(meta_path, 'rb') as f:
                        self.metadata[segment_type] = pickle.load(f)
                    self.index_ready[segment_type] = True
                    
                    logger.info(f"Loaded index for {segment_type} from disk")
                    return self.indexes[segment_type], self.metadata[segment_type]
                except Exception as e:
                    logger.error(f"Error loading index for {segment_type}: {e}")
            
            # Need to build the index
            logger.info(f"Building Faiss index for {segment_type}...")
            start_time = time.time()
            
            # Collect all vectors and metadata
            all_vectors = []
            all_metadata = []
            
            # Process in batches to avoid memory issues
            for batch in self.model_service.db_service.get_segmentation_data(
                self.model_service.config.MONGODB_DHN_COLLECTION, 
                segment_type
            ):
                for doc in batch:
                    all_vectors.append(doc['Binary Code'])
                    all_metadata.append({
                        'Image Name': doc['Image Name'],
                        'Segment': doc['Segment'],
                        'Country': doc['Country']
                    })
            
            # Convert to numpy array
            vectors = np.vstack(all_vectors).astype(np.float32)
            
            # Create and train index (IndexFlatL2 for exact search)
            dimension = vectors.shape[1]
            index = faiss.IndexFlatL2(dimension)
            
            # Add vectors to index
            index.add(vectors)
            
            # Save index and metadata
            faiss.write_index(index, index_path)
            with open(meta_path, 'wb') as f:
                pickle.dump(all_metadata, f)
            
            # Store in memory
            self.indexes[segment_type] = index
            self.metadata[segment_type] = all_metadata
            self.index_ready[segment_type] = True
            
            logger.info(
                f"Built index for {segment_type} with {len(all_metadata)} "
                f"vectors in {time.time() - start_time:.2f}s"
            )
            
            return index, all_metadata