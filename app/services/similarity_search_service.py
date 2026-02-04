import os
import time
import logging
import numpy as np
import gc
import torch

logger = logging.getLogger(__name__)

class SimilaritySearchService:
    """Handles similarity search using Faiss indexes."""
    
    def __init__(self, index_manager):
        self.index_manager = index_manager
    
    def find_similar_segments(self, target, query_vec_dhn, top_k=10):
        """
        Find similar segments using Faiss for ultra-fast similarity search.
        
        Args:
            target: Segment type (e.g., 'floor', 'ceiling')
            query_vec_dhn: Query feature vector
            top_k: Number of results to return
            
        Returns:
            list: List of similar segments with metadata
        """
        try:
            logger.info(f"\n=== Starting Faiss Similarity Search for {target} ===")
            start_time = time.time()
            
            # Get index and metadata for this segment type
            index, metadata = self.index_manager.get_or_build_index(target)
            
            # Convert query vector to the right format
            query = np.array([query_vec_dhn]).astype(np.float32)
            
            # Search for nearest neighbors
            distances, indices = index.search(query, top_k)
            
            # Get results
            closest_segments = []
            for i, idx in enumerate(indices[0]):
                if idx < len(metadata):  # Safety check
                    segment_data = metadata[idx]
                    closest_segments.append({
                        'Image Name': segment_data['Image Name'],
                        'Distance': float(distances[0][i]),  # L2 distance
                        'Segment': segment_data['Segment'],
                        'Country': segment_data['Country']
                    })
            
            # Log timing and results
            search_time = time.time() - start_time
            logger.info(f"Faiss search completed in {search_time:.3f}s")
            logger.info(f"Database size - DHN: {len(metadata)}")
            
            # Log final results
            logger.info("\nFinal Top 5 matches:")
            for segment in closest_segments[:5]:
                logger.info(f"Image: {segment['Image Name']}")
                logger.info(f"  DHN Distance: {segment['Distance']:.3f}")
                logger.info(f"  Country: {segment['Country']}")
            
            # Format result
            result = []
            for segment in closest_segments:
                full_path = os.path.join(
                    segment['Country'],
                    segment['Segment'],
                    segment['Image Name']
                )
                result.append({
                    'Full Path': full_path,
                    'Average_Distance': float(segment['Distance']),
                    'Country': segment['Country']
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
        finally:
            # Clean all intermediate data
            if 'closest_segments' in locals():
                del closest_segments
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()