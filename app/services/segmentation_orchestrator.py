import logging
import torch
from app.utils.gpu_utils import gpu_memory_manager

logger = logging.getLogger(__name__)

class SegmentationOrchestrator:
    """Orchestrates the automatic segmentation workflow."""
    
    def __init__(
        self, 
        model_service,
        external_service_client,
        similarity_search_service
    ):
        self.model_service = model_service
        self.external_service_client = external_service_client
        self.similarity_search_service = similarity_search_service
    
    def process_automatic_segmentation(self, image_path, targets=None):
        """
        Process automatic segmentation using Lang-SAM service.
        
        Args:
            image_path: Path to input image
            targets: List of targets to segment (default: ['floor', 'ceiling', 'door', 'window'])
            
        Returns:
            dict: Segmentation results with similar segments
        """
        if targets is None:
            targets = ['floor', 'ceiling', 'door', 'window']
        
        # Define variables before try block
        found_segments = {}
        segmentation_results = {}
        feature_vectors = {}
        
        with gpu_memory_manager():
            try:
                # Call Lang-SAM service
                results = self.external_service_client.get_langsam_segmentation(
                    image_path, 
                    targets
                )
                
                if results.get('status') != 'success':
                    logger.error(f"Lang-SAM service error: {results.get('error', 'Unknown error')}")
                    return {}
                
                # Collect found segments first
                found_segments = {
                    target: data 
                    for target, data in results['results'].items()
                    if data.get('found', False)
                }
                
                if not found_segments:
                    logger.info("No segments found")
                    return {}
                
                # Single model processing context for all segments
                with gpu_memory_manager():
                    # Process all segments at once
                    for target, data in found_segments.items():
                        # Get features for each segment (only DHN)
                        query_vec_dhn = self.model_service.base_predictor(
                            data['image_path'],
                            None,
                            None,
                            self.model_service.device,
                            not_segment=False,
                        )
                        feature_vectors[target] = query_vec_dhn
                    
                    # Process similar segments for all targets
                    for target, data in found_segments.items():
                        query_vec_dhn = feature_vectors[target]
                        
                        similar_segments = self.similarity_search_service.find_similar_segments(
                            target,
                            query_vec_dhn,
                            top_k=10
                        )
                        
                        segmentation_results[target] = {
                            'cropped_image_path': data['image_path'],
                            'similar_segments': similar_segments[:4],
                            'confidence': data['confidence'],
                            'similar_segments_info': [{
                                'path': segment['Full Path'],
                                'score': f"{segment['Average_Distance']:.3f}",
                                'country': segment['Country']
                            } for segment in similar_segments[:4]]
                        }
                    
                    # Cleanup feature vectors
                    del feature_vectors
                    torch.cuda.empty_cache()
                
                return segmentation_results
                
            except Exception as e:
                logger.error(f"Error in automatic segmentation: {str(e)}")
                return {}
            finally:
                # Ensure cleanup after processing all targets
                if 'feature_vectors' in locals():
                    del feature_vectors
                torch.cuda.empty_cache()