import logging
import torch
import numpy as np
from app.utils.gpu_utils import gpu_memory_manager

logger = logging.getLogger(__name__)

class PredictionOrchestrator:
    """Orchestrates the prediction workflow using all specialized services."""
    
    def __init__(
        self, 
        model_loader,
        image_processor,
        feature_extractor,
        attention_processor,
        prediction_service
    ):
        self.model_loader = model_loader
        self.image_processor = image_processor
        self.feature_extractor = feature_extractor
        self.attention_processor = attention_processor
        self.prediction_service = prediction_service
    
    def predict(self, image_path: str, param2=None, param3=None, device=None, not_segment=True):
        """Base predictor with single forward pass optimization."""
        with gpu_memory_manager():
            try:
                # Validate image path
                validated_path = self.image_processor.validate_image_path(image_path)
                
                if not_segment:
                    # Geographic prediction workflow
                    logger.info(f"Starting geographic prediction for: {image_path}")
                    
                    # Load and prepare image
                    original_image, image_tensor = (
                        self.image_processor.prepare_image_tensor(image_path)
                    )
                    
                    # Get the geo model
                    geo_model = self.model_loader.get_model('geo')
                    
                    # SINGLE forward pass for both attention and features
                    with torch.no_grad():
                        outputs, attention = geo_model(image_tensor, return_attention=True)
                        
                        # Compute attention map from the attention output (no model call)
                        attention_map = self.attention_processor.compute_attention_map(attention)
                        
                        # Save attention visualization
                        attention_map_path = (
                            self.attention_processor.save_attention_visualization(
                                attention_map, 
                                original_image
                            )
                        )
                        
                        # Extract features from outputs (reuse the same forward pass)
                        query_vec = outputs[0].sign().cpu().numpy()
                        query_vec = np.where(query_vec == -1, 0, query_vec)
                    
                    # Process database predictions
                    formatted_predictions = (
                        self.prediction_service
                        .process_database_predictions_custom_faiss(query_vec)
                    )
                    
                    return formatted_predictions, attention_map_path
                    
                else:
                    # Segment feature extraction workflow
                    logger.info(f"Starting segment feature extraction for: {image_path}")
                    
                    query_vec = self.feature_extractor.extract_features('dhn', image_path)
                    query_vec = np.where(query_vec == -1, 0, query_vec)
                    
                    return query_vec
                    
            except Exception as e:
                logger.error(f"Error in prediction: {str(e)}")
                raise