import torch
import logging
import numpy as np
from app.utils.gpu_utils import gpu_memory_manager

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extracts feature vectors from images using trained models."""
    
    def __init__(self, model_loader, image_processor):
        self.model_loader = model_loader
        self.image_processor = image_processor
    
    def extract_features(self, model_type: str, image_path: str) -> np.ndarray:
        """
        Extract feature vector from an image using specified model.
        
        Args:
            model_type: Type of model to use ('geo' or 'dhn')
            image_path: Path to the image file
            
        Returns:
            Binary feature vector as numpy array
        """
        with gpu_memory_manager():
            try:
                # Get the model
                model = self.model_loader.get_model(model_type)
                
                # Process image
                _, img_tensor = self.image_processor.prepare_image_tensor(image_path)
                
                # Extract features with no gradient computation
                with torch.no_grad():
                    features = model(img_tensor)[0].sign().cpu().numpy()
                
                logger.info(
                    f"Extracted features using {model_type} model, "
                    f"shape: {features.shape}"
                )
                
                return features
                
            except Exception as e:
                logger.error(
                    f"Feature extraction failed for {model_type}: {str(e)}"
                )
                raise