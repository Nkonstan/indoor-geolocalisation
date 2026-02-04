import torch
import logging
from app.config import Config
from app.services.database_service import DatabaseService
from app.services.prediction_service import PredictionService
from app.services.model_loader import ModelLoader
from app.services.image_processor import ImageProcessor
from app.services.feature_extractor import FeatureExtractor
from app.services.attention_processor import AttentionProcessor
from app.services.llava_service import LLaVAService
from app.services.prediction_orchestrator import PredictionOrchestrator
from app.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class ModelService:
    """
    Main orchestrator for all model-related operations.
    Delegates to specialized services for specific tasks.
    """
    
    def __init__(self):
        self.config = Config()
        self.device = self.config.DEVICE
        
        # Initialize database service
        self.db_service = DatabaseService(self.config)
        
        # Initialize all specialized services
        self.model_loader = ModelLoader(self.config)
        self.image_processor = ImageProcessor(self.config)
        self.feature_extractor = FeatureExtractor(
            self.model_loader, 
            self.image_processor
        )
        self.attention_processor = AttentionProcessor(self.config)
        self.llava_service = LLaVAService(self.config)
        
        # Initialize prediction service (needs self reference)
        self.prediction_service = PredictionService(self)
        
        # Initialize prediction orchestrator
        self.prediction_orchestrator = PredictionOrchestrator(
            self.model_loader,
            self.image_processor,
            self.feature_extractor,
            self.attention_processor,
            self.prediction_service
        )
        
        # Initialize all models
        self._initialize_all_services()
    
    def _initialize_all_services(self):
        """Initialize all models and services."""
        try:
            logger.info("Initializing ModelService and all sub-services...")
            
            # Load ML models
            self.model_loader.initialize_all_models()
            
            # Load LLaVA model
            self.llava_service.initialize()
            
            logger.info("All services initialized successfully")
            
        except Exception as e:
            logger.error(f"Service initialization failed: {str(e)}")
            raise

    
    @property
    def net(self):
        """get geo model."""
        return self.model_loader.get_model('geo')
    
    @property
    def net_dhn(self):
        """get dhn model."""
        return self.model_loader.get_model('dhn')
    
    @property
    def transform(self):
        """get image transform."""
        return self.image_processor.transform
    
    def get_feature_vector(self, model_type: str, image_path: str):
        """
        extract features.
        Delegates to FeatureExtractor.
        """
        return self.feature_extractor.extract_features(model_type, image_path)
    
    def compute_attention_map(self, attention_output):
        """Just delegate to AttentionProcessor."""
        return self.attention_processor.compute_attention_map(attention_output)
    
    def _save_attention_map(self, attention_map, original_image):
        """
        save attention visualization.
        Delegates to AttentionProcessor.
        """
        return self.attention_processor.save_attention_visualization(
            attention_map, 
            original_image
        )
    
    def base_predictor(self, image_path, param2, param3, device, not_segment=True):
        """
        run prediction.
        Delegates to PredictionOrchestrator.
        """
        return self.prediction_orchestrator.predict(
            image_path, 
            param2, 
            param3, 
            device, 
            not_segment
        )
    
    def invoke_llava_model(self, args):
        """
        invoke LLaVA.
        Delegates to LLaVAService.
        """
        return self.llava_service.invoke(args)
    
    # ========== Cleanup ==========
    
    def cleanup(self):
        """Release all resources."""
        try:
            logger.info("Cleaning up ModelService...")
            
            # Cleanup database service
            self.db_service.cleanup()
            
            # Cleanup LLaVA
            self.llava_service.cleanup()
            
            # Cleanup model loader
            self.model_loader.cleanup()
            
            logger.info("ModelService cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")