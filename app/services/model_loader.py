import torch
import logging
from transformers import DeiTModel
from app.models.network import DeiT384_exp, ResNet
from app.utils.gpu_utils import gpu_memory_manager
import gc

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles loading, unloading, and lifecycle of ML models."""
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self._deit_base = None
        
        # Model containers
        self.models = {
            'geo': None,
            'dhn': None
        }
        self.state_dicts = {
            'geo': None,
            'dhn': None
        }
        
        # Set random seeds
        torch.manual_seed(69)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(69)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def initialize_all_models(self):
        """Load all models on GPU and keep them there"""
        logger.info(" STARTING MODEL LOADING  ")
        try:
            logger.info("Loading all models on GPU...")
            with gpu_memory_manager():
                # Load state dicts to CPU first
                self.state_dicts['geo'] = torch.load(
                    self.config.GEO_MODEL_PATH, 
                    map_location='cpu'
                )
                self.state_dicts['dhn'] = torch.load(
                    self.config.DHN_MODEL_PATH, 
                    map_location='cpu'
                )
                
                # Initialize DeiT base on GPU
                if self._deit_base is None:
                    logger.info("Initializing DeiT base model on GPU...")
                    self._deit_base = DeiTModel.from_pretrained(
                        self.config.DEIT_MODEL_PATH,
                        ignore_mismatched_sizes=True,
                        output_attentions=True
                    )
                    self._deit_base.eval()
                    self._deit_base.to(self.device)
                
                # Create and load geo model
                self.models['geo'] = DeiT384_exp(128, base_model=self._deit_base)
                self.models['geo'].load_state_dict(self.state_dicts['geo'])
                self.models['geo'].eval()
                self.models['geo'].to(self.device)
                
                # Create and load dhn model
                self.models['dhn'] = ResNet(512, use_pretrained=False)
                self.models['dhn'].load_state_dict(self.state_dicts['dhn'])
                self.models['dhn'].eval()
                self.models['dhn'].to(self.device)
                
                logger.info("All models loaded and kept on GPU")
                
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise
    
    def get_model(self, model_type: str):
        """Get a specific model by type."""
        if model_type not in self.models:
            raise ValueError(f"Invalid model type: {model_type}")
        
        if self.models[model_type] is None:
            raise ValueError(f"Model {model_type} is not initialized")
        
        return self.models[model_type]
    
    def is_model_loaded(self, model_type: str) -> bool:
        """Check if a model is loaded."""
        return (model_type in self.models and 
                self.models[model_type] is not None)
    
    def cleanup(self):
        """Simplified cleanup - just clear GPU cache"""
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")