import torch
import logging
import gc
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from app.utils.gpu_utils import gpu_memory_manager

logger = logging.getLogger(__name__)

class LLaVAService:
    """Manages LLaVA model loading and inference."""
    
    def __init__(self, config):
        self.config = config
        self._llava_model = None
        self._tokenizer = None
        self._image_processor = None
        self.model_name = None
    
    def initialize(self):
        """Initialize LLaVA model and components."""
        logger.info(f"📍📍📍📍 LLaVA Model Path: {self.config.MODEL_PATH}")
        self.model_name = get_model_name_from_path(self.config.MODEL_PATH)
        self._load_llava_model()
    
    def _load_llava_model(self):
        """Load LLAVA model and its components."""
        with gpu_memory_manager():
            try:
                logger.info("Initializing LLAVA model...")
                model_path = self.config.MODEL_PATH
                
                # Controlled device mapping for mistral model
                model_kwargs = {
                    'load_4bit': True,
                    'device': "cuda",
                    'device_map': "cuda:0",
                    'use_flash_attn': False
                }
                
                # Load with quantization settings
                (self._tokenizer, 
                 self._llava_model, 
                 self._image_processor, 
                 _) = load_pretrained_model(
                    model_path,
                    None,
                    self.model_name,
                    **model_kwargs
                )
                
                logger.info("LLAVA model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load LLAVA model: {str(e)}")
                raise
    
    def invoke(self, args):
        """
        Enhanced LLaVA model invocation with proper cleanup.
        
        Args:
            args: Arguments for LLaVA model evaluation
            
        Returns:
            Model prediction response
        """
        from app.utils.llava_utils import eval_model_with_global_model
        
        try:
            # Force cleanup before LLaVA
            torch.cuda.empty_cache()
            gc.collect()
            
            if (not self._llava_model or 
                not self._tokenizer or 
                not self._image_processor):
                raise ValueError(
                    "LLaVA model components not properly initialized"
                )
            
            # Monitor memory
            before_mem = torch.cuda.memory_allocated()
            
            predicted_response = eval_model_with_global_model(
                args,
                self._llava_model,
                self._tokenizer,
                self._image_processor,
                self.model_name
            )
            
            after_mem = torch.cuda.memory_allocated()
            logger.info(
                f"Memory change: {(after_mem - before_mem) / 1e9:.2f}GB"
            )
            
            return predicted_response
            
        except Exception as e:
            logger.error(f"Error in LLaVA invocation: {str(e)}")
            raise
        finally:
            # Ensure cleanup even on error
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
    
    def cleanup(self):
        """Release LLaVA model resources."""
        self._llava_model = None
        self._tokenizer = None
        self._image_processor = None
        torch.cuda.empty_cache()