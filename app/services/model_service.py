import torch
import time
from app.config import Config
import logging
import numpy as np
import matplotlib.pyplot as plt
import cv2
import uuid
import os
from torchvision import transforms
from PIL import Image
from collections import Counter, defaultdict
import json
import pandas as pd
from scipy.spatial.distance import hamming
import pickle
import gc
import sys
import itertools
from contextlib import contextmanager
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from app.models.network import DeiT384_exp, ResNet  # Make sure this path is correct
from app.utils.gpu_utils import gpu_memory_manager
from app.utils.logging import setup_logging
from itertools import islice
from app.services.database_service import DatabaseService
from app.services.prediction_service import PredictionService
from pymongo import MongoClient
import faiss
# logger = logging.getLogger(__name__)
setup_logging()
logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.config = Config()
        self.device = self.config.DEVICE
        self._llava_model = None
        self._tokenizer = None
        self._image_processor = None
        self.current_model = None
        self.model_name = None
        self.transform = self._create_transform()
        # self.mongo_client = MongoClient(self.config.MONGODB_URI)
        # self.db = self.mongo_client[self.config.MONGODB_DATABASE]
        self.db_service = DatabaseService(self.config)
        # prediction service
        self.prediction_service = PredictionService(self)
        self._deit_base = None
        torch.manual_seed(69)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(69)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Initialize model containers
        self.models = {
            'geo': None,
            'dhn': None
        }
        self.state_dicts = {
            'geo': None,
            'dhn': None
        }

        # Initialize everything
        self._initialize_models()

    def _initialize_models(self):
        """Fully functional initialization with memory management"""
        try:
            # Load state dictionaries with memory management first
            logger.info("Loading model states...")
            with gpu_memory_manager():
                # Load state dicts to CPU
                self.state_dicts['geo'] = torch.load(self.config.GEO_MODEL_PATH, map_location='cpu')
                self.state_dicts['dhn'] = torch.load(self.config.DHN_MODEL_PATH, map_location='cpu')

                # Verify state dicts are loaded
                for model_type, state_dict in self.state_dicts.items():
                    if state_dict is None:
                        raise ValueError(f"Failed to load state dict for {model_type}")
                    logger.info(f"Successfully loaded state dict for {model_type}")

                # # Keep backward compatibility
                # self.geo_state = self.state_dicts['geo']
                # self.dhn_state = self.state_dicts['dhn']

            # Initialize models and load their weights on CPU first
            logger.info("Initializing models on CPU...")
            try:
                # Initialize DeiT base model once if not initialized
                if self._deit_base is None:
                    from transformers import DeiTModel
                    logger.info("Initializing DeiT base model...")
                    self._deit_base = DeiTModel.from_pretrained(
                        '/app/deit-base-distilled-patch16-384',  # local path inside the container
                        ignore_mismatched_sizes=True,
                        output_attentions=True
                    )
                    # self._deit_base = DeiTModel.from_pretrained(
                    #     'facebook/deit-base-distilled-patch16-384',
                    #     ignore_mismatched_sizes=True,
                    #     output_attentions=True
                    # )
                    self._deit_base.eval()
                    self._deit_base.cpu()
                    logger.info("DeiT base model initialized successfully")

                # Create model instances (removed hashnet)
                self.models['geo'] = DeiT384_exp(128, base_model=self._deit_base)
                # self.models['dhn'] = ResNet(512)
                self.models['dhn'] = ResNet(512, use_pretrained=False)
                # Verify models are created
                for model_type, model in self.models.items():
                    if model is None:
                        raise ValueError(f"Failed to create {model_type} model")
                    logger.info(f"Successfully created {model_type} model architecture")

                # Set traditional references first (keep on CPU)
                self.net = self.models['geo']
                self.net_dhn = self.models['dhn']

                # Load states with memory management
                with gpu_memory_manager():
                    for model_type in self.models:
                        logger.info(f"Loading state dict for {model_type} model")
                        self.models[model_type].load_state_dict(self.state_dicts[model_type])
                        self.models[model_type].eval()
                        self.models[model_type].cpu()  # Ensure on CPU
                        logger.info(f"Successfully initialized {model_type} model")

                # Verify models are properly loaded
                for model_type in self.models:
                    if not hasattr(self.models[model_type], 'state_dict'):
                        raise ValueError(f"Model {model_type} not properly initialized")
                    logger.info(f"Verified {model_type} model initialization")

            except Exception as e:
                logger.error(f"Error initializing models: {str(e)}")
                # Clear any partially initialized models
                self.models = {k: None for k in self.models}
                self.net = None
                self.net_dhn = None
                raise

            # Verify final model states
            logger.info("Verifying final model states...")
            if self.net is None or self.net_dhn is None:
                raise ValueError("One or more model references are None after initialization")

            if self.models['geo'] is None or self.models['dhn'] is None:
                raise ValueError("One or more models dict entries are None after initialization")

            # Initialize LLAVA model after other models are set up
            logger.info("Loading LLAVA model...")
            self.model_name = get_model_name_from_path(self.config.MODEL_PATH)
            self._load_llava_model()

            # # MongoDB initialization
            # logger.info("Loading segmentation databases from MongoDB...")
            # self._initialize_mongodb_data()

            logger.info("All models loaded successfully")
            logger.info(f"Model states: geo={self.net is not None}, dhn={self.net_dhn is not None}")

        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise


    def switch_model(self, model_type):
        """Enhanced model switching with proper device management"""
        try:
            # Validate model type
            if model_type not in self.models:
                raise ValueError(f"Invalid model type: {model_type}")

            # Skip if same model
            if self.current_model == model_type:
                return

            logger.info(f"Switching to model {model_type}. Current model: {self.current_model}")
            logger.info(f"Current GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

            # ADDED: Ensure thorough cleanup of current model
            self._unload_current_model()

            try:
                # Create new model instance
                if model_type == 'geo':
                    # Initialize DeiT if needed
                    if self._deit_base is None:
                        from transformers import DeiTModel
                        self._deit_base = DeiTModel.from_pretrained(
                            '/app/deit-base-distilled-patch16-384',
                            ignore_mismatched_sizes=True,
                            output_attentions=True
                        )
                        self._deit_base.eval()

                    # Create model instance first
                    self.models[model_type] = DeiT384_exp(128, base_model=self._deit_base)
                    # Load state dict while model is on CPU
                    self.models[model_type].load_state_dict(self.state_dicts[model_type])
                    self.models[model_type].eval()
                    # Then move everything to GPU
                    self._deit_base.to(self.device)
                    self.models[model_type].to(self.device)
                else:
                    self.models[model_type] = ResNet(512)
                    # Load state dict while on CPU
                    self.models[model_type].load_state_dict(self.state_dicts[model_type])
                    self.models[model_type].eval()
                    # Then move to GPU
                    self.models[model_type].to(self.device)

                # Validate model creation
                if self.models[model_type] is None:
                    raise ValueError(f"Failed to create {model_type} model")

                # Update current model reference
                self.current_model = model_type
                # Set attribute reference
                attr_name = f"net{'_' + model_type if model_type != 'geo' else ''}"
                setattr(self, attr_name, self.models[model_type])

                logger.info(f"Successfully switched to {model_type} model")
                logger.info(f"Final GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

            except Exception as e:
                logger.error(f"Error creating/loading {model_type} model: {e}")
                self._unload_current_model()
                raise
        except Exception as e:
            logger.error(f"Error in switch_model: {e}")
            raise

    def _unload_current_model(self):
        """Enhanced model unloading with complete cleanup"""
        try:
            if self.current_model is not None:
                # Get model attribute name based on current model
                attr_name = f"net{'_' + self.current_model if self.current_model != 'geo' else ''}"

                # Unload current model
                if hasattr(self, attr_name):
                    model = getattr(self, attr_name)
                    if model is not None:
                        # Move model to CPU
                        model.cpu()
                        # Remove reference
                        delattr(self, attr_name)
                        self.models[self.current_model] = None
                        del model

                # ADDED: Also clean DeiT base model if exists
                if hasattr(self, '_deit_base') and self._deit_base is not None:
                    self._deit_base.cpu()
                    self._deit_base = None

                # Reset current model reference
                self.current_model = None

                # Force cleanup
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            logger.error(f"Error in _unload_current_model: {e}")
            raise

    def _load_llava_model(self):
        """Load LLAVA model and its components."""
        with gpu_memory_manager():  # ensures cleanup happens no matter what
            try:
                logger.info("Initializing LLAVA model...")
                model_path = self.config.MODEL_PATH
                # More controlled device mapping for mistral model
                model_kwargs = {
                    'load_4bit': True,
                    'device': "cuda",
                    'device_map': "cuda:0",  # Explicit mapping to first GPU
                    'use_flash_attn': False  # Disable flash attention for more stable memory usage
                }

                # Load with quantization settings
                self._tokenizer, self._llava_model, self._image_processor, _ = load_pretrained_model(
                    model_path,
                    None,
                    self.model_name,
                    **model_kwargs
                )

                logger.info(f"LLAVA model loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load LLAVA model: {str(e)}")
                raise

    def _create_transform(self):
        return transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def get_feature_vector(self, model_type, image_path):
        """Memory-efficient feature extraction with enhanced validation"""
        with gpu_memory_manager():
            try:
                # Validate model type and image path
                if model_type not in self.models:
                    raise ValueError(f"Invalid model type: {model_type}")
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")

                logger.info(f"Starting feature extraction for {model_type}")

                # Process image on CPU
                try:
                    img = Image.open(image_path).convert('RGB')
                    img_tensor = self.transform(img).unsqueeze(0)
                    del img
                except Exception as e:
                    logger.error(f"Failed to process image: {e}")
                    raise

                # Validate model exists and move to device
                model = self.models[model_type]
                if model is None:
                    raise ValueError(f"Model {model_type} is not initialized")

                try:
                    # model.to(self.device)
                    img_tensor = img_tensor.to(self.device)
                except Exception as e:
                    logger.error(f"Failed to move tensors to device: {e}")
                    raise

                # Extract features
                try:
                    with torch.no_grad():
                        features = model(img_tensor)[0].sign().cpu().numpy()
                except Exception as e:
                    logger.error(f"Failed to extract features: {e}")
                    raise

                return features

            except Exception as e:
                logger.error(f"Feature extraction failed for {model_type}: {str(e)}")
                raise
            finally:
                # Cleanup
                try:
                    if 'model' in locals() and model is not None:
                        model.cpu()
                    cleanup_vars = ['img_tensor', 'img', 'model']
                    for var in cleanup_vars:
                        if var in locals():
                            del locals()[var]
                except Exception as e:
                    logger.error(f"Cleanup failed in feature extraction: {e}")
                torch.cuda.empty_cache()
                gc.collect()

    def compute_attention_map(self, net, image_tensor):
        """Compute attention map with enhanced validation and error handling"""
        try:
            # Validate inputs
            if net is None:
                raise ValueError("Model (net) is None")
            if image_tensor is None:
                raise ValueError("Image tensor is None")
            if not image_tensor.is_cuda:
                raise ValueError("Image tensor must be on GPU")
            logger.info("Computing attention map")
            with torch.no_grad():  # Keep this safety measure
                output, attention = net(image_tensor, return_attention=True)

                if attention is None:
                    raise ValueError("Model returned None attention map")

                # Keep operations on GPU until final numpy conversion
                attention = attention.sum(dim=1).squeeze(0)
                attention_patches = attention[1:-1, 1:-1]
                attention_patches_sum = attention_patches.sum(dim=0)
                attention_map = attention_patches_sum.reshape(24, 24)

                # Move to CPU only at the end and combine operations
                attention_map = attention_map.detach().cpu().numpy()
                attention_map = np.power(attention_map, 0.5)
                threshold = np.percentile(attention_map, 75)
                attention_map = np.where(attention_map > threshold, attention_map, 0)

                return attention_map

        except Exception as e:
            logger.error(f"Error computing attention map: {str(e)}")
            logger.error(f"Net type: {type(net)}")
            logger.error(f"Image tensor shape: {image_tensor.shape if image_tensor is not None else 'None'}")
            raise
        finally:
            cleanup_vars = ['output', 'attention', 'attention_patches', 'attention_patches_sum']
            for var in cleanup_vars:
                if var in locals():
                    del locals()[var]
            torch.cuda.empty_cache()

    def base_predictor(self, image_path, _, __, device, not_segment=True):
        """Base predictor with enhanced validation and memory management"""
        with gpu_memory_manager():
            try:
                # Validate inputs
                if not image_path or not os.path.exists(image_path):
                    raise FileNotFoundError(f"Invalid image path: {image_path}")

                if not_segment:
                    total_start = time.time()
                    logger.info(f"Starting geographic prediction for: {image_path}")

                    # Load and process image
                    image_load_start = time.time()
                    try:
                        with gpu_memory_manager():
                            original_image = Image.open(image_path).convert('RGB')
                            image_tensor = self.transform(original_image).unsqueeze(0)
                            del original_image
                        logger.info(f"Image loading and transformation took: {time.time() - image_load_start:.2f}s")
                    except Exception as e:
                        logger.error(f"Failed to load/process image: {e}")
                        raise
                    # Process with geo model and extract features
                    attention_map_path = None
                    try:
                        model_process_start = time.time()
                        with gpu_memory_manager():
                            # Switch model and get reference
                            model_switch_start = time.time()
                            self.switch_model('geo')
                            current_model = self.models['geo']
                            logger.info(f"Model switching took: {time.time() - model_switch_start:.2f}s")

                            # Compute attention map
                            attention_start = time.time()
                            image_tensor = image_tensor.to(self.device)
                            with torch.no_grad():
                                outputs, attention = current_model(image_tensor, return_attention=True)
                                attention_map = self.compute_attention_map(current_model, image_tensor)
                                attention_map_path = self._save_attention_map(attention_map, Image.open(image_path))
                            logger.info(
                                f"Attention map computation and saving took: {time.time() - attention_start:.2f}s")

                            # Extract features
                            features_start = time.time()
                            query_vec = current_model(image_tensor)[0].sign().cpu().numpy()
                            query_vec = np.where(query_vec == -1, 0, query_vec)
                            logger.info(f"Feature extraction took: {time.time() - features_start:.2f}s")

                            logger.info(f"Total model processing took: {time.time() - model_process_start:.2f}s")
                    except Exception as e:
                        logger.error(f"Failed in geo model processing: {e}")
                        raise
                    finally:
                        # Cleanup tensors but don't unload model yet
                        cleanup_vars = ['image_tensor', 'outputs', 'attention', 'attention_map']
                        for var in cleanup_vars:
                            if var in locals():
                                del locals()[var]
                        torch.cuda.empty_cache()

                    # Database processing
                    try:
                        db_start = time.time()
                        with gpu_memory_manager():
                            # formatted_predictions = self._process_database_predictions(query_vec)
                            # formatted_predictions = self.prediction_service.process_database_predictions(query_vec)
                            formatted_predictions = self.prediction_service.process_database_predictions_custom_faiss(query_vec)

                        logger.info(f"Database processing took: {time.time() - db_start:.2f}s")
                    except Exception as e:
                        logger.error(f"Failed in database processing: {e}")
                        raise
                    finally:
                        # Now we can safely unload the model
                        model_unload_start = time.time()
                        self._unload_current_model()
                        logger.info(f"Model unloading took: {time.time() - model_unload_start:.2f}s")

                    logger.info(f"Total geographic processing took: {time.time() - total_start:.2f}s")
                    return formatted_predictions, attention_map_path
                else:
                    # Segment feature extraction
                    try:
                        with gpu_memory_manager():
                            # Switch to dhn model
                            self.switch_model('dhn')
                            if self.models['dhn'] is None:
                                raise ValueError("DHN model is not initialized")
                            query_vec = self.get_feature_vector('dhn', image_path)
                            query_vec = np.where(query_vec == -1, 0, query_vec)
                            return query_vec

                    except Exception as e:
                        logger.error(f"Failed in segment feature extraction: {e}")
                        raise
            except Exception as e:
                logger.error(f"Error in base predictor: {str(e)}")
                raise
            finally:
                self._unload_current_model()
                gc.collect()
                torch.cuda.empty_cache()

    def _save_attention_map(self, attention_map, original_image):
        """Save attention map visualization with proper image resizing and memory management."""
        try:
            # Input validation
            if attention_map is None:
                raise ValueError("Attention map is None")
            if original_image is None:
                raise ValueError("Original image is None")

            logger.info("Processing attention map")
            logger.debug(f"Attention map shape: {attention_map.shape}")
            logger.debug(f"Original image size: {original_image.size}")

            try:
                # Resize attention map to match original image dimensions
                attention_map_resized = cv2.resize(attention_map, (384, 384))
                attention_map_normalized = (attention_map_resized * 255).astype(np.uint8)
                del attention_map_resized  # Clean up intermediate results

                # Convert original image to numpy array and resize
                original_array = np.array(original_image.resize((384, 384), Image.LANCZOS))

                # ADDED: Check for RGBA image and convert to RGB if needed
                if len(original_array.shape) == 3 and original_array.shape[2] == 4:
                    logger.info("Converting RGBA image to RGB")
                    original_array = original_array[:, :, :3]  # Keep only RGB channels

                # Create heatmap
                heatmap = cv2.applyColorMap(attention_map_normalized, cv2.COLORMAP_JET)
                del attention_map_normalized

                # Handle color conversion
                if len(original_array.shape) == 3 and original_array.shape[2] == 3:
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                else:
                    original_array = cv2.cvtColor(original_array, cv2.COLOR_GRAY2RGB)

                # Zero out green and blue channels
                heatmap[:, :, 1] = 0  # Green channel
                heatmap[:, :, 2] = 0  # Blue channel

                # Verify shapes match before overlay
                if heatmap.shape != original_array.shape:
                    logger.warning(f"Shape mismatch: heatmap {heatmap.shape} vs original {original_array.shape}")
                    # Resize original array to match heatmap if needed
                    if heatmap.shape[:2] != original_array.shape[:2]:
                        original_array = cv2.resize(original_array, (heatmap.shape[1], heatmap.shape[0]))
                    # Ensure same channel count
                    if heatmap.shape[2] != original_array.shape[2]:
                        if original_array.shape[2] == 1:
                            original_array = cv2.cvtColor(original_array, cv2.COLOR_GRAY2RGB)
                        elif heatmap.shape[2] == 1:
                            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)

                # Create overlay
                overlay = cv2.addWeighted(heatmap, 0.6, original_array, 0.4, 0)
                del heatmap, original_array

                # # Generate unique filename and ensure directory exists '.png'
                # attention_map_name = str(uuid.uuid4()) + '.jpg'
                # save_dir = 'static'
                # os.makedirs(save_dir, exist_ok=True)
                # attention_map_path = os.path.join(save_dir, attention_map_name)
                #
                # # Save the result
                # plt.figure(figsize=(10, 10))
                # plt.imshow(overlay)
                # plt.axis('off')
                # plt.savefig(
                #     attention_map_path,
                #     transparent=True,
                #     bbox_inches='tight',
                #     pad_inches=0,
                #     dpi=300
                # )
                # plt.close()
                # Generate unique filename and ensure directory exists
                attention_map_name = str(uuid.uuid4()) + '.jpg'  # Changed from PNG to JPG
                save_dir = 'static'
                os.makedirs(save_dir, exist_ok=True)
                attention_map_path = os.path.join(save_dir, attention_map_name)

                # Try to save with OpenCV first (more efficient)
                success = cv2.imwrite(attention_map_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])

                # Fallback to matplotlib if OpenCV fails
                if not success:
                    # Calculate figure size to maintain 384x384 pixels at 100 DPI
                    # Figure size = Pixels / DPI = 384/100 = 3.84 inches
                    plt.figure(figsize=(3.84, 3.84))  # This will create a 384x384 pixel image at 100 DPI
                    plt.imshow(overlay)
                    plt.axis('off')
                    plt.savefig(
                        attention_map_path,
                        transparent=False,  # No transparency for JPG
                        bbox_inches='tight',
                        pad_inches=0,
                        dpi=100  # Lower DPI (was 300)
                    )
                    plt.close()
                del overlay

                if not os.path.exists(attention_map_path):
                    raise FileNotFoundError("Failed to save attention map")

                logger.info(f"Attention map saved to {attention_map_path}")
                return attention_map_path

            except (cv2.error, ValueError) as e:
                logger.error(f"Image processing error: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Error saving attention map: {str(e)}")
            logger.error(f"Attention map shape: {attention_map.shape if attention_map is not None else 'None'}")
            logger.error(f"Original image size: {original_image.size if original_image is not None else 'None'}")
            raise
        finally:
            # Clean up matplotlib resources
            plt.close('all')
            # Clean up any remaining large objects
            cleanup_vars = [
                'attention_map_resized', 'attention_map_normalized',
                'heatmap', 'original_array', 'overlay'
            ]
            for var in cleanup_vars:
                if var in locals():
                    del locals()[var]
            gc.collect()

    def cleanup(self):
        """Release resources and close MongoDB connection."""
        try:
            self.db_service.cleanup()
            # self.mongo_client.close()
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {str(e)}")

        # Existing cleanup code
        self._llava_model = None
        self._tokenizer = None
        self._image_processor = None
        torch.cuda.empty_cache()


    def invoke_llava_model(self, args):
        """Enhanced LLaVA model invocation with proper cleanup"""
        from llava.eval.run_llava import eval_model_with_global_model  # Add this import
        try:
            # ADDED: Force cleanup before LLaVA
            self._unload_current_model()
            torch.cuda.empty_cache()
            gc.collect()

            if not self._llava_model or not self._tokenizer or not self._image_processor:
                raise ValueError("LLaVA model components not properly initialized")

            # Monitor memory before and after
            before_mem = torch.cuda.memory_allocated()

            predicted_response = eval_model_with_global_model(
                args,
                self._llava_model,
                self._tokenizer,
                self._image_processor,
                self.model_name
            )

            after_mem = torch.cuda.memory_allocated()
            logger.info(f"Memory change: {(after_mem - before_mem) / 1e9:.2f}GB")

            return predicted_response

        except Exception as e:
            logger.error(f"Error in ModelService.invoke_llava_model: {str(e)}")
            raise
        finally:
            # ADDED: Ensure cleanup even on error
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()