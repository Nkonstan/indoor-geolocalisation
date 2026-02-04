import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import cv2
import uuid
import os
import gc
from PIL import Image
from app.utils.gpu_utils import gpu_memory_manager

logger = logging.getLogger(__name__)

class AttentionProcessor:
    """Computes and visualizes attention maps from model outputs."""
    
    def __init__(self, config):
        self.config = config
    
    def compute_attention_map(self, attention_output):
        """
        Compute attention map from model attention output.
        
        Args:
            attention_output: Attention tensor from model (not the model itself)
            
        Returns:
            Processed attention map as numpy array
        """
        try:
            if attention_output is None:
                raise ValueError("Attention output is None")
            
            logger.info("Computing attention map")
            
            # Process attention (no model call needed)
            attention = attention_output.sum(dim=1).squeeze(0)
            attention_patches = attention[1:-1, 1:-1]
            attention_patches_sum = attention_patches.sum(dim=0)
            attention_map = attention_patches_sum.reshape(24, 24)
            
            # Move to CPU and process
            attention_map = attention_map.detach().cpu().numpy()
            attention_map = np.power(attention_map, 0.5)
            threshold = np.percentile(attention_map, 75)
            attention_map = np.where(
                attention_map > threshold, 
                attention_map, 
                0
            )
            
            return attention_map
            
        except Exception as e:
            logger.error(f"Error computing attention map: {str(e)}")
            raise
        finally:
            # Cleanup
            cleanup_vars = ['attention', 'attention_patches', 'attention_patches_sum']
            for var in cleanup_vars:
                if var in locals():
                    del locals()[var]
            torch.cuda.empty_cache()
    
    def save_attention_visualization(
        self, 
        attention_map: np.ndarray, 
        original_image: Image.Image
    ) -> str:
        """
        Save attention map visualization with proper image resizing.
        
        Args:
            attention_map: Computed attention map
            original_image: Original PIL Image
            
        Returns:
            Path to saved visualization
        """
        try:
            # Input validation
            if attention_map is None:
                raise ValueError("Attention map is None")
            if original_image is None:
                raise ValueError("Original image is None")
            
            logger.info("Processing attention map visualization")
            logger.debug(f"Attention map shape: {attention_map.shape}")
            logger.debug(f"Original image size: {original_image.size}")
            
            try:
                # Resize attention map to match original image dimensions
                attention_map_resized = cv2.resize(attention_map, (384, 384))
                attention_map_normalized = (
                    attention_map_resized * 255
                ).astype(np.uint8)
                del attention_map_resized
                
                # Convert original image to numpy array and resize
                original_array = np.array(
                    original_image.resize((384, 384), Image.LANCZOS)
                )
                
                # Handle RGBA to RGB conversion
                if (len(original_array.shape) == 3 and 
                    original_array.shape[2] == 4):
                    logger.info("Converting RGBA image to RGB")
                    original_array = original_array[:, :, :3]
                
                # Create heatmap
                heatmap = cv2.applyColorMap(
                    attention_map_normalized, 
                    cv2.COLORMAP_JET
                )
                del attention_map_normalized
                
                # Handle color conversion
                if (len(original_array.shape) == 3 and 
                    original_array.shape[2] == 3):
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                else:
                    original_array = cv2.cvtColor(
                        original_array, 
                        cv2.COLOR_GRAY2RGB
                    )
                
                # Zero out green and blue channels
                heatmap[:, :, 1] = 0  # Green channel
                heatmap[:, :, 2] = 0  # Blue channel
                
                # Verify shapes match before overlay
                if heatmap.shape != original_array.shape:
                    logger.warning(
                        f"Shape mismatch: heatmap {heatmap.shape} "
                        f"vs original {original_array.shape}"
                    )
                    # Resize to match
                    if heatmap.shape[:2] != original_array.shape[:2]:
                        original_array = cv2.resize(
                            original_array, 
                            (heatmap.shape[1], heatmap.shape[0])
                        )
                    # Ensure same channel count
                    if heatmap.shape[2] != original_array.shape[2]:
                        if original_array.shape[2] == 1:
                            original_array = cv2.cvtColor(
                                original_array, 
                                cv2.COLOR_GRAY2RGB
                            )
                        elif heatmap.shape[2] == 1:
                            heatmap = cv2.cvtColor(
                                heatmap, 
                                cv2.COLOR_GRAY2RGB
                            )
                
                # Create overlay
                overlay = cv2.addWeighted(heatmap, 0.6, original_array, 0.4, 0)
                del heatmap, original_array
                
                # Generate unique filename
                attention_map_name = str(uuid.uuid4()) + '.jpg'
                save_dir = 'static'
                os.makedirs(save_dir, exist_ok=True)
                attention_map_path = os.path.join(save_dir, attention_map_name)
                
                # Try OpenCV first (more efficient)
                success = cv2.imwrite(
                    attention_map_path,
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90]
                )
                
                # Fallback to matplotlib if OpenCV fails
                if not success:
                    plt.figure(figsize=(3.84, 3.84))
                    plt.imshow(overlay)
                    plt.axis('off')
                    plt.savefig(
                        attention_map_path,
                        transparent=False,
                        bbox_inches='tight',
                        pad_inches=0,
                        dpi=100
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
            logger.error(
                f"Attention map shape: "
                f"{attention_map.shape if attention_map is not None else 'None'}"
            )
            logger.error(
                f"Original image size: "
                f"{original_image.size if original_image is not None else 'None'}"
            )
            raise
        finally:
            # Clean up matplotlib resources
            plt.close('all')
            # Clean up large objects
            cleanup_vars = [
                'attention_map_resized', 'attention_map_normalized',
                'heatmap', 'original_array', 'overlay'
            ]
            for var in cleanup_vars:
                if var in locals():
                    del locals()[var]
            gc.collect()