import logging
import numpy as np
import cv2
from PIL import Image
import io

logger = logging.getLogger(__name__)

class WebImageProcessor:
    """Handles image manipulation operations."""
    
    def __init__(self, config):
        self.config = config
    
    def open_and_resize(self, file_content, max_size=1024):
        """Open image from bytes and resize."""
        try:
            file_bytes = io.BytesIO(file_content)
            image = Image.open(file_bytes)
            
            ratio = min(max_size / image.width, max_size / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)  # ALWAYS resize
        except Exception as e:
            logger.error(f"Error opening/resizing image: {str(e)}")
            raise
            
        return image
    
    def load_image_bgr_to_rgb(self, image_path):
        """
        Load image and convert BGR to RGB.
        
        Args:
            image_path: Path to image
            
        Returns:
            np.ndarray: Image in RGB format
        """
        try:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise
    
    def crop_to_non_black_region(self, segmented_image):
        """
        Crop image to non-black region.
        
        Args:
            segmented_image: np.ndarray image
            
        Returns:
            np.ndarray: Cropped image
        """
        try:
            non_black_mask = np.any(segmented_image > 0, axis=-1)
            
            if not np.any(non_black_mask):
                logger.warning("Image is completely black")
                return np.zeros_like(segmented_image)
            
            rows = np.any(non_black_mask, axis=1)
            cols = np.any(non_black_mask, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            cropped = segmented_image[ymin:ymax + 1, xmin:xmax + 1]
            logger.info(f"Cropped image from {segmented_image.shape} to {cropped.shape}")
            
            return cropped
            
        except Exception as e:
            logger.error(f"Error cropping image: {str(e)}")
            raise