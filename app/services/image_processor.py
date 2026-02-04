import os
import logging
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image loading, validation, and preprocessing."""
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.transform = self._create_transform()
    
    def _create_transform(self):
        """Create the image transformation pipeline."""
        return transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def validate_image_path(self, image_path: str) -> Path:
        """Validate that image path exists and is a supported format."""
        if not image_path:
            raise ValueError("Image path cannot be empty")
        
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        if path.suffix.lower() not in supported_formats:
            raise ValueError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported formats: {supported_formats}"
            )
        
        return path
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load and validate an image file."""
        validated_path = self.validate_image_path(image_path)
        
        try:
            img = Image.open(validated_path).convert('RGB')
            logger.info(f"Loaded image: {validated_path.name}, size: {img.size}")
            return img
        except Exception as e:
            logger.error(f"Failed to load image {validated_path}: {str(e)}")
            raise
    
    def prepare_image_tensor(self, image_path: str):
        """Load image and convert to tensor ready for model input."""
        img = self.load_image(image_path)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img, img_tensor