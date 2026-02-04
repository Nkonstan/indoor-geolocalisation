import os
import io
import uuid
import base64
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class ImageFileHandler:
    """Handles all file I/O operations for images."""
    
    def __init__(self, config):
        self.config = config
        self.static_dir = config.STATIC_DIR
        os.makedirs(self.static_dir, exist_ok=True)
    
    def validate_and_read_upload(self, file):
        """
        Validate and read uploaded file.
        
        Args:
            file: Flask file upload object
            
        Returns:
            bytes: File content as bytes
        """
        try:
            # Initial validation
            if not file or not file.content_type.startswith('image/'):
                raise ValueError("No valid image file provided")
            
            # Read file in chunks to reduce memory spikes
            chunks = []
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := file.read(chunk_size):
                chunks.append(chunk)
            
            file_content = b''.join(chunks)
            del chunks
            
            if not file_content:
                raise ValueError("Empty file content")
            
            # Verify it's a valid image
            file_bytes = io.BytesIO(file_content)
            with Image.open(file_bytes) as image:
                image.verify()
            file_bytes.seek(0)
            
            logger.info(f"Successfully validated uploaded file")
            return file_content
            
        except Exception as e:
            logger.error(f"Error validating upload: {str(e)}")
            raise
    
    def save_image(self, image, extension='png'):
        """
        Save PIL Image to disk with unique filename.
        
        Args:
            image: PIL Image object
            extension: File extension (default: 'png')
            
        Returns:
            str: Path to saved image
        """
        try:
            filename = str(uuid.uuid4()) + f'.{extension}'
            path = os.path.join(self.static_dir, filename)
            
            image.save(path, format=extension.upper(), optimize=True)
            logger.info(f"Image saved to {path}")
            
            return path
            
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            raise
    
    def encode_image_to_base64(self, image_path):
        """
        Encode image file to base64 string.
        
        Args:
            image_path: Path to image file
            
        Returns:
            str: Base64 encoded image or None if file not found
        """
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"Image not found for encoding: {image_path}")
            return None
        
        try:
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image to base64: {str(e)}")
            return None
    
    def find_image_path(self, path):
        """
        Find image path, trying multiple locations.
        
        Args:
            path: Original path
            
        Returns:
            str: Found path or None
        """
        # Try original path
        if os.path.exists(path):
            return path
        
        # Try with BASE_DIR prefix (Docker container)
        app_path = os.path.join(
            self.config.BASE_DIR, 
            path.lstrip('/')
        )
        if os.path.exists(app_path):
            logger.info(f"Found at Docker container path: {app_path}")
            return app_path
        
        logger.warning(f"Image not found at any path: {path}")
        return None