import logging
import requests

logger = logging.getLogger(__name__)

class ExternalServiceClient:
    """Handles communication with external services."""
    
    def __init__(self, config):
        self.config = config
        self.material_recognition_url = config.MATERIAL_RECOGNITION_URL
        self.langsam_url = config.LANGSAM_URL
    
    def get_material_recognition(self, image_path):
        """
        Get material recognition data from material service.
        
        Args:
            image_path: Path to image
            
        Returns:
            dict: Material recognition results
        """
        try:
            response = requests.post(
                self.material_recognition_url,
                json={'image_path': image_path},
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Material recognition successful for {image_path}")
            return result
            
        except requests.RequestException as e:
            logger.error(f"Material recognition request failed: {str(e)}")
            return {'error': 'Failed to get material recognition data'}
        except Exception as e:
            logger.error(f"Material recognition error: {str(e)}")
            return {'error': str(e)}
    
    def get_langsam_segmentation(self, image_path, targets):
        """
        Call Lang-SAM service for automatic segmentation.
        
        Args:
            image_path: Path to image
            targets: List of targets to segment (e.g., ['floor', 'ceiling'])
            
        Returns:
            dict: Segmentation results
        """
        try:
            response = requests.post(
                self.langsam_url,
                json={
                    'image_path': image_path,
                    'targets': targets
                },
                timeout=60
            )
            response.raise_for_status()
            
            results = response.json()
            
            if results.get('status') != 'success':
                error_msg = results.get('error', 'Unknown error')
                logger.error(f"Lang-SAM service error: {error_msg}")
                return {'status': 'error', 'error': error_msg}
            
            logger.info(f"Lang-SAM segmentation successful for {image_path}")
            return results
            
        except requests.RequestException as e:
            logger.error(f"Lang-SAM request failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
        except Exception as e:
            logger.error(f"Lang-SAM error: {str(e)}")
            return {'status': 'error', 'error': str(e)}