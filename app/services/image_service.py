import logging
import io
from PIL import Image
import os
import json
import torch
from app.services.model_service import ModelService
from app.services.database_service import DatabaseService
from app.services.image_file_handler import ImageFileHandler
from app.services.web_image_handler import WebImageProcessor
from app.services.external_service_client import ExternalServiceClient
from app.services.similarity_search_service import SimilaritySearchService
from app.services.segmentation_orchestrator import SegmentationOrchestrator
from app.utils.gpu_utils import gpu_memory_manager
from app.utils.logging import setup_logging

# Keep SegmentIndexManager in this file for now 
from app.services.segment_index_manager import SegmentIndexManager

setup_logging()
logger = logging.getLogger(__name__)

class ImageService:
    """
    Main orchestrator for all image-related operations.
    Delegates to specialized services for specific tasks.
    """
    
    def __init__(self):
        self.model_service = ModelService()
        
        # Initialize index manager
        self.index_manager = SegmentIndexManager(self.model_service)
        
        # Initialize all specialized services
        self.file_handler = ImageFileHandler(self.model_service.config)
        self.web_image_processor = WebImageProcessor(self.model_service.config)
        self.external_service_client = ExternalServiceClient(self.model_service.config)
        self.similarity_search_service = SimilaritySearchService(self.index_manager)
        self.segmentation_orchestrator = SegmentationOrchestrator(
            self.model_service,
            self.external_service_client,
            self.similarity_search_service
        )
    
    def process_image(self, file):
        """
        Process uploaded image with improved memory management.
        
        Args:
            file: Flask file upload object
            
        Returns:
            dict: Processing results
        """
        try:
            with gpu_memory_manager():
                logger.info("Starting image processing")
                
                # Step 1: Validate and read file
                file_content = self.file_handler.validate_and_read_upload(file)
                
                # Step 2: Open and resize image
                image = self.web_image_processor.open_and_resize(file_content)
                del file_content
                
                # Step 3: Save image
                path = self.file_handler.save_image(image, extension='png')
                del image
                
                # Step 4: Geographic predictions
                with torch.no_grad():
                    prediction_message, attention_map_path = (
                        self.model_service.base_predictor(
                            path,
                            None,
                            None,
                            self.model_service.device
                        )
                    )
                
                # Step 5: Process predictions
                prediction_dict = json.loads(prediction_message)
                prediction_dict = {k: f'{v}' for k, v in prediction_dict.items()}
                
                # Step 6: Calculate continent percentages
                continent_percentages = self._calculate_continent_percentages(
                    prediction_dict,
                    self.model_service.config.CONTINENTS_DICT
                )
                
                # Step 7: Get material recognition data
                material_data = self.external_service_client.get_material_recognition(path)
                
                logger.info(f"Continent percentages: {continent_percentages}")
                logger.info(f"Predictions: {prediction_dict}")
                logger.info(f"Material recognition: {material_data}")
                
                # Prepare result
                result = {
                    'message': "Processed the image successfully",
                    'image_path': path,
                    'attention_map_path': attention_map_path,
                    'prediction_message': prediction_dict,
                    'continent_percentages': continent_percentages,
                    'material_data': material_data
                }
                
                return result
                
        except json.JSONDecodeError as e:
            logger.error(f"Error processing prediction result: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise
        finally:
            # Single final cleanup
            torch.cuda.empty_cache()
    
    def process_automatic_segmentation(self, image_path):
        """
        Process automatic segmentation using Lang-SAM service.
        Delegates to SegmentationOrchestrator.
        
        Args:
            image_path: Path to input image
            
        Returns:
            dict: Segmentation results
        """
        return self.segmentation_orchestrator.process_automatic_segmentation(image_path)
    
    def encode_images_to_base64(self, result, segmentation_results, reference_data_path):
        """
        Encode all image artifacts to base64 format.
        
        Args:
            result: The result dictionary containing paths to images
            segmentation_results: Dictionary containing segmentation data
            reference_data_path: Path to reference segmentation images
            
        Returns:
            tuple: (attention_map_base64, material_mask_base64, segmentation_results)
        """
        # Encode attention map
        attention_map_base64 = self.file_handler.encode_image_to_base64(
            result.get('attention_map_path')
        )
        
        # Encode material mask
        material_mask_base64 = None
        if result['material_data'].get('mask_image_path'):
            mask_path = result['material_data']['mask_image_path']
            logger.info(f"Material data keys: {list(result['material_data'].keys())}")
            logger.info(f"Material mask path: {mask_path}")
            
            # Find the actual path
            found_path = self.file_handler.find_image_path(mask_path)
            if found_path:
                material_mask_base64 = self.file_handler.encode_image_to_base64(found_path)
            else:
                logger.warning(f"Material mask file not found")
        
        # Encode segmentation images
        for element_type, element_data in segmentation_results.items():
            # Encode cropped segment image
            if element_data.get('cropped_image_path'):
                element_data['cropped_image_base64'] = (
                    self.file_handler.encode_image_to_base64(
                        element_data['cropped_image_path']
                    )
                )
            
            # Process similar segments
            if element_data.get('similar_segments_info'):
                for idx, info in enumerate(element_data['similar_segments_info']):
                    ref_image_path = os.path.join(reference_data_path, info['path'])
                    
                    encoded = self.file_handler.encode_image_to_base64(ref_image_path)
                    if encoded:
                        element_data['similar_segments_info'][idx]['image_base64'] = encoded
        
        return attention_map_base64, material_mask_base64, segmentation_results
    
    # ========== Backward Compatibility Methods ==========
    
    def _calculate_continent_percentages(self, prediction_dict, continents_dict):
        """Calculate percentage totals for each continent."""
        continents_percentages = {}
        for continent, countries in continents_dict.items():
            continents_percentages[continent] = sum(
                float(prediction_dict[country])
                for country in countries
                if country in prediction_dict
            )
        return continents_percentages
    
    def _get_material_recognition(self, image_path):
        """
        Backward compatibility: Get material recognition data.
        Delegates to ExternalServiceClient.
        """
        return self.external_service_client.get_material_recognition(image_path)
    
    def _load_image(self, image_path):
        """
        Backward compatibility: Load and convert image to RGB.
        Delegates to WebImageProcessor.
        """
        return self.web_image_processor.load_image_bgr_to_rgb(image_path)
    
    def _crop_to_non_black_region(self, segmented_image):
        """
        Backward compatibility: Crop image to non-black region.
        Delegates to WebImageProcessor.
        """
        return self.web_image_processor.crop_to_non_black_region(segmented_image)
    
    def get_similar_segments_merged(self, target, query_vec_dhn):
        """
        Backward compatibility: Get similar segments using Faiss.
        Delegates to SimilaritySearchService.
        """
        return self.similarity_search_service.find_similar_segments(
            target, 
            query_vec_dhn
        )
    
    def _get_similar_segments(self, query_vec_dhn, _, target):
        """
        Backward compatibility: Get similar segments for a specific target.
        Delegates to SimilaritySearchService.
        """
        return self.similarity_search_service.find_similar_segments(
            target,
            query_vec_dhn
        )