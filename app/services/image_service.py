from PIL import Image
import numpy as np
import cv2
import logging
import io
import uuid
import os
import torch
import time
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import json
import requests
import pandas as pd
import gc
import sys
import itertools
from app.services.model_service import ModelService
from app.services.database_service import DatabaseService
from app.utils.gpu_utils import gpu_memory_manager  # Add this import
from app.utils.logging import setup_logging
from flask import current_app

setup_logging()
logger = logging.getLogger(__name__)

import faiss
import threading
import pickle
import base64

class SegmentIndexManager:
    """Manages Faiss indexes for fast similarity search"""

    def __init__(self, model_service):
        self.model_service = model_service
        self.indexes = {}  # Dictionary of {segment_type: faiss_index}
        self.metadata = {}  # Dictionary of {segment_type: list_of_metadata}
        self.index_ready = {}  # Track which indexes are ready
        self.index_lock = threading.RLock()  # Thread safety
        # self.index_dir = "/app/faiss_indexes"  # Where to save/load indexes
        self.index_dir = self.model_service.config.FAISS_INDEX_DIR   # Where to save/load indexes
        os.makedirs(self.index_dir, exist_ok=True)

    def get_or_build_index(self, segment_type):
        """Get an existing index or build it if needed"""
        with self.index_lock:
            # Check if index exists and is ready
            if segment_type in self.indexes and self.index_ready.get(segment_type, False):
                return self.indexes[segment_type], self.metadata[segment_type]

            # Try to load from disk first
            index_path = os.path.join(self.index_dir, f"{segment_type}_index.bin")
            meta_path = os.path.join(self.index_dir, f"{segment_type}_metadata.pkl")

            if os.path.exists(index_path) and os.path.exists(meta_path):
                try:
                    # Load index and metadata
                    self.indexes[segment_type] = faiss.read_index(index_path)
                    with open(meta_path, 'rb') as f:
                        self.metadata[segment_type] = pickle.load(f)
                    self.index_ready[segment_type] = True
                    return self.indexes[segment_type], self.metadata[segment_type]
                except Exception as e:
                    logger.error(f"Error loading index for {segment_type}: {e}")

            # Need to build the index
            logger.info(f"Building Faiss index for {segment_type}...")
            start_time = time.time()

            # Collect all vectors and metadata
            all_vectors = []
            all_metadata = []

            # Process in batches to avoid memory issues
            for batch in self.model_service.db_service.get_segmentation_data(
                    self.model_service.config.MONGODB_DHN_COLLECTION, segment_type
            ):
                for doc in batch:
                    all_vectors.append(doc['Binary Code'])
                    all_metadata.append({
                        'Image Name': doc['Image Name'],
                        'Segment': doc['Segment'],
                        'Country': doc['Country']
                    })

            # Convert to numpy array
            vectors = np.vstack(all_vectors).astype(np.float32)

            # Create and train index (IndexFlatL2 for exact search)
            dimension = vectors.shape[1]
            index = faiss.IndexFlatL2(dimension)

            # Add vectors to index
            index.add(vectors)

            # Save index and metadata
            faiss.write_index(index, index_path)
            with open(meta_path, 'wb') as f:
                pickle.dump(all_metadata, f)

            # Store in memory
            self.indexes[segment_type] = index
            self.metadata[segment_type] = all_metadata
            self.index_ready[segment_type] = True

            logger.info(
                f"Built index for {segment_type} with {len(all_metadata)} vectors in {time.time() - start_time:.2f}s")

            return index, all_metadata

class ImageService:
    def __init__(self):
        self.model_service = ModelService()
        self.index_manager = SegmentIndexManager(self.model_service)
    def process_image(self, file):
        """Process uploaded image with improved memory management."""
        try:
            # Initial validation
            if not file or not file.content_type.startswith('image/'):
                raise ValueError("No valid image file provided")

            # Single memory management context for all operations
            with gpu_memory_manager():
                logger.info("Starting image processing")
                logger.debug(f"File type: {type(file)}")
                # Process image in chunks to reduce memory spikes
                chunks = []
                chunk_size = 1024 * 1024  # 1MB chunks
                while chunk := file.read(chunk_size):
                    chunks.append(chunk)
                file_content = b''.join(chunks)
                del chunks
                if not file_content:
                    raise ValueError("Empty file content")
                # Save and process image
                file_bytes = io.BytesIO(file_content)
                # Verify and open image
                with Image.open(file_bytes) as image:
                    image.verify()
                file_bytes.seek(0)
                image = Image.open(file_bytes)
                # Resize with memory management
                MAX_SIZE = 1024
                ratio = min(MAX_SIZE / image.width, MAX_SIZE / image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.LANCZOS)
                # Save resized image
                filename = str(uuid.uuid4()) + '.png'
                static_dir = self.model_service.config.STATIC_DIR
                path = os.path.join(static_dir, filename)
                os.makedirs(static_dir, exist_ok=True)
                image.save(path, format='PNG', optimize=True)
                logger.info(f"Image saved to {path}")
                # Clean up image processing resources
                del image, file_bytes, file_content
                # Switch to geo model only once
                self.model_service.switch_model('geo')
                # Geographic predictions phase
                with torch.no_grad():
                    prediction_message, attention_map_path = self.model_service.base_predictor(
                        path,
                        None,
                        None,
                        self.model_service.device
                    )
                # Process predictions
                prediction_dict = json.loads(prediction_message)
                prediction_dict = {k: f'{v}' for k, v in prediction_dict.items()}
                # Calculate metrics
                continent_percentages = self._calculate_continent_percentages(
                    prediction_dict,
                    self.model_service.config.CONTINENTS_DICT
                )
                # Get material recognition data
                material_data = self._get_material_recognition(path)
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
            if 'process_response' in locals():
                logger.error(f"Response content: {process_response.text[:1000]}")
            raise
        finally:
            # Single final cleanup
            self.model_service._unload_current_model()
            torch.cuda.empty_cache()


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
        """Get material recognition data from material service."""
        try:
            response = requests.post(
                self.model_service.config.MATERIAL_RECOGNITION_URL,
                json={'image_path': image_path}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Material recognition failed: {str(e)}")
            return {'error': 'Failed to get material recognition data'}

    def _load_image(self, image_path):
        """Load and convert image to RGB."""
        image_bgr = cv2.imread(image_path)
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _crop_to_non_black_region(self, segmented_image):
        """Crop image to non-black region."""
        non_black_mask = np.any(segmented_image > 0, axis=-1)
        if not np.any(non_black_mask):
            return np.zeros_like(segmented_image)
        rows = np.any(non_black_mask, axis=1)
        cols = np.any(non_black_mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return segmented_image[ymin:ymax + 1, xmin:xmax + 1]


    def process_automatic_segmentation(self, image_path):
        """Process automatic segmentation using Lang-SAM service with DHN-only model operations."""
        # Define variables before try block
        found_segments = {}
        segmentation_results = {}
        feature_vectors = {}

        with gpu_memory_manager():  # Main cleanup context
            try:
                # Call Lang-SAM service
                response = requests.post(
                    self.model_service.config.LANGSAM_URL,
                    json={
                        'image_path': image_path,
                        'targets': ['floor', 'ceiling', 'door', 'window']
                    }
                )
                response.raise_for_status()
                results = response.json()
                if results.get('status') != 'success':
                    logger.error(f"Lang-SAM service error: {results.get('error', 'Unknown error')}")
                    return {}
                # Collect found segments first
                found_segments = {
                    target: data for target, data in results['results'].items()
                    if data.get('found', False)
                }
                if found_segments:
                    # Single model processing context for all segments
                    with gpu_memory_manager():
                        # Process all segments at once
                        for target, data in found_segments.items():
                            # Get features for each segment (only DHN)
                            query_vec_dhn = self.model_service.base_predictor(
                                data['image_path'],
                                None,
                                None,
                                self.model_service.device,
                                not_segment=False,
                            )
                            feature_vectors[target] = query_vec_dhn

                        # Process similar segments for all targets
                        for target, data in found_segments.items():
                            query_vec_dhn = feature_vectors[target]
                            similar_segments = self._get_similar_segments(
                                query_vec_dhn,
                                None,  # No HashNet vector needed
                                target
                            )

                            segmentation_results[target] = {
                                'cropped_image_path': data['image_path'],
                                'similar_segments': similar_segments[:4],
                                'confidence': data['confidence'],
                                'similar_segments_info': [{
                                    'path': segment['Full Path'],
                                    'score': f"{segment['Average_Distance']:.3f}",  # Changed from Average_Distance to Distance
                                    'country': segment['Country']
                                } for segment in similar_segments[:4]]
                            }

                        # Cleanup feature vectors
                        del feature_vectors
                        torch.cuda.empty_cache()

                return segmentation_results

            except requests.RequestException as e:
                logger.error(f"Error calling Lang-SAM service: {str(e)}")
                return {}
            except Exception as e:
                logger.error(f"Error in automatic segmentation: {str(e)}")
                self.model_service._unload_current_model()
                return {}
            finally:
                # Ensure cleanup after processing all targets
                if 'feature_vectors' in locals():
                    del feature_vectors
                self.model_service._unload_current_model()
                torch.cuda.empty_cache()

    def get_similar_segments_merged(self, target, query_vec_dhn):
        """Get similar segments using Faiss for ultra-fast similarity search"""
        try:
            logger.info(f"\n=== Starting Faiss Similarity Search for {target} ===")
            start_time = time.time()

            # Get index and metadata for this segment type
            index, metadata = self.index_manager.get_or_build_index(target)

            # Convert query vector to the right format
            query = np.array([query_vec_dhn]).astype(np.float32)

            # Search for nearest neighbors
            k = 10  # Number of results to return
            distances, indices = index.search(query, k)

            # Get results
            closest_segments = []
            for i, idx in enumerate(indices[0]):  # indices[0] because we only have one query
                if idx < len(metadata):  # Safety check
                    segment_data = metadata[idx]
                    closest_segments.append({
                        'Image Name': segment_data['Image Name'],
                        'Distance': float(distances[0][i]),  # L2 distance
                        'Segment': segment_data['Segment'],
                        'Country': segment_data['Country']
                    })

            # Log timing and results
            search_time = time.time() - start_time
            logger.info(f"Faiss search completed in {search_time:.3f}s")

            # Log total processed
            logger.info(f"Database size - DHN: {len(metadata)}")

            # Log final results in the same format as before
            logger.info("\nFinal Top 5 matches:")
            for segment in closest_segments[:5]:
                logger.info(f"Image: {segment['Image Name']}")
                logger.info(f"  DHN Distance: {segment['Distance']:.3f}")
                logger.info(f"  Country: {segment['Country']}")

            # Format result in the same way as before
            result = []
            for segment in closest_segments:
                full_path = os.path.join(segment['Country'],
                                         segment['Segment'],
                                         segment['Image Name'])
                result.append({
                    'Full Path': full_path,
                    'Average_Distance': float(segment['Distance']),  # Renamed but kept for compatibility
                    'Country': segment['Country']
                })
            return result

        except Exception as e:
            logger.error(f"Error in get_similar_segments_merged: {str(e)}")
            return []
        finally:
            # Clean all intermediate data - kept for compatibility
            if 'closest_segments' in locals():
                del closest_segments
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _get_similar_segments(self, query_vec_dhn, _, target):
        """Get similar segments for a specific target using only DHN features."""
        try:
            similar_segments = self.get_similar_segments_merged(
                target,
                query_vec_dhn
            )
            return similar_segments

        except Exception as e:
            logger.error(f"Error getting similar segments: {str(e)}")
            return []
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def encode_images_to_base64(self, result, segmentation_results, reference_data_path):
        """
        Encode all image artifacts to base64 format.

        Args:
            result (dict): The result dictionary containing paths to images
            segmentation_results (dict): Dictionary containing segmentation data
            reference_data_path (str): Path to reference segmentation images

        Returns:
            tuple: (attention_map_base64, material_mask_base64, segmentation_results)
        """
        # Add base64 encoding for attention map
        attention_map_base64 = None
        if result['attention_map_path'] and os.path.exists(result['attention_map_path']):
            with open(result['attention_map_path'], 'rb') as img_file:
                attention_map_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # 1. Log material data structure to check if mask_image_path exists
        logger.info(f"Material data keys: {list(result['material_data'].keys())}")
        if 'mask_image_path' in result['material_data']:
            logger.info(f"Material mask path: {result['material_data']['mask_image_path']}")
        else:
            logger.info("No mask_image_path in material_data")

        # Add base64 encoding for material mask
        material_mask_base64 = None
        if result['material_data'].get('mask_image_path'):
            mask_path = result['material_data']['mask_image_path']
            logger.info(f"Checking material mask path: {mask_path}")

            # First try the original path
            if os.path.exists(mask_path):
                logger.info(f"Found material mask at original path")
                with open(mask_path, 'rb') as img_file:
                    material_mask_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            else:
                # Try with /app prefix (Docker container)
                app_path = os.path.join(self.model_service.config.BASE_DIR, mask_path.lstrip('/'))
                logger.info(f"Trying Docker container path: {app_path}")

                if os.path.exists(app_path):
                    logger.info(f"Found material mask at Docker container path")
                    with open(app_path, 'rb') as img_file:
                        material_mask_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                else:
                    logger.warning(f"Material mask file not found at either path")

        # Add base64 encoding for segmentation images
        for element_type, element_data in segmentation_results.items():
            # Encode cropped segment image
            if element_data.get('cropped_image_path') and os.path.exists(element_data['cropped_image_path']):
                with open(element_data['cropped_image_path'], 'rb') as img_file:
                    element_data['cropped_image_base64'] = base64.b64encode(img_file.read()).decode('utf-8')

            # Process similar segments (keeping existing structure)
            if element_data.get('similar_segments_info'):
                for idx, info in enumerate(element_data['similar_segments_info']):
                    ref_image_path = os.path.join(reference_data_path, info['path'])
                    if os.path.exists(ref_image_path):
                        with open(ref_image_path, 'rb') as img_file:
                            element_data['similar_segments_info'][idx]['image_base64'] = base64.b64encode(
                                img_file.read()).decode('utf-8')

        return attention_map_base64, material_mask_base64, segmentation_results