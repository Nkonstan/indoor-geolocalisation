import os
import logging
import uuid
import torch
import gc
import numpy as np
import cv2
from PIL import Image
from lang_sam import LangSAM
from transformers import AutoProcessor, AutoConfig, AutoModelForZeroShotObjectDetection

# Set environment variable for your local grounding-dino directory.
os.environ["GROUNDING_DINO_DIR"] = "/app/grounding-dino-base"

# Monkey-patch AutoProcessor.from_pretrained
_orig_processor_from_pretrained = AutoProcessor.from_pretrained
def custom_processor_from_pretrained(identifier, *args, **kwargs):
    if identifier == "IDEA-Research/grounding-dino-base":
        identifier = os.getenv("GROUNDING_DINO_DIR", identifier)
        kwargs["local_files_only"] = True
    return _orig_processor_from_pretrained(identifier, *args, **kwargs)
AutoProcessor.from_pretrained = custom_processor_from_pretrained

# Monkey-patch AutoConfig.from_pretrained
_orig_config_from_pretrained = AutoConfig.from_pretrained
def custom_config_from_pretrained(identifier, *args, **kwargs):
    if identifier == "IDEA-Research/grounding-dino-base":
        identifier = os.getenv("GROUNDING_DINO_DIR", identifier)
        kwargs["local_files_only"] = True
    return _orig_config_from_pretrained(identifier, *args, **kwargs)
AutoConfig.from_pretrained = custom_config_from_pretrained

# Monkey-patch AutoModelForZeroShotObjectDetection.from_pretrained
_orig_model_from_pretrained = AutoModelForZeroShotObjectDetection.from_pretrained
def custom_model_from_pretrained(identifier, *args, **kwargs):
    if identifier == "IDEA-Research/grounding-dino-base":
        identifier = os.getenv("GROUNDING_DINO_DIR", identifier)
        kwargs["local_files_only"] = True
    return _orig_model_from_pretrained(identifier, *args, **kwargs)
AutoModelForZeroShotObjectDetection.from_pretrained = custom_model_from_pretrained

logger = logging.getLogger(__name__)


def cleanup_gpu_memory():
    """Helper function to cleanup GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

def get_model():
    global model
    if model is None:
        model = LangSAM()
        # Move model to CPU after initialization
        if hasattr(model, 'model'):
            model.model.cpu()
    return model

# Initialize model
model = None
# Get model instance
current_model = get_model()


def get_largest_connected_component(segmented_image):
    """
    Find largest connected component and apply smart cropping based on black area percentage
    with optimized memory management
    """
    try:
        # Convert to grayscale to find non-black areas
        gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
        
        # Create binary mask: non-black pixels = 255, black pixels = 0
        binary_mask = (gray > 10).astype(np.uint8) * 255
        
        # Clean up gray immediately
        del gray
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        # Clean up binary_mask immediately
        del binary_mask
        
        # Skip background (label 0), find largest component among the rest
        if num_labels <= 1:  # Only background found
            del labels, stats, centroids
            return None
        
        # Get areas of all components (excluding background at index 0)
        areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
        largest_component_idx = np.argmax(areas) + 1  # +1 because we skipped background
        
        # Clean up areas immediately
        del areas
        
        # Create mask for largest component only
        largest_mask = (labels == largest_component_idx).astype(np.uint8)
        
        # Clean up labels immediately
        del labels, stats, centroids
        
        # Apply mask to original segmented image
        largest_mask_3d = np.stack([largest_mask] * 3, axis=-1)
        result = np.where(largest_mask_3d, segmented_image, 0)
        
        # Clean up masks immediately
        del largest_mask, largest_mask_3d
        
        # **CROPPING LOGIC**
        # First, get the basic bounding box
        non_zero = np.any(result > 10, axis=2)
        rows = np.any(non_zero, axis=1)
        cols = np.any(non_zero, axis=0)
        
        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            basic_crop = result[rmin:rmax + 1, cmin:cmax + 1]
            
            # Clean up intermediate arrays
            del non_zero, rows, cols
            
            # Calculate black area percentage in the basic crop
            total_pixels = basic_crop.shape[0] * basic_crop.shape[1]
            colored_pixels = np.sum(np.any(basic_crop > 10, axis=2))
            black_percentage = (total_pixels - colored_pixels) / total_pixels
            
            # If black area > 30%, apply aggressive cropping
            if black_percentage > 0.30:
                final_result = apply_aggressive_cropping(basic_crop)
                del basic_crop
                return final_result
            else:
                del result  # Clean up original result
                return basic_crop
        
        # Clean up if no valid crop found
        del non_zero, rows, cols
        return result
        
    except Exception as e:
        logger.error(f"Error in connected components analysis: {str(e)}")
        return None
    finally:
        gc.collect()

def apply_aggressive_cropping(image):
    """
    Apply aggressive cropping to remove internal black areas with memory optimization
    """
    try:
        # Find all colored pixels
        colored_mask = np.any(image > 10, axis=2)
        
        if not np.any(colored_mask):
            del colored_mask
            return image
        
        # Get coordinates of all colored pixels
        y_coords, x_coords = np.where(colored_mask)
        
        # Clean up mask immediately
        del colored_mask
        
        # Find the densest region using percentiles to ignore outliers
        y_min = np.percentile(y_coords, 5)   # Ignore bottom 5% of points
        y_max = np.percentile(y_coords, 95)  # Ignore top 5% of points
        x_min = np.percentile(x_coords, 5)   # Ignore leftmost 5% of points
        x_max = np.percentile(x_coords, 95)  # Ignore rightmost 5% of points
        
        # Clean up coordinate arrays immediately
        del y_coords, x_coords
        
        # Convert to integers and ensure valid bounds
        y_min = max(0, int(y_min))
        y_max = min(image.shape[0] - 1, int(y_max))
        x_min = max(0, int(x_min))
        x_max = min(image.shape[1] - 1, int(x_max))
        
        # Crop to the dense region
        aggressive_crop = image[y_min:y_max + 1, x_min:x_max + 1]
        
        return aggressive_crop
        
    except Exception as e:
        logger.error(f"Error in aggressive cropping: {str(e)}")
        return image
    finally:
        gc.collect()

# Enhanced cleanup function
def enhanced_cleanup():
    """Enhanced cleanup function for memory management"""    
    # Force Python garbage collection multiple times
    for _ in range(1):
        gc.collect()


def process_segmentation(image_path, targets):
    """
    Main segmentation function with logic 100% identical to the first script.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    image = Image.open(image_path).convert('RGB')
    results = {}
    
    # Move model to GPU for predictions
    model = get_model()
    if hasattr(model, 'model'):
        model.model.cuda()

    # Different confidence thresholds for different elements
    CONFIDENCE_THRESHOLDS = {
        'door': 0.44,  # 50%
        'window': 0.44,  # 50%
        'ceiling': 0.44,  # 50%
        'floor': 0.44  # 50%
    }

    try:
        for target in targets:
            try:
                prompt = f"a {target} in the interior image"
                prediction = model.predict(
                    [image],
                    [prompt],
                    box_threshold=0.3,  # Keep base threshold low to get all potential detections
                    text_threshold=0.25
                )

                if prediction and len(prediction) > 0:
                    pred = prediction[0]
                    box_scores = pred.get('scores', np.array([]))
                    masks = pred.get('masks', [])

                    if not isinstance(box_scores, np.ndarray):
                        box_scores = np.array(box_scores)

                    logger.info(f"{target} detection scores: {box_scores}")

                    if len(masks) > 0 and len(box_scores) > 0:
                        # Handle both single and multiple detections
                        if box_scores.ndim == 0:
                            best_score = float(box_scores)
                            best_idx = 0
                        else:
                            best_idx = box_scores.argmax()
                            best_score = float(box_scores[best_idx])

                        logger.info(f"{target} best score: {best_score}")

                        # Apply confidence threshold based on element type
                        required_confidence = CONFIDENCE_THRESHOLDS.get(target, 0.5)
                        if best_score >= required_confidence:
                            best_mask = masks[best_idx]

                            # Create segmented image
                            image_array = np.array(image)
                            mask_3d = np.stack([best_mask] * 3, axis=-1)
                            segmented = np.where(mask_3d, image_array, 0)

                            # SPECIAL PROCESSING FOR FLOOR ONLY
                            if target == 'floor':
                                # Get largest connected component with smart cropping
                                processed_segmented = get_largest_connected_component(segmented)
                                if processed_segmented is None:
                                    results[target] = {
                                        'found': False,
                                        'reason': 'No connected components found in floor segmentation'
                                    }
                                    # Clean up before continuing
                                    del image_array, mask_3d, segmented, best_mask
                                    enhanced_cleanup()
                                    continue  # This exits the current loop iteration and moves to next target
                                
                                # Replace segmented with processed version and clean up old one
                                del segmented
                                segmented = processed_segmented
                                del processed_segmented
                                
                                # Skip normal cropping since get_largest_connected_component handles it
                                filename = f"{target}_{uuid.uuid4()}.png"
                                output_path = os.path.join('static', filename)
                                
                                # Use copy to ensure OpenCV doesn't hold references
                                segmented_bgr = cv2.cvtColor(segmented.copy(), cv2.COLOR_RGB2BGR)
                                cv2.imwrite(output_path, segmented_bgr)
                                
                                # Clean up immediately
                                del segmented_bgr, segmented, image_array, mask_3d, best_mask
                                
                                results[target] = {
                                    'found': True,
                                    'image_path': output_path,
                                    'confidence': best_score
                                }
                                
                                # Force cleanup after floor processing
                                enhanced_cleanup()
                                
                                # Continue to next target (no return needed, just continue the loop)
                                continue

                            # Crop to content
                            non_zero = np.any(segmented > 0, axis=2)
                            rows = np.any(non_zero, axis=1)
                            cols = np.any(non_zero, axis=0)

                            if np.any(rows) and np.any(cols):
                                rmin, rmax = np.where(rows)[0][[0, -1]]
                                cmin, cmax = np.where(cols)[0][[0, -1]]
                                cropped = segmented[rmin:rmax + 1, cmin:cmax + 1]

                                filename = f"{target}_{uuid.uuid4()}.png"
                                output_path = os.path.join('static', filename)
                                cv2.imwrite(output_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

                                results[target] = {
                                    'found': True,
                                    'image_path': output_path,
                                    'confidence': best_score
                                }
                            else:
                                results[target] = {
                                    'found': False,
                                    'reason': 'No valid segmentation after cropping',
                                    'scores': {'box': best_score}
                                }
                        else:
                            results[target] = {
                                'found': False,
                                'reason': f'Low confidence ({best_score:.3f} < required {required_confidence})',
                                'scores': {'box': best_score}
                            }
                    else:
                        results[target] = {
                            'found': False,
                            'reason': 'No valid masks or scores found'
                        }
                else:
                    results[target] = {
                        'found': False,
                        'reason': 'No prediction results'
                    }

            except Exception as e:
                logger.error(f"Error processing {target}: {str(e)}")
                results[target] = {
                    'found': False,
                    'error': str(e),
                    'reason': 'Processing error'
                }
            finally:
                cleanup_gpu_memory()

    finally:
        # Move model back to CPU after all predictions
        if hasattr(model, 'model'):
            model.model.cpu()
        cleanup_gpu_memory()

    return results