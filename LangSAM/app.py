from flask import Flask, request, jsonify
from lang_sam import LangSAM
from PIL import Image
import numpy as np
import cv2
import os
import logging
import uuid
import torch
import gc
import sys
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

app = Flask(__name__)


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



@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route('/segment', methods=['POST'])
def segment_image():
    try:
        data = request.json
        image_path = data.get('image_path')
        targets = data.get('targets', ['floor', 'ceiling', 'door', 'window'])

        if not image_path:
            return jsonify({"error": "No image path provided"}), 400

        if not os.path.exists(image_path):
            return jsonify({"error": f"Image not found at {image_path}"}), 404

        image = Image.open(image_path).convert('RGB')
        results = {}


        # Move model to GPU for predictions
        if hasattr(current_model, 'model'):
            current_model.model.cuda()

        # Different confidence thresholds for different elements
        CONFIDENCE_THRESHOLDS = {
            'door': 0.44,  # 50%
            'window': 0.44,  # 50%
            'ceiling': 0.44,  # 50%
            'floor': 0.44  # 50%
        }

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

        # Move model back to CPU after all predictions
        if hasattr(current_model, 'model'):
            current_model.model.cpu()

        return jsonify({
            'status': 'success',
            'results': results
        })

    except Exception as e:
        logger.error(f"Error in segmentation: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Ensure model is on CPU and memory is cleaned
        if model is not None and hasattr(model, 'model'):
            model.model.cpu()
        cleanup_gpu_memory()




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)