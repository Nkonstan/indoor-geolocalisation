from flask import Flask, request, jsonify
import logging
import sys
import os
import numpy as np

from materobot_logic import material_pred

logger = logging.getLogger(__name__)

# Configure root logger to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
app = Flask(__name__)
# Configure the app logger
app.logger.setLevel(logging.INFO)
for handler in app.logger.handlers:
    app.logger.removeHandler(handler)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
app.logger.addHandler(handler)

logger.info("MATERobot service starting up")



@app.route('/material_recognition', methods=['POST'])
def material_recognition():
    try:
        logger.info("Received material recognition request")
        data = request.json
        logger.info(f"Request data: {data}")

        if not data or 'image_path' not in data:
            logger.error("No image_path in request data")
            return jsonify({'error': 'No image_path provided'}), 400

        image_path = data['image_path']
        abs_image_path = os.path.join('/app', image_path)
        logger.info(f"Processing image path: {abs_image_path}")

        if not os.path.exists(abs_image_path):
            logger.error(f"Image path does not exist: {abs_image_path}")
            return jsonify({'error': f"Image path does not exist: {abs_image_path}"}), 400

        logger.info("Calling material_pred function")
        result = material_pred(abs_image_path)
        logger.info(f"material_pred returned: {result}")

        image_materials, mask_image_full_path, material_country_matches = result

        if image_materials is None or mask_image_full_path is None:
            logger.error("Failed to process material recognition - null results returned")
            raise ValueError("Failed to process material recognition.")

        logger.info(f"Material recognition successful: {image_materials}")
        logger.info(f"Mask image path: {mask_image_full_path}")
        logger.info(f"Material country matches: {material_country_matches}")

        if isinstance(image_materials, np.ndarray):
            image_materials = image_materials.tolist()
            logger.info("Converted image_materials from ndarray to list")

        if not os.path.exists(mask_image_full_path):
            logger.error(f"Mask image file does not exist: {mask_image_full_path}")
            return jsonify({'error': f"Mask image file does not exist: {mask_image_full_path}"}), 500

        mask_image_url = f"/static/{os.path.basename(mask_image_full_path)}"
        logger.info(f"Final mask image URL: {mask_image_url}")

        response_data = {
            'image_materials': image_materials,
            'mask_image_path': mask_image_url,
            'material_country_matches': material_country_matches
        }
        logger.info(f"Returning response: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Exception in material_recognition: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # CRITICAL: Set threaded=False to prevent Flask threading memory leak
    app.run(host='0.0.0.0', port=5001, threaded=False)
