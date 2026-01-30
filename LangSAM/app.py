from flask import Flask, request, jsonify
import logging
import sys
from langsam_logic import process_segmentation, get_model

# Setup Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

app = Flask(__name__)


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

        # Call the logic function
        results = process_segmentation(image_path, targets)

        return jsonify({
            'status': 'success',
            'results': results
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error in segmentation endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)