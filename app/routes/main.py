import os
from flask import Blueprint, request, render_template, jsonify, url_for, send_from_directory, session, current_app
from app.services.image_service import ImageService
import logging
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import base64
import json
import torch
import time
from llava.mm_utils import get_model_name_from_path
import gc
import sys
from app.utils.timing_utils import ProcessTimer
from app.utils.logging import setup_logging
from app.utils.llava_utils import log_memory_usage, LLaVAHandler, ResponseParser, PromptGenerator
from app.utils.process_helper import ProcessHelper
from app.utils.gpu_utils import cleanup_gpu_memory
from app.config import Config
setup_logging()
logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)
# Initialize services
image_service = ImageService()


# Basic routes
@main_bp.route('/', methods=['GET'])
def upload_form():
    return render_template('index.html')

@main_bp.route('/static/<path:filename>')
def serve_static_files(filename):
    return send_from_directory('static', filename)


@main_bp.route('/segmentations/<path:filename>')
def serve_segmentations(filename):
    """
    Serve segmentation reference images.
    Used by the HTML to display 'Similar Segments' images.
    """
    return send_from_directory(Config.SEGMENTATION_REFERENCE_PATH, filename)

@main_bp.route('/predict/top', methods=['POST'])
def predict_image_top():
    timer = ProcessTimer()
    try:
        with timer.time_process("Top Prediction Request"):
            file = request.files['image']
            if not file:
                return jsonify({'error': 'No file provided'}), 400
            
            result = image_service.process_image(file)
            
            # --- Use Helper ---
            predictions, (top_country, top_score) = ProcessHelper.parse_predictions(result['prediction_message'])
            
            return jsonify({
                'status': 'success',
                'top_prediction': {
                    'country': top_country,
                    'score': predictions[top_country] # Keep original format string
                },
                'all_predictions': predictions,
                'timing': timer.timings.get('Top Prediction Request', 0)
            })
    except Exception as e:
        logger.error(f"Error getting top prediction: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500
    finally:
        cleanup_gpu_memory()

@main_bp.route('/process', methods=['POST'])
def process_image():
    timer = ProcessTimer()
    try:
        # [Step 0] Initial Setup
        with timer.time_process("Initial Setup"):
            file = request.files['image']
            if not file:
                if request.headers.get('Accept') == 'application/json':
                    return jsonify({'error': 'No file provided'}), 400
                return render_template('index.html', error='No file provided')

        # [Step 1] Geographic Processing
        with timer.time_process("Geographic Processing"):
            with torch.no_grad():
                result = image_service.process_image(file)
            del file

        # [Step 2] Segmentation & Data Formatting
        reference_data_path = Config.SEGMENTATION_REFERENCE_PATH
        
        with timer.time_process("Segmentation Processing"):
            # A. Parse Predictions using Helper
            predictions, (top_country, top_score) = ProcessHelper.parse_predictions(result['prediction_message'])
            # B. Segmentation Logic 
            segmentation_results = image_service.process_automatic_segmentation(result['image_path'])
            # C. Encode Images (This modifies state, so we keep it here or move to image_service)
            attention_map_base64, material_mask_base64, segmentation_results = image_service.encode_images_to_base64(
                result, segmentation_results, reference_data_path
            )
            # D. Format Text & Context using Helper
            segmentation_text = ProcessHelper.format_segmentation_text(segmentation_results)
            country_description = image_service.model_service.config.COUNTRY_DESCRIPTIONS.get(top_country, "")
            
            system_context = ProcessHelper.build_system_context(
                result, predictions, (top_country, top_score), segmentation_text, country_description
            )
            session['system_context'] = system_context
            logger.info(f"system_context: ({system_context})")

        # [Step 3] LLaVA Processing
        with timer.time_process("LLaVA Processing"):
            try:
                with torch.no_grad():
                    initial_prompt = PromptGenerator.get_analysis_prompt(
                        system_context, Config.LLAVA_ANALYSIS_PROMPT
                    )
                    args = LLaVAHandler.create_args(image_service.model_service, initial_prompt, result['image_path'])
                    initial_llava_response = image_service.model_service.invoke_llava_model(args)
                    del args, initial_prompt
                parsed_response = ResponseParser.parse_llava_response(initial_llava_response)
            except Exception as e:
                logger.error(f"Error in LLaVA processing: {str(e)}")
                parsed_response = {"response": "Error processing LLaVA", "confidence_level": "Low", "uncertainty": str(e)}

        # [Step 4] Response
        timer.print_summary()
        
        if request.headers.get('Accept') == 'application/json':
            return jsonify({
                'status': 'success',
                'image_path': result['image_path'],
                'attention_map_path': result['attention_map_path'],
                'attention_map_base64': attention_map_base64,
                'prediction_message': predictions,
                'continent_percentages': result['continent_percentages'],
                'material_data': result['material_data'],
                'material_mask_base64': material_mask_base64,
                'system_context': system_context,
                'llava_analysis': parsed_response,
                'segmentation_results': segmentation_results,
            })

        # Render Template (Web Interface)
        initial_chat = [
            {"role": "system", "content": system_context},
            {"role": "assistant", "content": parsed_response}
        ]
        
        return render_template(
            'index.html',
            message=result['message'],
            image_path=result['image_path'],
            attention_map_path=result['attention_map_path'],
            prediction_message=result['prediction_message'],
            continent_percentages=result['continent_percentages'],
            material_data=result['material_data'],
            initial_chat=initial_chat,
            segmentation_results=segmentation_results,
            timing_summary=timer.timings
        )

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        gc.collect()
        torch.cuda.empty_cache()
        timer.print_summary()
        
        if request.headers.get('Accept') == 'application/json':
            return jsonify({'status': 'error', 'error': str(e), 'timing_summary': timer.timings}), 500

        return render_template('index.html', error=str(e), timing_summary=timer.timings)
    finally:
        cleanup_gpu_memory()

@main_bp.route('/send_message', methods=['POST'])
def send_message():
    try:
        data = request.json
        user_message = data['message']
        image_path = data.get('image_path') or session.get('last_segmented_image_file')

        if not image_path:
            return jsonify({
                "reply": {
                    "response": "No image available for analysis.",
                    "confidence_level": "Low",
                    "uncertainty": "Missing image context"
                }
            })

        logger.info(f"Using {'provided' if data.get('image_path') else 'session'} image path: {image_path}")

        system_context = session.get('system_context', '')
        start_time = time.time()

        prompt = PromptGenerator.get_analysis_prompt(system_context, user_message)
        llava_response = LLaVAHandler.get_response(image_service.model_service, prompt, image_path)

        logger.info(f"Time taken for LLaVA response: {time.time() - start_time:.2f} seconds")

        # Get parsed response
        parsed_response = ResponseParser.parse_llava_response(llava_response)

        # We DO NOT format the string here. We send raw data.
        return jsonify({
            "reply": {
                "response": parsed_response['response'], # Raw text only
                "confidence_level": parsed_response['confidence_level'],
                "uncertainty": parsed_response['uncertainty'],
                "status": "success"
            }
        })

    except Exception as e:
        logger.error(f"Error in send_message: {str(e)}")
        return jsonify({
            "reply": {
                "response": "Sorry, there was an error processing your request.",
                "confidence_level": "Low",
                "uncertainty": f"Error occurred: {str(e)}",
                "status": "error"
            }
        })
    finally:
        cleanup_gpu_memory()