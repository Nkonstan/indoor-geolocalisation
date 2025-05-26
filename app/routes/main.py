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
setup_logging()
logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)
# Initialize services
image_service = ImageService()


@main_bp.route('/llava-interpret', methods=['POST'])
def llava_interpret():
    try:
        # Get base64 image and metadata from request
        data = request.json
        if not data or 'image_base64' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_base64 = data['image_base64']
        country = data.get('country', 'Unknown')
        prompt_type = data.get('prompt_type', 'general')

        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        # Create temp directory if it doesn't exist
        temp_dir = '/app/static/temp'
        os.makedirs(temp_dir, exist_ok=True)
        # Save temporary image
        temp_image_path = os.path.join('/app/static/temp', f"temp_{int(time.time())}.jpg")
        with open(temp_image_path, 'wb') as f:
            f.write(image_data)

        # Generate appropriate prompt based on the type
        if prompt_type == 'traffic_sign':
            system_context, user_message = PromptGenerator.get_traffic_sign_prompt(country)
            prompt = PromptGenerator.get_analysis_prompt(system_context, user_message)
            # system_context, user_message = PromptGenerator.get_traffic_sign_prompt(country)
        elif prompt_type == 'license_plate':
            system_context, user_message = PromptGenerator.get_license_plate_prompt(country)
            prompt = PromptGenerator.get_analysis_prompt(system_context, user_message)
        else:  # general scene analysis
            system_context, user_message = PromptGenerator.get_outdoor_scene_prompt(country)
            prompt = PromptGenerator.get_analysis_prompt(system_context, user_message)

        # Generate prompt and get LLaVA response
        llava_response = LLaVAHandler.get_response(image_service.model_service, prompt, temp_image_path)

        # Parse response
        parsed_response = ResponseParser.parse_llava_response(llava_response)

        # Clean up temp file
        os.remove(temp_image_path)

        return jsonify({
            'interpretation': parsed_response,
            'country': country,
            'prompt_type': prompt_type
        })
    except Exception as e:
        logger.error(f"Error in llava-interpret: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Basic routes
@main_bp.route('/', methods=['GET'])
def upload_form():
    return render_template('upload.html')

@main_bp.route('/static/<path:filename>')
def serve_static_files(filename):
    return send_from_directory('static', filename)

@main_bp.route('/segmentations/<path:filename>')
def serve_segmentations(filename):
    root_dir = current_app.root_path
    base_dir = os.path.dirname(root_dir)
    segmentations_dir = os.path.join(base_dir, 'segmentations')
    path_parts = filename.split('/')
    file_name = path_parts[-1]
    sub_dir = '/'.join(path_parts[:-1])
    full_dir = os.path.join(segmentations_dir, sub_dir)
    logger.info(f"Serving file: {file_name} from directory: {full_dir}")
    return send_from_directory(full_dir, file_name)


@main_bp.route('/predict/top', methods=['POST'])
def predict_image_top():
    timer = ProcessTimer()
    try:
        with timer.time_process("Top Prediction Request"):
            # Initial setup
            file = request.files['image']
            if not file:
                return jsonify({'error': 'No file provided'}), 400
            # Clear memory before processing
            gc.collect()
            torch.cuda.empty_cache()
            # Process image to get predictions
            result = image_service.process_image(file)
            # Get predictions dictionary
            predictions = result['prediction_message']
            if isinstance(predictions, str):
                predictions = json.loads(predictions)
            # Find the top country prediction
            top_country = None
            top_score = -1
            for country, score in predictions.items():
                # Convert score to float (handling percentage strings)
                score_value = float(str(score).replace('%', ''))
                if score_value > top_score:
                    top_score = score_value
                    top_country = country
            # Clean up
            gc.collect()
            torch.cuda.empty_cache()
            # Return the top prediction AND all predictions
            return jsonify({
                'status': 'success',
                'top_prediction': {
                    'country': top_country,
                    'score': predictions[top_country]
                },
                'all_predictions': predictions,
                'timing': timer.timings.get('Top Prediction Request', 0)
            })
    except Exception as e:
        logger.error(f"Error getting top prediction: {str(e)}")
        gc.collect()
        torch.cuda.empty_cache()
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@main_bp.route('/process', methods=['POST'])
def process_image():
    timer = ProcessTimer()
    try:
        with timer.time_process("Total Request"):
            with timer.time_process("Initial Setup"):
                log_memory_usage("Start of request")
                file = request.files['image']
                if not file:
                    if request.headers.get('Accept') == 'application/json':
                        return jsonify({'error': 'No file provided'}), 400
                    return render_template('message.html', error='No file provided')
                # Clear memory before processing
                gc.collect()
                torch.cuda.empty_cache()

            # Step 1: Geographic models processing
            with timer.time_process("Geographic Processing"):
                with torch.no_grad():
                    result = image_service.process_image(file)
                log_memory_usage("After geographic processing")
                del file
                gc.collect()
                torch.cuda.empty_cache()

            # # In the segmentation processing section, after you've processed the segmentation results
            # with timer.time_process("Segmentation Processing"):
            #     log_memory_usage("Before segmentation")
            #     segmentation_results = image_service.process_automatic_segmentation(result['image_path'])
            #
            #     # Define the reference data path as it exists in the container
            #     reference_data_path = "/app/segmentations"  # Adjust this to the actual mount point in your container
            #
            #     logger.info(f"Using reference data path: {reference_data_path}")
            #     logger.info(f"Path exists: {os.path.exists(reference_data_path)}")
            #     # Call the helper function to encode all images to base64
            #     attention_map_base64, material_mask_base64, segmentation_results = image_service.encode_images_to_base64(
            #         result, segmentation_results, reference_data_path
            # )

            # Define the reference data path as it exists in the container
            reference_data_path = "/app/segmentations"  # Adjust this to the actual mount point in your container

            with timer.time_process("Segmentation Processing"):
                log_memory_usage("Before segmentation")

                # Get predictions
                predictions = result['prediction_message'] if isinstance(result['prediction_message'],
                                                                         dict) else json.loads(
                    result['prediction_message'])

                # Find the top country efficiently
                top_country = None
                top_score = -1

                for country, score in predictions.items():
                    score_value = float(str(score).replace('%', ''))
                    if score_value > top_score:
                        top_score = score_value
                        top_country = country

                # Skip segmentation if top country is Portugal
                if top_country == 'Portugal':
                    logger.info(
                        f"Portugal detected as top country with {predictions['Portugal']}% - skipping segmentation")
                    segmentation_results = {}
                else:
                    segmentation_results = image_service.process_automatic_segmentation(result['image_path'])

                logger.info(f"Using reference data path: {reference_data_path}")
                logger.info(f"Path exists: {os.path.exists(reference_data_path)}")

                # Call the helper function to encode all images to base64
                attention_map_base64, material_mask_base64, segmentation_results = image_service.encode_images_to_base64(
                    result, segmentation_results, reference_data_path
                )

            with timer.time_process("Result Formatting"):
                # NEW: Process segmentation results
                segmentation_analysis = []
                for element_type, element_data in segmentation_results.items():
                    if element_data.get('similar_segments_info'):
                        # Get unique countries from similar segments
                        similar_countries = [
                            info['country']
                            for info in element_data['similar_segments_info']
                        ]
                        confidence = element_data.get('confidence', 0) * 100
                        element_text = (
                            f"{element_type.title()} - detected with {confidence:.1f}% confidence. "
                            f"Most similar {element_type.title()} styles found in: {', '.join(similar_countries)}"
                        )
                        segmentation_analysis.append(element_text)
                segmentation_text = "\n      ".join(
                    segmentation_analysis) if segmentation_analysis else "No architectural elements detected"
                # Format dynamic continents
                continent_text = ", ".join([
                    f"{continent}: {percentage:.1f}%"
                    for continent, percentage in result['continent_percentages'].items()
                    if percentage > 0
                ])

                predictions = (json.loads(result['prediction_message'])
                               if isinstance(result['prediction_message'], str)
                               else result['prediction_message'])
                top_country = max(predictions.items(), key=lambda x: float(str(x[1]).replace('%', '')))
                country_description = image_service.model_service.config.COUNTRY_DESCRIPTIONS.get(top_country[0], "")
                predictions_text = ", ".join([
                    f"{country} ({score}%)"
                    for country, score in predictions.items()
                ])
                materials_list = result['material_data']['image_materials'].split('%')
                materials_text = ", ".join([mat.strip() + '%' for mat in materials_list if mat.strip()])

                # Add material country matches
                material_country_text = ""
                if 'material_country_matches' in result['material_data']:
                    matches = result['material_data']['material_country_matches']
                    material_country_text = "\n\n   MATERIAL COUNTRY MATCHES:"
                    for material, match_info in matches.items():
                        material_country_text += f"\n   {material}: {match_info['country']} ({match_info['country_value']}%)"

                system_context = f"""Image Analysis Results:
                1. Location Analysis:
                   CONTINENTAL DISTRIBUTION:
                   {continent_text}

                   COUNTRY PREDICTIONS:
                   {predictions_text}

                2. Typical Interior Characteristics of {top_country[0]} ({top_country[1]}%):
                {country_description}

                3. Material Analysis:
                {materials_text}

                4. Architectural Elements Analysis:
                  {segmentation_text}"""
                session['system_context'] = system_context
            # Clear memory before LLaVA
            gc.collect()
            torch.cuda.empty_cache()
            # Step 2: LLaVA processing with memory management
            with timer.time_process("LLaVA Processing"):
                try:
                    # image_service.model_service._unload_current_model()
                    gc.collect()
                    torch.cuda.empty_cache()
                    # initial_prompt = PromptGenerator.get_analysis_prompt(system_context,
                    #                                                      "I want your detailed geo-localization analysis based on all the above. Clearly observable features that support the AI location analysis.")
                    with torch.no_grad():
                        initial_prompt = PromptGenerator.get_analysis_prompt(system_context,
                                                                         "I want your detailed geo-localization analysis based on all the above AND your final prediction of which country this is. Start with clearly observable features that support the AI location analysis, and end with a specific country prediction.")
                        args = LLaVAHandler.create_args(image_service.model_service, initial_prompt, result['image_path'])
                        log_memory_usage("Before LLaVA processing")
                        initial_llava_response = image_service.model_service.invoke_llava_model(args)
                                    # CRITICAL: Clean up args immediately
                        del args, initial_prompt
                        torch.cuda.empty_cache()
                        gc.collect()
                    log_memory_usage("After LLaVA processing")
                    parsed_response = ResponseParser.parse_llava_response(initial_llava_response)

                except Exception as e:
                    logger.error(f"Error in LLaVA processing: {str(e)}")
                    parsed_response = {
                        "response": "Error processing LLaVA analysis",
                        "confidence_level": "Low",
                        "uncertainty": str(e)
                    }
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()
            initial_chat = [
                {"role": "system", "content": system_context},
                {"role": "assistant", "content": parsed_response}
            ]
            log_memory_usage("End of request")
            # Print timing summary
            timer.print_summary()
            # logger.debug(f"Returning Prediction Message: {json.dumps(predictions, indent=2)}")
            # logger.debug(f"Returning attention_map_path: {json.dumps(result['attention_map_path'], indent=2)}")
            #
            # logger.debug(f"Returning Continent Percentages: {json.dumps(result['continent_percentages'], indent=2)}")
            # logger.debug(f"Returning Material Data: {json.dumps(result['material_data'], indent=2)}")
            # logger.debug(f"Returning LLaVA Analysis: {json.dumps(parsed_response, indent=2)}")
            # logger.debug(f"Returning Segmentation Results: {json.dumps(segmentation_results, indent=2)}")
            logger.debug(f"Returning Timing Summary: {json.dumps(timer.timings, indent=2)}")
            # If API request
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
                    # 'timing_summary': timer.timings
                })
            # For web interface
            return render_template(
                'message.html',
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

        # Print timing summary even on error
        timer.print_summary()

        if request.headers.get('Accept') == 'application/json':
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timing_summary': timer.timings
            }), 500

        return render_template(
            'message.html',
            error=str(e),
            continent_percentages={},
            prediction_message={},
            material_data={},
            initial_chat=[],
            segmentation_results={},
            timing_summary=timer.timings
        )


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

        # Format the response before sending it to frontend
        confidence_emoji = {
            "High": "🟢",
            "Medium": "🟡",
            "Low": "🔴"
        }

        final_response = f"{confidence_emoji[parsed_response['confidence_level']]} {parsed_response['confidence_level']} Confidence\n{parsed_response['response']}"

        # Add uncertainty section if it exists
        if parsed_response['uncertainty']:
            final_response = f"{final_response}\n\nNote: {parsed_response['uncertainty']}"

        return jsonify({
            "reply": {
                "response": final_response,
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