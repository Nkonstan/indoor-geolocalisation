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

setup_logging()
logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)
# Initialize services
image_service = ImageService()


def log_memory_usage(location: str):
    """Log detailed GPU memory status"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        logger.info(f"\n=== Memory Status at {location} ===")
        logger.info(f"Allocated: {allocated / 1e9:.2f}GB")
        logger.info(f"Reserved: {reserved / 1e9:.2f}GB")
        logger.info(f"Max Allocated: {max_allocated / 1e9:.2f}GB")
        logger.info(f"Memory Summary:\n{torch.cuda.memory_summary(abbreviated=True)}")
        # Log fragmention info
        if reserved > 0:
            fragmentation = 1.0 - (allocated / reserved)
            logger.info(f"Memory Fragmentation: {fragmentation:.2%}")


class LLaVAHandler:
    @staticmethod
    def create_args(model_service, prompt, image_path):
        return type('Args', (), {
            "model_path": model_service.config.MODEL_PATH,
            "model_base": None,
            "model_name": get_model_name_from_path(model_service.config.MODEL_PATH),
            "query": prompt,
            "conv_mode": None,
            "image_file": image_path,
            "sep": ",",
            "temperature": 0.0,  # Zero temperature for complete determinism
            "top_p": 1.0,  # No nucleus sampling cutoff
            "num_beams": 1,  # Single beam for deterministic path
            "max_new_tokens": 512
        })()

    @staticmethod
    def get_response(model_service, prompt, image_path):
        try:
            # Force garbage collection before LLaVA inference
            # import gc
            gc.collect()
            torch.cuda.empty_cache()
            model_service._unload_current_model()
            args = LLaVAHandler.create_args(model_service, prompt, image_path)
            # Run inference with memory cleanup
            with torch.inference_mode():
                response = model_service.invoke_llava_model(args)
            # Clear memory after inference
            torch.cuda.empty_cache()
            gc.collect()
            return response

        except Exception as e:
            logger.error(f"Error in LLaVA inference: {str(e)}")
            # Ensure cleanup even on error
            torch.cuda.empty_cache()
            gc.collect()
            raise

class ResponseParser:
    @staticmethod
    def parse_llava_response(response):
        try:
            result = {
                "response": "",
                "confidence_level": "Medium",
                "uncertainty": ""  # Changed to empty string default
            }
            # Extract main response
            if "Response:" in response:
                main_response = response.split("Response:", 1)[1]
                main_response = main_response.split("Uncertainty Analysis:", 1)[0].strip()
                result["response"] = main_response
            # Extract confidence level
            if "Confidence Level Assessment:" in response:
                conf_part = response.split("Confidence Level Assessment:", 1)[1].lower()
                if "low" in conf_part:
                    result["confidence_level"] = "Low"
                elif "high" in conf_part:
                    result["confidence_level"] = "High"
            # Extract Final Country Prediction
            if "Final Country Prediction:" in response:
                try:
                    # Split at the marker and get everything after it
                    country_part = response.split("Final Country Prediction:", 1)[1]
                    # Clean up the text
                    country_part = country_part.strip()
                    # Split by newlines and get first non-empty line
                    country_lines = [line.strip() for line in country_part.split('\n')]
                    # Filter out empty lines and get the first non-empty one
                    country_name = next((line for line in country_lines if line), "")
                    result["predicted_country"] = country_name
                    logger.debug(f"Extracted country prediction: '{country_name}'")  # Add debug logging
                except Exception as e:
                    logger.error(f"Error extracting country prediction: {str(e)}")
                    result["predicted_country"] = ""
            # Extract uncertainty analysis
            if "Uncertainty Analysis:" in response:
                uncertainty_section = response.split("Uncertainty Analysis:", 1)[1]
                if "Confidence Level Assessment:" in uncertainty_section:
                    uncertainty_section = uncertainty_section.split("Confidence Level Assessment:", 1)[0]
                alternatives = ""
                limitations = ""
                # Extract alternatives
                if "ALTERNATIVES:" in uncertainty_section:
                    alternatives = uncertainty_section.split("ALTERNATIVES:", 1)[1]
                    if "LIMITATIONS:" in alternatives:
                        alternatives = alternatives.split("LIMITATIONS:", 1)[0]
                    alternatives = alternatives.strip("- \n")
                # Extract limitations
                if "LIMITATIONS:" in uncertainty_section:
                    limitations = uncertainty_section.split("LIMITATIONS:", 1)[1]
                    limitations = limitations.split("\n", 1)[0]
                    limitations = limitations.strip("- \n")
                # Format uncertainty as string
                if alternatives or limitations:
                    uncertainty_parts = []
                    if alternatives:
                        uncertainty_parts.append(f"ALTERNATIVES: {alternatives}")
                    if limitations:
                        uncertainty_parts.append(f"LIMITATIONS: {limitations}")
                    result["uncertainty"] = " | ".join(uncertainty_parts)
            return result
        except Exception as e:
            logger.error(f"Error parsing LLaVA response: {str(e)}")
            logger.error(f"Original response: {response}")
            return {
                "response": response,
                "confidence_level": "Medium",
                "uncertainty": ""
            }


class PromptGenerator:
    @staticmethod
    def get_analysis_prompt(system_context, user_message):
        return f"""AI image analysis:{system_context}

        IMPORTANT: This is an AI-powered image analysis system.
        RESPONSE TIP: When addressing questions related to identifying the location (country, continent, or city) of an image, you MUST begin with the AI's location predictions and align your response to support the provided analysis. If you have additional insights or notes, include them in the Uncertainty Analysis.
        MAIN MISSION: You are a Geo-localization assistant whose sole purpose is to help and answer user questions with the utmost caution.

        IMPORTANT REMINDER: Machine learning models, including this one, can make mistakes and are not infallible. Do NOT declare "High" confidence unless the evidence is overwhelmingly clear and beyond reasonable doubt. Always consider the possibility of error and reflect this in your confidence level and justification.

        Confidence Level Guidelines:
        High: Multiple clear, unambiguous visual evidence points AND strong alignment with predicted locations
        Medium: Some clear evidence but with notable uncertainties OR partial alignment with predictions
        Low: Limited or ambiguous evidence OR significant discrepancy with predictions

        User Question: {user_message}

        You MUST use this EXACT format in your response (including markers, bullet points and ALL sections):

        Response: [Your answer here. If for something that you are asked you don't have enough clues you must indicate it]

        Uncertainty Analysis:

        - ALTERNATIVES: [Other potential interpretations that could lead your evidence into misinterpretation]
        - LIMITATIONS: [Factors that may affect your answer, such as image quality, ambiguous elements, cultural overlaps etc]

        Confidence Level Assessment: [ONLY USE 'High', 'Medium', or 'Low', NO ADDITIONAL EXPLANATION]

        Final Country Prediction: [Based on BOTH the AI image analysis AND your visual assessment, you MUST provide your final country prediction. ONLY THE COUNTRY NAME, NO ADDITIONAL EXPLANATION]
        """


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
            system_context = "You are an expert in traffic signs analysis."
            user_message = f"""This is a traffic sign possibly from {country} as detected by AI algorithms. Analyze it with extreme precision.

            ⚠️ CRITICAL WARNING ⚠️
            Text hallucination is strictly prohibited. DO NOT report seeing ANY text unless you can verify each individual character with 100% certainty.

            MANDATORY INSTRUCTIONS:
            1. TEXT VERIFICATION PROTOCOL: Before claiming ANY text exists:
               - Verify each character individually
               - If even a single character is uncertain, you MUST state "Text may be present but cannot be reliably transcribed"
               - NEVER attempt to guess words or complete partial text
               - NEVER translate text unless you are absolutely certain of both the characters and their meaning

            2. VISUAL DESCRIPTION ONLY:
               - Describe ONLY colors, shapes, borders, and symbols that are unambiguously visible
               - Use precise color terminology (e.g., "cobalt blue" rather than just "blue")
               - For symbols, describe their exact shape without interpretation unless absolutely certain

            3. CERTAINTY INDICATORS:
               - For each element described, indicate certainty level: "Clearly visible:", "Partially visible:", or "Suggested but unclear:"
               - Do not include any element in your analysis without these indicators

            FINAL VERIFICATION: Before submitting, review your response and remove ANY text or symbol descriptions that aren't 100% verifiable in the image.

            Based EXCLUSIVELY on verifiable visual elements, assess whether this sign's characteristics align with traffic signage from {country}."""
        elif prompt_type == 'license_plate':
            system_context = "You are an expert in license plate analysis."
            user_message = f"""This image shows a license plate. Please analyze it thoroughly to determine which country it likely originates from.

            In your analysis, focus specifically on:
            1. The overall format pattern (exact letter-number arrangement, spacing, and grouping)
            2. Presence of country identifiers (blue bands, flags, country codes, EU stars, or other national symbols)
            3. Color scheme (background color, character color, borders, and any special markings)
            4. First letters/numbers and their potential regional significance
            5. Font style and character appearance (Latin, Cyrillic, Arabic, etc.)
            6. Any distinctive features unique to specific countries (holograms, security features)

            If visible, note whether this appears to be a front or rear plate, as some countries use different formats.

            IMPORTANT: If there is insufficient evidence to determine the country of origin with reasonable confidence, you MUST explicitly state this limitation. Do not guess if the visual evidence is unclear or ambiguous.

            Based on these observations, provide your best assessment of which country this license plate is from, noting any distinctive regional indicators within that country (state, province, etc.) if identifiable. If multiple countries use similar formats, list the possibilities.
            """
        else:  # general scene analysis
            system_context = "You are an expert in outdoor scene analysis."
            user_message = f"This is an outdoor scene from {country}. Please describe what you see in detail, including notable landmarks, architectural styles, environmental features, and cultural elements visible."

        # Generate prompt and get LLaVA response
        prompt = PromptGenerator.get_analysis_prompt(system_context, user_message)
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
                result = image_service.process_image(file)
                log_memory_usage("After geographic processing")
                gc.collect()
                torch.cuda.empty_cache()

            # In the segmentation processing section, after you've processed the segmentation results
            with timer.time_process("Segmentation Processing"):
                log_memory_usage("Before segmentation")
                segmentation_results = image_service.process_automatic_segmentation(result['image_path'])

                # Define the reference data path as it exists in the container
                reference_data_path = "/app/segmentations"  # Adjust this to the actual mount point in your container

                logger.info(f"Using reference data path: {reference_data_path}")
                logger.info(f"Path exists: {os.path.exists(reference_data_path)}")


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
                    image_service.model_service._unload_current_model()
                    gc.collect()
                    torch.cuda.empty_cache()
                    # initial_prompt = PromptGenerator.get_analysis_prompt(system_context,
                    #                                                      "I want your detailed geo-localization analysis based on all the above. Clearly observable features that support the AI location analysis.")
                    initial_prompt = PromptGenerator.get_analysis_prompt(system_context,
                                                                         "I want your detailed geo-localization analysis based on all the above AND your final prediction of which country this is. Start with clearly observable features that support the AI location analysis, and end with a specific country prediction.")
                    args = LLaVAHandler.create_args(image_service.model_service, initial_prompt, result['image_path'])
                    log_memory_usage("Before LLaVA processing")
                    initial_llava_response = image_service.model_service.invoke_llava_model(args)
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
            logger.debug(f"Returning Prediction Message: {json.dumps(predictions, indent=2)}")
            logger.debug(f"Returning attention_map_path: {json.dumps(result['attention_map_path'], indent=2)}")

            logger.debug(f"Returning Continent Percentages: {json.dumps(result['continent_percentages'], indent=2)}")
            logger.debug(f"Returning Material Data: {json.dumps(result['material_data'], indent=2)}")
            logger.debug(f"Returning LLaVA Analysis: {json.dumps(parsed_response, indent=2)}")
            logger.debug(f"Returning Segmentation Results: {json.dumps(segmentation_results, indent=2)}")
            logger.debug(f"Returning Timing Summary: {json.dumps(timer.timings, indent=2)}")
            # If API request
            if request.headers.get('Accept') == 'application/json':
                return jsonify({
                    'status': 'success',
                    'image_path': result['image_path'],
                    'attention_map_path': result['attention_map_path'],
                    'prediction_message': predictions,
                    'continent_percentages': result['continent_percentages'],
                    'material_data': result['material_data'],
                    'system_context': system_context,
                    'llava_analysis': parsed_response,
                    'segmentation_results': segmentation_results,
                    'timing_summary': timer.timings
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