import logging
import torch
import gc
from llava.mm_utils import get_model_name_from_path
import argparse
import torch
import requests
from PIL import Image
from io import BytesIO
import re
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

# --- Helper Functions ---

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

# --- Main Logic Function ---

def eval_model_with_global_model(args, global_model, global_tokenizer, global_image_processor, model_name):
    # Use the globally loaded model, tokenizer, and image processor
    model = global_model
    tokenizer = global_tokenizer
    image_processor = global_image_processor

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    
    # Handle Image Placeholders
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # Determine Conversation Mode
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(f"[WARNING] the auto inferred conversation mode is {conv_mode}, while `--conv-mode` is {args.conv_mode}, using {args.conv_mode}")
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Process Images
    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    # Run Generation
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

logger = logging.getLogger(__name__)


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