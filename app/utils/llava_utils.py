import logging
import torch
import gc
from llava.mm_utils import get_model_name_from_path

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

    # @staticmethod
    # def get_traffic_sign_analysis_prompt(system_context, user_message):
    #     return f"""AI image analysis:{system_context}
    #
    #     CRITICAL OCR MISSION: Your primary task is to correctly identify the ALPHABET SYSTEM and TEXT on this traffic sign.
    #
    #     1. ALPHABET IDENTIFICATION FIRST:
    #        • Greek: Λ Δ Σ Ω Θ Ε Ρ (Example: "ΑΘΗΝΑ", "ΛΑΜΙΑ")
    #        • Cyrillic: Я Ж Ц Ш Щ Ъ Ь (Example: "МОСКВА")
    #        • Latin: A-Z with variations
    #
    #     2. CHARACTER VERIFICATION:
    #        • Report ONLY characters you can verify with 100% certainty
    #        • NEVER list more than 10 items total
    #        • If text is present but unclear, state "Text visible but cannot be reliably transcribed"
    #
    #     3. SIGN ANALYSIS:
    #        • Color: Identify exact background/text colors
    #        • Shape: Note overall sign shape and border
    #        • Symbols: Describe arrows, route numbers, pictograms
    #
    #     ⚠️ CRITICAL ANTI-HALLUCINATION RULE ⚠️
    #     NEVER generate lists longer than 10 items. If you see multiple similar items (like route numbers),
    #     mention only 2-3 examples followed by "and similar items" rather than listing all of them.
    #
    #     User Question: {user_message}
    #
    #     ⚠️ CRITICAL FORMAT RULE ⚠️
    #     You MUST use this EXACT format in your response (including markers, bullet points and ALL sections):
    #
    #     Response: [Your answer here. If for something that you are asked you don't have enough clues you must indicate it]
    #
    #     Uncertainty Analysis:
    #
    #     - ALTERNATIVES: [Other potential interpretations that could lead your evidence into misinterpretation]
    #     - LIMITATIONS: [Factors that may affect your answer, such as image quality, ambiguous elements, cultural overlaps etc]
    #
    #     Confidence Level Assessment: [ONLY USE 'High', 'Medium', or 'Low', NO ADDITIONAL EXPLANATION]
    #
    #     Final Country Prediction: [Based on BOTH the AI image analysis AND your visual assessment, you MUST provide your final country prediction. ONLY THE COUNTRY NAME, NO ADDITIONAL EXPLANATION]
    #     """

    @staticmethod
    def get_traffic_sign_prompt(country):
        """
        Generate specialized traffic sign prompt with enhanced OCR and anti-hallucination measures
        """
        system_context = "You are an expert in European traffic signs analysis with specialized OCR capabilities."

        user_message = f"""Analyze this traffic sign to determine if it aligns with {country}'s signage.

        CRITICAL OCR MISSION: Your primary task is to correctly identify the ALPHABET SYSTEM and TEXT on this traffic sign.

        1. ALPHABET IDENTIFICATION FIRST:
           • Greek: Λ Δ Σ Ω Θ Ε Ρ (Example: "ΑΘΗΝΑ", "ΛΑΜΙΑ")
           • Cyrillic: Я Ж Ц Ш Щ Ъ Ь (Example: "МОСКВА")
           • Latin: A-Z with variations

        2. CHARACTER VERIFICATION:
           • Report ONLY characters you can verify with 100% certainty
           • NEVER list more than 10 items total
           • If text is present but unclear, state "Text visible but cannot be reliably transcribed"

        3. SIGN ANALYSIS:
           • Color: Identify exact background/text colors
           • Shape: Note overall sign shape and border
           • Symbols: Describe arrows, route numbers, pictograms

        ⚠️ CRITICAL ANTI-HALLUCINATION RULE ⚠️
        NEVER generate lists longer than 10 items. If you see multiple similar items (like route numbers),
        mention only 2-3 examples followed by "and similar items" rather than listing all of them.

        **Country-Specific Context for {country}:**
        {PromptGenerator._get_country_context(country)}

        **VISUAL FIRST APPROACH:**
        Begin by describing ONLY the sign's shape, color, and basic visual elements before attempting text recognition.

        Based EXCLUSIVELY on verifiable visual elements, assess whether this sign's characteristics align with traffic signage from {country}.
        """

        return system_context, user_message

    # @staticmethod
    # def get_traffic_sign_prompt(country):
    #     """
    #     Generate specialized traffic sign prompt with enhanced context
    #     """
    #     system_context = "You are an expert in European traffic signs analysis."
    #
    #     user_message = f"""Analyze this traffic sign to determine if it aligns with {country}'s signage.
    #
    #     **Domain Knowledge (MUST CONSIDER):**
    #     - Vienna Convention standards:
    #       • Warning: Red-bordered triangles
    #       • Prohibitory: Red-bordered circles
    #       • Mandatory: Blue circles
    #     - Primary differentiators: Language text > symbols > color shades
    #
    #     **Country-Specific Context for {country}:**
    #     {PromptGenerator._get_country_context(country)}
    #
    #     **Analysis Protocol:**
    #     1. Text Analysis (IF PRESENT):
    #        - Language: Verify characters match {country}'s official language(s)
    #        - Font: Note distinctive national font characteristics
    #     2. Visual Elements:
    #        • Shape: Compare to Vienna Convention standards
    #        • Colors: Use precise terminology (e.g., Pantone 485C red)
    #        • Symbols: Describe EXACTLY without interpretation
    #     3. Unique {country} Features Check
    #
    #     **Absolute Restrictions:**
    #     - NO text interpretation unless 100% character certainty
    #     - NO symbol guessing - describe only visible elements"""
    #
    #     return system_context, user_message

    @staticmethod
    def _get_country_context(country):
        """
        Return country-specific traffic sign context for geolocation, focusing on visual/textual clues.
        """
        country = country.strip().lower()
        if country in ["netherlands", "amsterdam"]:
            return (
                "• Language: Dutch (e.g., 'Fietspad', 'Uitgezonderd', 'Gracht')\n"
                "• Unique: Canal warning triangles (often with pictograms of cars falling into water), "
                "and blue rectangular 'Fietsstraat' (bicycle street) signs\n"
                "• Font: ANWB/Rijkswaterstaat typeface, squared '0'\n"
                "• Speed limits: Standard circular red-bordered signs, Dutch text possible\n"
                "• Regulatory framework: RVV 1990[3][7]"
            )
        elif country in ["greece", "athens"]:
            return (
                "• Language: Greek (Greek alphabet), sometimes bilingual with English\n"
                "• Unique: Yellow-background warning triangles (red border, black pictogram), "
                "realistic human silhouettes, orange backgrounds for temporary signs\n"
                "• Font: Transport (since 1974), DIN 1451 on motorways\n"
                "• Most signs follow Vienna Convention shapes and colors\n"
                "• Night: Highly reflective or illuminated signs[4]"
            )
        elif country in ["germany", "berlin"]:
            return (
                "• Language: German (e.g., 'Umweltzone', 'Anlieger frei', 'Baustelle')\n"
                "• Unique: Umweltzone (environmental zone) discs, brown historical route markers (esp. Berlin Wall), "
                "'Snow chains required' blue circle, minimum following distance, detailed parking/no waiting signs\n"
                "• Font: DIN 1451\n"
                "• Color: RAL 3020 red for borders\n"
                "• Regulatory: Vienna Convention[5]"
            )
        elif country in ["hungary", "budapest"]:
            return (
                "• Language: Hungarian (e.g., 'Utca', 'Tér', 'Behajtani tilos', 'Vigyázz')\n"
                "• Unique: Modern navigation signs (since 2025), sometimes multilingual at major hubs\n"
                "• Font: Hungarian road sign font, similar to DIN\n"
                "• Bridge and hill warnings common, Danube bridges often named\n"
                "• Regulatory: Vienna Convention"
            )
        elif country in ["finland", "helsinki"]:
            return (
                "• Language: Bilingual Finnish/Swedish (e.g., 'Keskusta / Centrum', 'Varoitus / Varning')\n"
                "• Unique: Snow/ice warnings, ferry/harbor/tram signs, winter route markers\n"
                "• Font: Finnish road sign font\n"
                "• Regulatory: Vienna Convention"
            )
        elif country in ["united kingdom", "uk", "london"]:
            return (
                "• Language: English\n"
                "• Unique: Congestion charge zone (red circle, white 'C'), left-side driving arrows, "
                "distinct roundabout and 'Give Way' signs\n"
                "• Font: Transport\n"
                "• Regulatory: Not Vienna Convention; UK-specific shapes and priorities"
            )
        elif country in ["russia", "moscow"]:
            return (
                "• Language: Russian (Cyrillic alphabet)\n"
                "• Unique: Moscow Metro 'M' logo, Cyrillic on all signs, winter and lane management signs\n"
                "• Font: Russian road sign font\n"
                "• Color: Slightly different reds/blues than Western Europe\n"
                "• Regulatory: GOST standards (not Vienna Convention)"
            )
        elif country in ["france", "paris"]:
            return (
                "• Language: French (e.g., 'Rue', 'Boulevard', 'Interdit', 'Zone piétonne')\n"
                "• Unique: Boulevard Périphérique (green/blue signs with 'Périphérique'), "
                "historic district blue/brown markers, roundabouts common (triangular sign with circular arrows)\n"
                "• Font: Caractères\n"
                "• Regulatory: Vienna Convention, priority to the right unless otherwise posted[6]"
            )
        elif country in ["sweden", "stockholm"]:
            return (
                "• Language: Swedish (e.g., 'Gata', 'Väg', 'Miljözon', 'Färja')\n"
                "• Unique: Miljözon (environmental zone) markers, ferry/island access, winter warnings\n"
                "• Font: Swedish road sign font\n"
                "• Regulatory: Vienna Convention"
            )
        elif country in ["norway", "trondheim"]:
            return (
                "• Language: Norwegian (e.g., 'Gate', 'Vei', 'Farlig sving', 'Kollektivfelt')\n"
                "• Unique: Bicycle elevator (Trampe) signage, fjord proximity, climate/winter route signs\n"
                "• Font: Norwegian road sign font\n"
                "• Regulatory: Vienna Convention"
            )
        elif country in ["switzerland", "zurich"]:
            return (
                "• Language: Primarily German, sometimes French/Italian (e.g., 'Ausfahrt / Sortie / Uscita')\n"
                "• Unique: Multilingual on one sign, altitude/gradient, tram and lake proximity ('Zürichsee')\n"
                "• Font: Swiss road sign font\n"
                "• Regulatory: Vienna Convention"
            )
        else:
            return (
                "• Primary Identifier: Language on sign\n"
                "• Check for country-specific pictograms, fonts, or colors\n"
                "• Most European countries follow Vienna Convention standards"
            )

    @staticmethod
    def get_license_plate_prompt(country):
        """
        Generate a specialized prompt for license plate analysis.

        Args:
            country (str): The country to consider in the analysis

        Returns:
            tuple: (system_context, user_message)
        """
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

        return system_context, user_message

    @staticmethod
    def get_outdoor_scene_prompt(country):
        """
        Generate a specialized prompt for general outdoor scene analysis.

        Args:
            country (str): The country to consider in the analysis

        Returns:
            tuple: (system_context, user_message)
        """
        system_context = "You are an expert in outdoor scene analysis."
        user_message = f"This is an outdoor scene from {country}. Please describe what you see in detail, including notable landmarks, architectural styles, environmental features, and cultural elements visible."

        return system_context, user_message