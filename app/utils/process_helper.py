import json
import logging

logger = logging.getLogger(__name__)

class ProcessHelper:
    @staticmethod
    def parse_predictions(prediction_data):
        """
        Handles the parsing of predictions with robust logging.
        """
        try:
            logger.debug(f"Raw prediction data type: {type(prediction_data)}")
            
            # 1. Normalize input to dict
            if isinstance(prediction_data, str):
                logger.debug("Parsing prediction string to JSON")
                predictions = json.loads(prediction_data)
            else:
                # [CRITICAL FIX] Handle the case where it's already a dict
                logger.debug("Prediction data is already a dict, using directly")
                predictions = prediction_data

            # Log the keys to prove it exists before we use it
            logger.debug(f"Predictions keys found: {list(predictions.keys())}")

            # 2. Find top country efficiently
            top_country_item = max(
                predictions.items(), 
                key=lambda x: float(str(x[1]).replace('%', ''))
            )
            
            top_country_name = top_country_item[0]
            top_score_val = float(str(top_country_item[1]).replace('%', ''))
            
            logger.info(f"Top country found: {top_country_name} ({top_score_val}%)")
            
            return predictions, (top_country_name, top_score_val)

        except Exception as e:
            # This log will tell us exactly why it failed if it happens again
            logger.error(f"ProcessHelper Error: {str(e)}")
            raise e

    @staticmethod
    def format_segmentation_text(segmentation_results):
        """Converts the complex segmentation dictionary into a readable string."""
        if not segmentation_results:
            return "No architectural elements detected"

        analysis_lines = []
        for element_type, element_data in segmentation_results.items():
            if element_data.get('similar_segments_info'):
                similar_countries = [
                    info['country'] 
                    for info in element_data['similar_segments_info']
                ]
                confidence = element_data.get('confidence', 0) * 100
                
                line = (
                    f"{element_type.title()} - detected with {confidence:.1f}% confidence. "
                    f"Most similar {element_type.title()} styles found in: {', '.join(similar_countries)}"
                )
                analysis_lines.append(line)

        return "\n      ".join(analysis_lines) if analysis_lines else "No architectural elements detected"

    @staticmethod
    def build_system_context(result, predictions, top_country_tuple, segmentation_text, country_description):
        """
        Reconstructs the system_context EXACTLY as it appeared in the original code.
        Includes exact whitespace/indentation and excludes unused material matches.
        """
        top_country_name, top_score = top_country_tuple
        
        # Format dynamic continents
        continent_text = ", ".join([
            f"{continent}: {percentage:.1f}%"
            for continent, percentage in result['continent_percentages'].items()
            if percentage > 0
        ])

        # Format predictions
        predictions_text = ", ".join([
            f"{country} ({score}%)"
            for country, score in predictions.items()
        ])

        # Format Materials
        materials_list = result['material_data']['image_materials'].split('%')
        materials_text = ", ".join([mat.strip() + '%' for mat in materials_list if mat.strip()])

        return f"""Image Analysis Results:
                1. Location Analysis:
                   CONTINENTAL DISTRIBUTION:
                   {continent_text}

                   COUNTRY PREDICTIONS:
                   {predictions_text}

                2. Typical Interior Characteristics of {top_country_name} ({top_score}%):
                {country_description}

                3. Material Analysis:
                {materials_text}

                4. Architectural Elements Analysis:
                  {segmentation_text}"""