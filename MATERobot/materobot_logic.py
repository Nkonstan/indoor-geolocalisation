import torch
import logging
import os
import sys
import numpy as np
import gc
import pandas as pd
import uuid
from math import sqrt
from materobot.apis import inference_moe_model, init_model, show_result_pyplot
from mmseg.utils import register_all_modules

logger = logging.getLogger(__name__)

# --- Configuration Constants ---
CSV_PATH = '/app/average_material_scores_per_country.csv'

# Singleton class for MATERobot model
class MATERobotModel:
    """Singleton class for MATERobot model"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        config_path = '/app/MATERobot/materobot/configs/matevit_vit-t_single-task_dms.py'
        checkpoint_path = '/app/MATERobot/pretrain/best_mIoU_epoch_89.pth'
        
        # Register all modules in mmseg into the registries
        register_all_modules()
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # Build the model from a config file and a checkpoint file
        self.model = init_model(config_path, checkpoint_path, device=device)
        
        # CRITICAL: Disable gradients for ALL parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Force garbage collection after loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("MATERobot model initialized successfully")

CSV_PATH = '/app/average_material_scores_per_country.csv'
if os.path.exists(CSV_PATH):
    logger.info(f"CSV file found at {CSV_PATH}")
else:
    logger.error(f"CSV file not found at {CSV_PATH}. Material country matching will fail.")

def find_material_country_matches(detected_materials, csv_path='/app/average_material_scores_per_country.csv'):
    """
    Find the closest country match for each detected material.

    Parameters:
    detected_materials (str): String of detected materials and their proportions
    csv_path (str): Path to the CSV file with country material distributions

    Returns:
    dict: Dictionary mapping each material to its closest country match
    """
    try:
        # Load the CSV data
        df = pd.read_csv(csv_path)

        # Parse the detected materials string into a dictionary
        material_dict = {}
        prev_item = None
        for item in detected_materials.split():
            if '%' in item:
                # Convert percentage to proportion (e.g., 40.25% -> 0.4025)
                proportion = float(item.strip('%')) / 100
                if prev_item:
                    material_dict[prev_item] = proportion
            else:
                prev_item = item

        # Find closest country for each material
        material_country_matches = {}

        for material, proportion in material_dict.items():
            # Check if this material exists in our dataset (as a row)
            if material in df['Country'].values:
                # Get the row for this material
                material_row = df[df['Country'] == material].iloc[0]

                # Find country with highest value for this material
                countries = [col for col in df.columns if col != 'Country']
                best_country = None
                highest_value = -1

                for country in countries:
                    country_value = material_row[country]
                    if country_value > highest_value:
                        highest_value = country_value
                        best_country = country

                material_country_matches[material] = {
                    "country": best_country,
                    "country_value": round(highest_value * 100, 1),  # Convert to percentage
                    "detected_value": round(proportion * 100, 1)  # Convert to percentage
                }

        return material_country_matches
    except Exception as e:
        logger.error(f"Error finding material country matches: {e}")
        return {}

def calculate_material_proportions(model, result) -> str:
    class_names = model.dataset_meta['classes']
    # Move tensor to CPU immediately
    sem_seg_2D = result.pred_sem_seg.data.cpu().numpy().squeeze()
    label_counts = {class_name: 0 for class_name in class_names}
    total_pixels = sem_seg_2D.size

    for label in np.unique(sem_seg_2D):
        if label == 0:
            continue
        mask = sem_seg_2D == label
        label_counts[class_names[label]] += np.sum(mask)

    label_proportions = {k: v / total_pixels for k, v in label_counts.items() if v > 0}
    material_info = " ".join(f"{label} {proportion * 100:.2f}%" for label, proportion in label_proportions.items())
    return material_info

def material_pred(im_path):
    # Get singleton model instance
    materobot_model = MATERobotModel.get_instance()
    
    try:
        logger.info(f"Starting material prediction for {im_path}")
        
        # Use torch.no_grad() for inference
        with torch.no_grad():
            logger.info("Running inference on image")
            result = inference_moe_model(materobot_model.model, im_path)
            logger.info("Inference completed successfully")

            # Move result to CPU immediately
            sem_seg_2D = result.pred_sem_seg.data.cpu().numpy().squeeze()
            pred_labels = np.unique(sem_seg_2D).tolist()
            logger.info(f"Detected material labels: {pred_labels}")

            pred_materials = [materobot_model.model.dataset_meta['classes'][label] for label in pred_labels]
            logger.info(f"Detected materials: {pred_materials}")

            info_mat = calculate_material_proportions(materobot_model.model, result)
            logger.info(f"Material proportions: {info_mat}")

            # Get material country matches
            logger.info("Finding material country matches")
            material_country_matches = find_material_country_matches(info_mat)
            logger.info(f"Material country matches: {material_country_matches}")

            mask_filename = str(uuid.uuid4()) + '.png'
            save_dir = "/app/static"
            out_file_path = os.path.join(save_dir, mask_filename)
            logger.info(f"Generating visualization mask at {out_file_path}")

            vis_img = show_result_pyplot(materobot_model.model, im_path, result, show=False, opacity=0.8,
                                         out_file=out_file_path, save_dir=save_dir)

            logger.info(f"Mask path: {out_file_path}")
            
            # Clean up result tensors
            del result, pred_labels, sem_seg_2D
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return info_mat, out_file_path, material_country_matches

    except Exception as e:
        logger.error(f"Error in material_pred: {e}", exc_info=True)
        
        # Force cleanup on error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return None, None, None