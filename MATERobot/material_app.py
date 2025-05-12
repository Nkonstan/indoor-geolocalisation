from .materobot.apis import inference_moe_model, inference_single_model, init_model, show_result_pyplot
from mmseg.utils import register_all_modules
import os
import numpy as np
import json
from torchvision.io import read_image
from PIL import Image, ImageFile
import torch
import shutil

ImageFile.LOAD_TRUNCATED_IMAGES = True

import pandas as pd
import numpy as np


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

        print(f"Material country matches: {material_country_matches}")
        return material_country_matches
    except Exception as e:
        print(f"Error finding material country matches: {e}")
        return {}


def calculate_material_proportions(model, result) -> str:
    """Calculate the proportions of each material in the segmentation result."""

    # Get classes names
    class_names = model.dataset_meta['classes']

    # Convert tensor to 2D numpy array
    sem_seg_2D = result.pred_sem_seg.data.squeeze().numpy()

    # Initialize a dictionary to store counts of each label
    label_counts = {class_name: 0 for class_name in class_names}
    total_pixels = sem_seg_2D.size

    # Iterate through each unique label in the predicted segmentation map
    for label in np.unique(sem_seg_2D):
        if label == 0:  # Typically, 0 is the label for background. If not, remove this check.
            continue
        mask = sem_seg_2D == label
        label_counts[class_names[label]] += np.sum(mask)

    # Compute the proportions
    label_proportions = {k: v / total_pixels for k, v in label_counts.items() if v > 0}

    # Prepare the output string
    material_info = " ".join(f"{label} {proportion*100:.2f}%" for label, proportion in label_proportions.items())

    return material_info
import uuid


def material_pred(im_path):
    config_path = '/app/material_recognition/MATERobot/materobot/configs/matevit_vit-t_single-task_dms.py'
    checkpoint_path = '/app/material_recognition/MATERobot/pretrain/best_mIoU_epoch_89.pth'
    # register all modules in mmseg into the registries
    register_all_modules()
    # build the model from a config file and a checkpoint file
    model = init_model(config_path, checkpoint_path, device='cpu')
    try:
        # Your image processing code here
        result = inference_moe_model(model, im_path)
    except RuntimeError as e:
        print(f"Error processing image {im_path}: {e}")
        return None, None, None

    # import pdb; pdb.set_trace()
    pred_labels = result[0].pred_sem_seg.data.squeeze().unique().tolist()

    # Print unique predicted material names
    # pred_materials = [model.dataset_meta['classes'][label] for label in pred_labels]
    info_mat = calculate_material_proportions(model, result[0])

    # Get material country matches
    material_country_matches = find_material_country_matches(info_mat)

    # Print record
    record = info_mat
    print(record)

    mask_path = show_result_pyplot(model, im_path, result[0], show=False, opacity=0.8,
                                   out_file=(str(uuid.uuid4()) + '.png'),
                                   save_dir="/app/static")

    return record, mask_path, material_country_matches

# with open('material_info.txt', 'a') as f:
#     f.write(record + "\n")  # \n means newline