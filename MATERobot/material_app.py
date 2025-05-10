from .materobot.apis import inference_moe_model, inference_single_model, init_model, show_result_pyplot
from mmseg.utils import register_all_modules
import os
import numpy as np
# config_path = 'materobot/configs/matevit_vit-t_single-task_dms.py'
# checkpoint_path = 'pretrain/best_mIoU_epoch_89.pth'
# # register all modules in mmseg into the registries
# register_all_modules()
# # build the model from a config file and a checkpoint file
# model = init_model(config_path, checkpoint_path, device='cpu')
import json
from torchvision.io import read_image
from PIL import Image, ImageFile
import torch
import shutil

ImageFile.LOAD_TRUNCATED_IMAGES = True




# print('Unique Predicted Materials:')
# print(pred_materials)
# visualize results
# os.system(f'cp {img_path} work_dirs/ori.png')

# show_result_pyplot(model, img_path, result[0], show=False, opacity=0.8, out_file='work_dirs/dms.png', save_dir='work_dirs')  # visualize moe model

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

    # import pdb; pdb.set_trace()
    pred_labels = result[0].pred_sem_seg.data.squeeze().unique().tolist()

    # Print unique predicted material names
    pred_materials = [model.dataset_meta['classes'][label] for label in pred_labels]
    info_mat= calculate_material_proportions(model,result[0])
    # print(full_path + " info_mat: " + info_mat)
    record =  info_mat
    print(record)
    mask_path = show_result_pyplot(model, im_path, result[0], show=False, opacity=0.8, out_file=(str(uuid.uuid4()) + '.png'),
                       save_dir="/app/static")
    return record, mask_path

# with open('material_info.txt', 'a') as f:
#     f.write(record + "\n")  # \n means newline