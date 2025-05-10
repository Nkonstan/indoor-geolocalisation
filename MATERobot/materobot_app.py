import uuid
from flask import Flask, request, jsonify
from materobot.apis import inference_moe_model, init_model, show_result_pyplot
from mmseg.utils import register_all_modules
import torch
import logging
import os
import sys
import numpy as np
import gc
import sys

logger = logging.getLogger(__name__)
app = Flask(__name__)
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


def cleanup_gpu_memory():
    """Helper function to cleanup GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()


# Initialize MATERobot model
def init_materobot_model(device):
    config_path = '/app/MATERobot/materobot/configs/matevit_vit-t_single-task_dms.py'
    checkpoint_path = '/app/MATERobot/pretrain/best_mIoU_epoch_89.pth'
    register_all_modules()
    model = init_model(config_path, checkpoint_path, device='cpu')  # Initialize on CPU
    return model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
materobot_model = init_materobot_model(device)


def calculate_material_proportions(model, result) -> str:
    class_names = model.dataset_meta['classes']
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
    try:
        # Move model to GPU for inference
        materobot_model.to(device)

        result = inference_moe_model(materobot_model, im_path)
        sem_seg_2D = result.pred_sem_seg.data.cpu().numpy().squeeze()
        pred_labels = np.unique(sem_seg_2D).tolist()
        pred_materials = [materobot_model.dataset_meta['classes'][label] for label in pred_labels]
        info_mat = calculate_material_proportions(materobot_model, result)

        mask_filename = str(uuid.uuid4()) + '.png'
        save_dir = "/app/static"
        out_file_path = os.path.join(save_dir, mask_filename)

        vis_img = show_result_pyplot(materobot_model, im_path, result, show=False, opacity=0.8,
                                     out_file=out_file_path, save_dir=save_dir)

        logger.info(f"Mask path: {out_file_path}")
        return info_mat, out_file_path

    finally:
        # Move model back to CPU and cleanup
        materobot_model.cpu()
        cleanup_gpu_memory()


@app.route('/material_recognition', methods=['POST'])
def material_recognition():
    try:
        data = request.json
        image_path = data['image_path']
        abs_image_path = os.path.join('/app', image_path)
        logger.info(f"Received image path: {abs_image_path}")

        if not os.path.exists(abs_image_path):
            logger.error(f"Image path does not exist: {abs_image_path}")
            return jsonify({'error': f"Image path does not exist: {abs_image_path}"}), 400

        image_materials, mask_image_full_path = material_pred(abs_image_path)
        if image_materials is None or mask_image_full_path is None:
            raise ValueError("Failed to process material recognition.")

        logger.info(f"Material recognition successful: {image_materials}, {mask_image_full_path}")

        if isinstance(image_materials, np.ndarray):
            image_materials = image_materials.tolist()

        if not os.path.exists(mask_image_full_path):
            logger.error(f"Mask image file does not exist: {mask_image_full_path}")
            return jsonify({'error': f"Mask image file does not exist: {mask_image_full_path}"}), 500

        mask_image_url = f"/static/{os.path.basename(mask_image_full_path)}"

        logger.info(f"Mask image full path: {mask_image_full_path}")
        logger.info(f"Mask image URL: {mask_image_url}")

        return jsonify({'image_materials': image_materials, 'mask_image_path': mask_image_url})

    except Exception as e:
        logger.error(f"Exception in material_recognition: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        cleanup_gpu_memory()


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)