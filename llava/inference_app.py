import torch
from PIL import Image
from torchvision import transforms
from scipy.spatial import distance
from collections import Counter, defaultdict
from operator import itemgetter
import json
from network import DeiT384, DeiT384_exp  # Make sure the network file is in the same directory or correctly installed
import shutil
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import uuid
import pickle
import pandas as pd
import ast
from scipy.spatial.distance import hamming  # Ensure hamming is imported

def image_transform(resize_size, crop_size):
    return transforms.Compose([
        # transforms.ColorJitter(0,0,0,0),  # this converts grayscale images to RGB
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_feature_vector(net, device, image_path, transform):
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0).to(device)

    # Only get the output, ignore the attention
    features = net(img)[0].sign()
    features = features.cpu().detach().numpy()  # Moving tensor to CPU and converting to numpy
    return features


# Function to load the CSV database
def parse_binary_code(binary_code_str):
    try:
        # Replace decimal points with spaces and split into individual binary digits
        binary_code_str = binary_code_str.replace('.', ' ').replace('[', '').replace(']', '').split()
        binary_code_list = [int(bit) for bit in binary_code_str]
        return np.array(binary_code_list, dtype=int).flatten()
    except Exception as e:
        logger.error(f"Error parsing binary code: {binary_code_str} - {e}")
        return np.array([])

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_database(csv_file):
    df = pd.read_csv(csv_file)
    df['Binary Code'] = df['Binary Code'].apply(parse_binary_code)
    df['Full Path'] = df.apply(lambda row: os.path.join('segmentations', row['Country'], row['Segment'], row['Image Name']), axis=1)
    # Log the unique segment names
    unique_segments = df['Segment'].unique()
    logger.info(f"Unique segment names in the loaded database: {unique_segments}")

    return df



def classify_segment(user_feature, df):
    # Ensure user_feature is 1-D
    user_feature = user_feature.flatten()

    # Function to check if a vector is already in 0 and 1 format
    def is_binary_format(vector):
        return np.array_equal(vector, vector.astype(bool))

    # Ensure user_feature is in 0 and 1 format
    if not is_binary_format(user_feature):
        user_feature = (user_feature + 1) // 2

    # Ensure all binary codes in the dataframe are in 0 and 1 format
    df['Binary Code'] = df['Binary Code'].apply(lambda x: (x.flatten() + 1) // 2 if not is_binary_format(x) else x)

    # Compute Hamming distances
    def compute_distance(x):
        flat_x = x.flatten()
        return hamming(user_feature, flat_x)

    distances = df['Binary Code'].apply(compute_distance)
    df = df.copy()  # Avoid SettingWithCopyWarning
    df['Distance'] = distances

    # Perform initial filtering to quickly narrow down segment classes
    initial_closest_matches = df.nsmallest(120, 'Distance')  # Adjust the number for initial filtering as needed
    initial_segment_classes = initial_closest_matches['Segment'].unique()

    # Further refine comparison within the filtered segment classes
    filtered_df = df[df['Segment'].isin(initial_segment_classes)]
    closest_matches = filtered_df.nsmallest(70, 'Distance')  # Adjust the number of comparisons as needed

    # Majority voting
    segment_class = closest_matches['Segment'].mode()[0]
    print(f"Segment Class: {segment_class}")

    return segment_class, closest_matches



# Function to retrieve the most similar segments
def retrieve_similar_segments(segment_class, query_vec, segmentation_database, top_n=10):
    # Print the column names of segmentation_database for debugging
    print("segmentation_database columns:", segmentation_database.columns)

    # Use the correct column name for class, assuming it is 'Segment' based on the provided columns
    class_column_name = 'Segment'  # Adjust this based on your actual column name

    class_df = segmentation_database[segmentation_database[class_column_name] == segment_class]
    # class_df['Distance'] = class_df['vector'].apply(lambda vec: np.linalg.norm(vec - query_vec))
    class_df['Distance'] = class_df['Binary Code'].apply(lambda vec: np.linalg.norm(vec - query_vec))

    # Convert 'Distance' column to numeric type
    class_df['Distance'] = pd.to_numeric(class_df['Distance'], errors='coerce')

    # Handle potential NaN values after conversion
    class_df = class_df.dropna(subset=['Distance'])

    similar_segments = class_df.nsmallest(top_n, 'Distance')
    return similar_segments


def predict_class(db_binary, db_labels, db_paths, query_vec, topk):
    # import pdb; pdb.set_trace()
    print("db_binary[0]",db_binary[0] , db_binary[0].shape )
    print("query_vec[0]",query_vec[0] , query_vec[0].shape )
    query_vec = query_vec.flatten()

    hamming_distances = [distance.hamming(query_vec, db_vec) for db_vec in db_binary]
    topk_indices = np.argsort(hamming_distances)[:topk]

    label_counts = defaultdict(int)
    topk_paths = []
    for idx in topk_indices:
        label_counts[db_labels[idx]] += 1
        topk_paths.append(db_paths[idx])
        print("path: ", db_paths[idx])
    return label_counts, topk_paths



# Function to load the model
def load_model(device):
    net = DeiT384_exp(128).to(device)
    net.load_state_dict(torch.load("./airbnb_14countries_train_database_128bits_0.6296825813840561/model.pt", map_location=device))
    net.eval()
    return net

# Function to preprocess and load image
def preprocess_image(image_path, transform, device):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor, image

# Function to compute and process attention map
def compute_attention_map(net, image_tensor):
    output, attention = net(image_tensor, return_attention=True)
    attention = attention.sum(axis=1).squeeze(0)
    attention_patches = attention[1:-1, 1:-1]
    attention_patches_sum = attention_patches.sum(axis=0)
    attention_map = attention_patches_sum.reshape(24, 24).detach().cpu().numpy()
    attention_map = np.power(attention_map, 0.5)
    threshold = np.percentile(attention_map, 60)
    attention_map = np.where(attention_map > threshold, attention_map, 0)
    return attention_map

# Function to overlay heatmap on image
def overlay_heatmap_on_image(attention_map, original_image):
    original_image = np.array(original_image)
    original_image = cv2.resize(original_image, (384, 384))
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap[:, :, 1] = 0
    heatmap[:, :, 2] = 0
    heatmap_transparent = cv2.addWeighted(heatmap, 0.6, original_image, 0.4, 0)
    return heatmap_transparent

# Your main function
def base_predictor(image_path, net, device, not_segment=True):
    transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if not_segment:

        image_tensor, original_image = preprocess_image(image_path, transform, device)
        attention_map = compute_attention_map(net, image_tensor)
        heatmap_transparent = overlay_heatmap_on_image(attention_map, original_image)

        attention_map_name = str(uuid.uuid4()) + '.png'
        attention_map_path = os.path.join('static', attention_map_name)
        plt.imshow(heatmap_transparent)
        plt.savefig(attention_map_path)  # Save the image
        # plt.show()

        query_vec = get_feature_vector(net, device, image_path, transform)

        db_binary = np.load('./save_binarycodes_paths_labels_new/airbnb_14countries_trainedinwholedata_database_binary.npy')
        db_paths = pickle.load(open('./save_binarycodes_paths_labels_new/airbnb_14countries_trainedinwholedatabase_128bits_0.6296_database_labelspaths.ob', 'rb'))
        db_labels = [path.split('/')[2] for path in db_paths]
        db_binary = np.where(db_binary == -1, 0, db_binary)
        query_vec = np.where(query_vec == -1, 0, query_vec)
        # label_counts = predict_class(db_binary, db_labels, query_vec, topk=500)

        # db_paths = [path for path in db_data]  # assuming that `db_data` contains paths

        label_counts, topk_paths = predict_class(db_binary, db_labels, db_paths, query_vec, topk=500)

        # Convert counts into percentages
        total_count = sum(label_counts.values())
        label_percentages = {label: round((count / total_count)*100,2) for label, count in label_counts.items()}

        # Sort the dictionary based on the percentage
        sorted_label_percentages = dict(sorted(label_percentages.items(), key=itemgetter(1), reverse=True))

        print(json.dumps(sorted_label_percentages, indent=2))

        prediction  = json.dumps(sorted_label_percentages, indent=2)
        return prediction, attention_map_path

    else:

        query_vec = get_feature_vector(net, device, image_path, transform)
        query_vec = np.where(query_vec == -1, 0, query_vec)

        return query_vec
    # Copy top10 images to a new directory
    # top10_paths = topk_paths[:10]
    # for idx, img_path in enumerate(top10_paths):
    #     shutil.copy(img_path, f"./temp/{idx}.jpg")


