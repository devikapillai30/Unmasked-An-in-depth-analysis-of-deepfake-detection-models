'''
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
import pickle
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import yaml
from tqdm import tqdm
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset


from trainer.trainer import Trainer
from detectors import DETECTOR

import argparse
from torchvision import transforms
    
def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])
        
# Function to load data from a pickle file
def load_pickle_data(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Function to load and preprocess images from file paths
def load_images_from_paths(image_paths, transform):
    images = []
    for path in image_paths:
        img = Image.open(path)  # Open the image
        #print('getting images')
        img_tensor = transform(img)  # Apply transformation (resize, to tensor, etc.)
        images.append(img_tensor)
        print('stacking images')
    return torch.stack(images)  # Stack the list of tensors into a batch

# Define the image transformation (you can customize it based on your model's requirements)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image to 224x224 (modify based on your model input)
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize (ImageNet stats)
])

# Feature extraction using your model's `features` method
def extract_custom_features_from_pickle(model, data, device):
    model.eval()  # Set model to evaluation mode
    forgery_features_list = []
    labels_list = []

    with torch.no_grad():
        # Assuming 'image' contains file paths, and 'label' contains the labels in your pickle file
        image_paths = data['image']
        print('got images')
        labels = torch.tensor(data['label']).to(device)  # Convert labels to tensor and send to device
        print('labels loaded')
        # Create batches if needed, here we assume the data can fit in one forward pass
        batch_size = 32  # Adjust batch size according to your GPU memory
        num_batches = len(image_paths) // batch_size
        print(num_batches)

        for i in tqdm(range(10), desc="Processing Batches"):
            batch_image_paths = image_paths[i * batch_size: (i + 1) * batch_size]
            batch_labels = labels[i * batch_size: (i + 1) * batch_size]
            print(batch_image_paths)
            # Load and preprocess the images from file paths
            batch_images = load_images_from_paths(batch_image_paths, transform).to(device)

            # Extract forgery-related features using your model's method
            print('creating datadict')
            data_dict = {'image': batch_images}
            print('getting feat_dict')
            feat_dict = model.features(data_dict)
            f_all = feat_dict['forgery']  # Get forgery features from the feature extraction method
            print('got forgery')

            forgery_features_list.append(f_all.cpu().numpy())
            labels_list.append(batch_labels.cpu().numpy())
            print('appends done')
            torch.cuda.empty_cache()

        # If there are remaining samples not fitting into a full batch
        if len(image_paths) % batch_size != 0:
            batch_image_paths = image_paths[num_batches * batch_size:]
            batch_labels = labels[num_batches * batch_size:]

            # Load and preprocess the images from file paths
            batch_images = load_images_from_paths(batch_image_paths, transform).to(device)

            data_dict = {'image': batch_images}
            feat_dict = model.features(data_dict)
            f_all = feat_dict['forgery']

            forgery_features_list.append(f_all.cpu().numpy())
            labels_list.append(batch_labels.cpu().numpy())
            torch.cuda.empty_cache()

    # Concatenate features and labels into arrays
    chunk_size = 1000  # Adjust based on your memory capacity
    all_forgery_features = []
    all_labels = []

    for i in range(0, len(forgery_features_list), chunk_size):
        chunk_features = np.concatenate(forgery_features_list[i:i + chunk_size], axis=0)
        chunk_labels = np.concatenate(labels_list[i:i + chunk_size], axis=0)
    
        all_forgery_features.append(chunk_features)
        all_labels.append(chunk_labels)

    # Finally concatenate all chunks if needed
    all_forgery_features = np.concatenate(all_forgery_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    #all_forgery_features = np.concatenate(forgery_features_list, axis=0)
    #all_labels = np.concatenate(labels_list, axis=0)
    print('returning')
    return all_forgery_features, all_labels

# Apply t-SNE and visualize the results
def plot_tsne(features, labels):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(features)
    print('Applied t-SNE')

    # Define unique classes and corresponding colors
    unique_labels = np.unique(labels)
    colors = ['blue', 'orange', 'green', 'red', 'purple']  # Define your colors here

    # Create a color mapping for each label
    color_mapping = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Plotting the t-SNE results
    plt.figure(figsize=(10, 7))

    # Create a scatter plot with single colors for each class
    for label in unique_labels:
        plt.scatter(tsne_results[labels == label, 0], 
                    tsne_results[labels == label, 1], 
                    color=color_mapping[label], 
                    label=f'Class {label}', 
                    alpha=0.7)
    plt.title("t-SNE Visualization of Forgery Features")
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    plt.savefig('tsne_visualization.png')  # Specify your desired filename and format
    plt.close()

# Load pickle data, extract features, and perform t-SNE
def tsne_from_pickle(pickle_file_path, model, device):
    # Load data from pickle
    data = load_pickle_data(pickle_file_path)
    print('got data')
    
    # Extract forgery features from the loaded data using the model
    forgery_features, labels = extract_custom_features_from_pickle(model, data, device)
    print(forgery_features.shape)
    forgery_features = forgery_features.reshape(forgery_features.shape[0], -1)
    print(forgery_features.shape)
    
    # Perform t-SNE and plot the results
    plot_tsne(forgery_features, labels)


# Example usage:
pickle_file_path = './log/ucf_2024-10-12-02-33-52/train/DeepFakes,Face2Face,FaceSwap,NeuralTextures,real/data_dict_train.pickle'
with open('./training/config/detector/ucf.yaml', 'r') as file:
    config = yaml.safe_load(file)
init_seed(config)
model_class = DETECTOR[config['model_name']]
model = model_class(config)  # Assuming your model is already loaded and on the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print('model done')
tsne_from_pickle(pickle_file_path, model,device)
'''


import os
import pandas as pd
import warnings
import pickle
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import random
from PIL import Image
import yaml
from tqdm import tqdm
from copy import deepcopy

# Environment configuration to reduce thread conflicts
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Configure PyTorch for single-threaded execution
torch.set_num_threads(1)

# Import custom modules after setting thread configurations
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from trainer.trainer import Trainer
from detectors import DETECTOR
from torchvision import transforms

# Initialize random seeds for reproducibility
def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])

# Function to load data from a pickle file
def load_pickle_data(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Function to load and preprocess images from file paths
def load_images_from_paths(image_paths, transform):
    images = []
    for path in image_paths:
        img = Image.open(path)
        img_tensor = transform(img)
        images.append(img_tensor)
    return torch.stack(images)

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Feature extraction using your model's `features` method
def extract_custom_features_from_pickle(model, data, device):
    model.eval()
    forgery_features_list = []
    labels_list = []

    with torch.no_grad():
        '''
        image_paths = data['image']
        labels = torch.tensor(data['label']).to(device)
        '''
        image_paths = data['Image Path'].tolist()
        labels = torch.tensor(data['Fake classification'].values).to(device)
        batch_size = 32
        num_batches = len(image_paths) // batch_size

        for i in tqdm(range(1500), desc="Processing Batches"):
            batch_image_paths = image_paths[i * batch_size: (i + 1) * batch_size]
            batch_labels = labels[i * batch_size: (i + 1) * batch_size]

            # Load and preprocess the images from file paths
            batch_images = load_images_from_paths(batch_image_paths, transform).to(device)

            data_dict = {'image': batch_images}
            feat_dict = model.features(data_dict)
            f_all = feat_dict['forgery']

            forgery_features_list.append(f_all.cpu().numpy())
            labels_list.append(batch_labels.cpu().numpy())
            torch.cuda.empty_cache()

        if len(image_paths) % batch_size != 0:
            batch_image_paths = image_paths[num_batches * batch_size:]
            batch_labels = labels[num_batches * batch_size:]

            batch_images = load_images_from_paths(batch_image_paths, transform).to(device)

            data_dict = {'image': batch_images}
            feat_dict = model.features(data_dict)
            f_all = feat_dict['forgery']
            forgery_features_list.append(f_all.cpu().numpy())
            labels_list.append(batch_labels.cpu().numpy())
            torch.cuda.empty_cache()
            print("batch_images shape:", batch_images.shape)
            print("f_all shape:", f_all.shape)
            print("batch_labels shape:", batch_labels.shape)

         

    # Concatenate features and labels
    all_forgery_features = np.concatenate(forgery_features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    return all_forgery_features, all_labels
 
def visual_analysis(features, labels):
    class_0_features = features[labels == 0]
    class_1_features = features[labels == 1]
    # Visual Analysis: Plot feature distributions for a specific feature dimension (e.g., the first feature)
    feature_index = 0  # Choose a feature dimension (index)
    plt.figure(figsize=(8, 6))
    sns.kdeplot(class_0_features[:, feature_index], label='Class 0 (Real)', color='blue')
    sns.kdeplot(class_1_features[:, feature_index], label='Class 1 (Fake)', color='red')
    plt.title('Feature Distribution Comparison')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('hist.png')
    plt.close()

# Apply t-SNE and visualize the results
def plot_tsne(features, labels, img_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(features)

    unique_labels = np.unique(labels)
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    color_mapping = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(10, 7))
    for label in unique_labels:
        plt.scatter(tsne_results[labels == label, 0], 
                    tsne_results[labels == label, 1], 
                    color=color_mapping[label], 
                    label=f'Class {label}', 
                    alpha=0.7)
    plt.title("t-SNE Visualization of Forgery Features")
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.savefig(img_path)
    plt.close()

# Load pickle data, extract features, and perform t-SNE
def tsne_from_pickle(csv_file_path, model, device, pretrained_weights_path=None):
    #data = load_pickle_data(pickle_file_path)
    data = pd.read_csv(csv_file_path)
    '''
    data_hair_0 = data[data['hair_color'] == 0]
    data_hair_1 = data[data['hair_color'] == 1]
    data_hair_2 = data[data['hair_color'] == 2]
    data_hair_3 = data[data['hair_color'] == 3]
    '''
    '''
    data_inter_4 = data[data['Intersection'] == 4]
    data_inter_5 = data[data['Intersection'] == 5]
    data_inter_6 = data[data['Intersection'] == 6]
    data_inter_7 = data[data['Intersection'] == 7]
    '''
    # Load pretrained weights if a path is provided
 
    if pretrained_weights_path is not None and os.path.exists(pretrained_weights_path):
        model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
        print(f"Loaded pretrained weights from {pretrained_weights_path}")
    else:
        print("No pretrained weights loaded.")
    
    img_path = 'tsne_visualization_all.png'
    forgery_features, labels = extract_custom_features_from_pickle(model, data, device)
    forgery_features = forgery_features.reshape(forgery_features.shape[0], -1)
    #forgery_features[np.isinf(forgery_features)] = np.nan
    #visual_analysis(forgery_features, labels)
    plot_tsne(forgery_features, labels, img_path)
    '''
    img_path = 'tsne_visualization_hair_1.png'
    forgery_features, labels = extract_custom_features_from_pickle(model, data_hair_1, device)
    forgery_features = forgery_features.reshape(forgery_features.shape[0], -1)
    #forgery_features[np.isinf(forgery_features)] = np.nan
    #visual_analysis(forgery_features, labels)
    plot_tsne(forgery_features, labels, img_path)

    img_path = 'tsne_visualization_hair_2.png'
    forgery_features, labels = extract_custom_features_from_pickle(model, data_hair_2, device)
    forgery_features = forgery_features.reshape(forgery_features.shape[0], -1)
    #forgery_features[np.isinf(forgery_features)] = np.nan
    #visual_analysis(forgery_features, labels)
    plot_tsne(forgery_features, labels, img_path)

    img_path = 'tsne_visualization_hair_3.png'
    forgery_features, labels = extract_custom_features_from_pickle(model, data_hair_3, device)
    forgery_features = forgery_features.reshape(forgery_features.shape[0], -1)
    #forgery_features[np.isinf(forgery_features)] = np.nan
    #visual_analysis(forgery_features, labels)
    plot_tsne(forgery_features, labels, img_path)
    '''
    '''
    img_path = 'tsne_visualization_age_3.png'
    forgery_features, labels = extract_custom_features_from_pickle(model, data_age_3, device)
    forgery_features = forgery_features.reshape(forgery_features.shape[0], -1)
    #forgery_features[np.isinf(forgery_features)] = np.nan
    #visual_analysis(forgery_features, labels)
    plot_tsne(forgery_features, labels, img_path)
    '''

    '''
    img_path = 'tsne_visualization_inter_4.png'
    forgery_features, labels = extract_custom_features_from_pickle(model, data_inter_4, device)
    forgery_features = forgery_features.reshape(forgery_features.shape[0], -1)
    #forgery_features[np.isinf(forgery_features)] = np.nan
    #visual_analysis(forgery_features, labels)
    plot_tsne(forgery_features, labels, img_path)

    img_path = 'tsne_visualization_inter_5.png'
    forgery_features, labels = extract_custom_features_from_pickle(model, data_inter_5, device)
    forgery_features = forgery_features.reshape(forgery_features.shape[0], -1)
    #forgery_features[np.isinf(forgery_features)] = np.nan
    #visual_analysis(forgery_features, labels)
    plot_tsne(forgery_features, labels, img_path)

    img_path = 'tsne_visualization_inter_6.png'
    forgery_features, labels = extract_custom_features_from_pickle(model, data_inter_6, device)
    forgery_features = forgery_features.reshape(forgery_features.shape[0], -1)
    #forgery_features[np.isinf(forgery_features)] = np.nan
    #visual_analysis(forgery_features, labels)
    plot_tsne(forgery_features, labels, img_path)

    img_path = 'tsne_visualization_inter_7.png'
    forgery_features, labels = extract_custom_features_from_pickle(model, data_inter_7, device)
    forgery_features = forgery_features.reshape(forgery_features.shape[0], -1)
    #forgery_features[np.isinf(forgery_features)] = np.nan
    #visual_analysis(forgery_features, labels)
    plot_tsne(forgery_features, labels, img_path)
    '''

# Example usage
#pickle_file_path = './log/ucf_2024-10-12-02-33-52/train/DeepFakes,Face2Face,FaceSwap,NeuralTextures,real/data_dict_train.pickle'
csv_file_path = './datasets/train_final.csv'
pretrained_weights_path = './training/weights/ucf_best.pth'  # specify the path to your pretrained weights
with open('./training/config/detector/ucf.yaml', 'r') as file:
    config = yaml.safe_load(file)
init_seed(config)
model_class = DETECTOR[config['model_name']]
model = model_class(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Pass the path to the pretrained weights
tsne_from_pickle(csv_file_path, model, device, pretrained_weights_path)
