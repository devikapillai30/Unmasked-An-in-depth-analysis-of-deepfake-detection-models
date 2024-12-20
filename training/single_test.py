import pickle
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import yaml
from tqdm import tqdm
from copy import deepcopy
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from grad import gradcam 
from grad import load_images_from_paths
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from trainer.trainer import Trainer
from detectors import DETECTOR
import argparse
from torchvision import transforms

pretrained_weights_path = './training/weights/ucf_best.pth'

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])
        
# Load your trained model weights
with open('./training/config/detector/ucf.yaml', 'r') as file:
    config = yaml.safe_load(file)
init_seed(config)
model_class = DETECTOR[config['model_name']]
model = model_class(config)
model.load_state_dict(torch.load('./training/weights/ckpt_best_with_noise.pth', map_location=torch.device('cpu')))  # Replace with your weights path
model.eval()  # Set the model to evaluation mode

# Step 2: Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Replace with your input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust based on your dataset
])

# Step 3: Load and Predict on External Images
'''image_paths = [
    '/shared/rc/defake/FaceForensics++_All/FaceForensics++/real/youtube/test/906.mp4/0-188.jpeg',
'/shared/rc/defake/FaceForensics++_All/FaceForensics++/real/youtube/test/906.mp4/0-122.jpeg',
'/shared/rc/defake/FaceForensics++_All/FaceForensics++/real/youtube/test/906.mp4/0-275.jpeg',
'/shared/rc/defake/FaceForensics++_All/FaceForensics++/real/youtube/test/906.mp4/0-24.jpeg',
    '/shared/rc/defake/FaceForensics++_All/FaceForensics++/real/youtube/test/906.mp4/0-219.jpeg',
    '/shared/rc/defake/FaceForensics++_All/FaceForensics++/fake/FaceSwap/test/314_347.mp4/0-199.jpeg',
    '/shared/rc/defake/FaceForensics++_All/FaceForensics++/fake/NeuralTextures/test/924_917.mp4/0-359.jpeg',
'/shared/rc/defake/FaceForensics++_All/FaceForensics++/fake/NeuralTextures/test/924_917.mp4/0-85.jpeg',
'/shared/rc/defake/FaceForensics++_All/FaceForensics++/fake/NeuralTextures/test/924_917.mp4/0-369.jpeg',
'/shared/rc/defake/FaceForensics++_All/FaceForensics++/fake/NeuralTextures/test/924_917.mp4/0-257.jpeg'
]  # Replace with your image paths'''

# Load the CSV file (replace 'your_file.csv' with your actual file path)
csv_file = 'datasets/selected_image_paths_1_test.csv'

# Assuming the column containing image paths is named 'image_path'
df = pd.read_csv(csv_file, header=None)

# Extract the image paths into a list
image_paths = df[0].tolist()
print(len(image_paths))
# Create a list with 133 zeros
zeros = [0] * 178
# Create a list with 384 ones
ones = [1] * 339
# Combine both lists
result_list = zeros + ones
predictions1 = []
pt_weight = './training/weights/ucf_best.pth'
predictions2 = []
idx=0
for image_path in image_paths:
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add a batch dimension

    # Wrap the image tensor into a dictionary with key 'image'
    data_dict = {'image': image, 'label':None}

    with torch.no_grad():  # Disable gradient computation
        #print(image.shape)
        output = model(data_dict)  # Get model output (pass the dictionary instead of tensor)
        prediction = torch.softmax(output['cls'], dim=1)[:, 1]  # Apply sigmoid for binary classification
        #print(prediction)
        prediction2 = torch.sigmoid(output['feat'])  # Apply sigmoid for binary classification
        #print(prediction2)

    # Convert prediction to a class label
    #predicted_class = (prediction > 0.9).int()  # Adjust threshold as necessary
    #print(predicted_class)
    predicted_class1 = (prediction > 0.83).int().item()
    predicted_class2 = (prediction > 0.9).int().item()
    #print(predicted_class)
    predictions1.append(predicted_class1)
    predictions2.append(predicted_class2)
    img_list=[]
    img_list.append(image_path)
    batch_images = load_images_from_paths(img_list, transform)
    data_dict = {'image': batch_images}
 
    if (result_list[idx] == 0 and predicted_class1 == 1) or (result_list[idx] == 1 and predicted_class1 == 0):
        # FP or FN for threshold 0.5
        print(f"Running Grad-CAM for image: {image_path} (FP/FN at threshold 0.83)")
        heatmap, result_image = gradcam(data_dict, pt_weight)  # Call your grad_cam function here
        # Save or display the result
        result_image = np.uint8(result_image)  # Ensure the image is in uint8 format
        pil_image = Image.fromarray(result_image)
        pil_image.save(f'grad_cam_fp_fn_0.5_{os.path.basename(image_path)}.png')

    if (result_list[idx] == 0 and predicted_class2 == 1) or (result_list[idx] == 1 and predicted_class2 == 0):
        # FP or FN for threshold 0.7
        print(f"Running Grad-CAM for image: {image_path} (FP/FN at threshold 0.9)")
        heatmap, result_image = gradcam(data_dict, pt_weight)  # Call your grad_cam function here
        # Save or display the result
        result_image = np.uint8(result_image)  # Ensure the image is in uint8 format
        pil_image = Image.fromarray(result_image)
        pil_image.save(f'grad_cam_fp_fn_0.5_{os.path.basename(image_path)}.png')
    idx+=1
    #predictions.append(predicted_class.item())
    # Convert prediction to class label (0 or 1)
    #predicted_class = torch.argmax(prediction, dim=1).item()  # Get the class with highest probability
    #predictions.append(predicted_class)
#print(predictions)
    '''with torch.no_grad():  # Disable gradient computation
        output = model(data_dict)  # Get model output (pass the dictionary instead of tensor)
        prediction_prob = torch.softmax(output['cls'], dim=1)[:, 1]  # Get probability of the positive class
        probabilities.append(prediction_prob.item()) 

    
    #result = "deepfake" if predicted_class.item() == 1 else "real"

    #print(f"{image_path}: {result}")'''
# Compute and Print the Confusion Matrix


print("Threshold: 0.5")
cm = confusion_matrix(result_list, predictions1, labels=[0, 1])

print("Confusion Matrix:")
print(cm)
print(f"True Negatives (TN): {cm[0, 0]}")
print(f"False Positives (FP): {cm[0, 1]}")
print(f"False Negatives (FN): {cm[1, 0]}")
print(f"True Positives (TP): {cm[1, 1]}")

print("Threshold: 0.7")
cm = confusion_matrix(result_list, predictions2, labels=[0, 1])

print("Confusion Matrix:")
print(cm)
print(f"True Negatives (TN): {cm[0, 0]}")
print(f"False Positives (FP): {cm[0, 1]}")
print(f"False Negatives (FN): {cm[1, 0]}")
print(f"True Positives (TP): {cm[1, 1]}")

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(result_list, predictions1)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')

# Save the figure
plt.savefig('roc_curve1.png')  # Specify your desired file name and format
plt.close() 

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(result_list, predictions2)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')

# Save the figure
plt.savefig('roc_curve2.png')  # Specify your desired file name and format
plt.close() 