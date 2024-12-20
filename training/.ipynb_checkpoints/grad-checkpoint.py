import torch
import torch.nn.functional as F
import cv2
import os
import numpy as np
import yaml
from PIL import Image
from torchvision import transforms
from detectors import DETECTOR  # Ensure your UCFDetector model is correctly imported

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration file for the model
with open('./training/config/detector/ucf.yaml', 'r') as file:
    config = yaml.safe_load(file)
model_class = DETECTOR[config['model_name']]
model = model_class(config)

# Move model to the selected device
model.to(device)

# Load and set the model to evaluation mode
model.eval()

# Identify the target layer for Grad-CAM (the last conv layer before self.pool)
target_layer = model.encoder_f.conv4

# Hook functions to capture the features and gradients
features = []
gradients = []

def forward_hook(module, input, output):
    features.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

# Register hooks
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

def gradcam(input_image, pretrained_weights_path, class_idx=None):
    """
    Generate Grad-CAM heatmap for a given class index.
    
    Args:
    - input_image (torch.Tensor): Preprocessed input image tensor of shape (1, C, H, W).
    - class_idx (int, optional): Class index for visualization. Defaults to predicted class.

    Returns:
    - heatmap (np.array): Grad-CAM heatmap.
    - cam_on_image (np.array): Heatmap overlay on the original image.
    """
    if pretrained_weights_path is not None and os.path.exists(pretrained_weights_path):
        model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
        print(f"Loaded pretrained weights from {pretrained_weights_path}")
    else:
        print("No pretrained weights loaded.")
    
    # Forward pass
    output = model(input_image)
    output = output['cls']
    input_image = input_image['image']
    
    if class_idx is None:
        class_idx = torch.argmax(output, dim=1).item()

    # Backward pass to get gradients
    model.zero_grad()
    output[0, class_idx].backward()

    # Compute weights and Grad-CAM
    grads = gradients[0].cpu().data.numpy()
    feats = features[0].cpu().data.numpy()
    weights = np.mean(grads, axis=(2, 3))
    cam = np.sum(weights[:, :, None, None] * feats, axis=1).squeeze()

    # Apply ReLU and normalize
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    # Convert to heatmap and overlay on original image
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    input_image_np = input_image.squeeze().permute(1, 2, 0).cpu().numpy()
    input_image_np = (input_image_np - input_image_np.min()) / (input_image_np.max() - input_image_np.min())
    input_image_np = np.uint8(255 * input_image_np)
    cam_on_image = cv2.addWeighted(input_image_np, 0.6, heatmap, 0.4, 0)

    return heatmap, cam_on_image

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

# Example usage
pretrained_weights_path = './training/weights/ucf_best.pth'
input_image = ['/shared/rc/defake/FaceForensics++_All/FaceForensics++/real/youtube/test/015.mp4/0-101.jpeg']  # Replace with your preprocessed input
batch_images = load_images_from_paths(input_image, transform)

# Move the images to the correct device
batch_images = batch_images.to(device)

data_dict = {'image': batch_images}

# Generate Grad-CAM heatmap
heatmap, cam_on_image = gradcam(data_dict, pretrained_weights_path)

# Save or display
cv2.imwrite("gradcam_heatmap.jpg", heatmap)
cv2.imwrite("gradcam_on_image.jpg", cam_on_image)
