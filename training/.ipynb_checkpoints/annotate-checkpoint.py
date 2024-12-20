import pandas as pd
from deepface import DeepFace
import csv
import cv2
import numpy as np
from PIL import Image

# Define the transformation function
def apply_transform(image):
    # Resize the image and normalize
    img = image.resize((256, 256))  # Resize to 256x256
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
    img_array = (img_array - 0.5) / 0.5  # Normalize to [-1, 1]
    return img_array  # Return as a numpy array

# Load the CSV with file paths
input_csv_path = 'datasets/selected_image_paths_1_test.csv'  # Replace with your actual input CSV path
output_csv_path = 'datasets/annotated_dataset.csv'  # Path to save the annotated CSV

# Read the CSV
df = pd.read_csv(input_csv_path, header=None)  # Assuming your CSV has a column named 'file_path'
file_paths = df[0].tolist()

# Prepare to write to a new CSV
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['File Path', 'Gender', 'Race', 'Age'])

    # Process each file path
    for file_path in file_paths:
        try:
            # Read the image using OpenCV
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            
            # Convert image to PIL format for applying transformations
            #image_pil = Image.fromarray(image_rgb)
            
            # Apply the custom transformation
            #transformed_image = apply_transform(image_pil)
            
            # Analyze the transformed image with DeepFace
            analysis = DeepFace.analyze(image_rgb, actions=['gender', 'race', 'age'], enforce_detection=False)
             
            # Handle cases where multiple faces might be detected
            if isinstance(analysis, list):
             analysis = analysis[0]  # Use the first face's analysis
             print(analysis[0])
            
            # Extract relevant attributes
            gender = analysis['gender']
            race = analysis['dominant_race']
            age = analysis['age']

            # Write the analysis to the CSV
            writer.writerow([file_path, gender, race, age])
         
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Optionally write errors or handle missing values
            writer.writerow([file_path, 'Error', 'Error', 'Error'])

print(f"Annotation complete. Results saved to {output_csv_path}")