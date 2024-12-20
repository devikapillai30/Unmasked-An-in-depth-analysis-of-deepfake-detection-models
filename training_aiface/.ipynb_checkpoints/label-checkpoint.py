import csv

# Input and output file paths
input_csv = 'dataset/test.csv'  # Replace with the path to your input CSV
output_csv = 'filtered_images_with_classification_test.csv'  # Output file

# Mapping for classification labels
classification_mapping = {
    'real': 0,
    'Deepfakes': 1,
    'Face2Face': 2,
    'FaceSwap': 3,
    'NeuralTextures': 4
}

# Function to classify image paths
def classify_image_path(image_path):
    for key, value in classification_mapping.items():
        if key in image_path:
            return value
    return None  # Return None if no matching classification found

# Process the input CSV
with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Read and write the header row
    header = next(reader)
    writer.writerow(header + ['Fake classification'])  # Add new column to header
    
    # Process each row
    for row in reader:
        image_path = row[0]  # First column is the image path
        if 'ff++' in image_path:  # Check if "FF++" is in the image path
            classification = classify_image_path(image_path)
            if classification is not None:
                writer.writerow(row + [classification])  # Append classification

print(f"Filtered and classified data saved to: {output_csv}")
