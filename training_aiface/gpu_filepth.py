import pandas as pd
import re
from tqdm import tqdm

# Try importing CuPy for GPU support, fallback to NumPy if unavailable
try:
    import cupy as xp
    xp_available = xp.is_available()
    print("CuPy is available. Running on GPU.")
except ImportError:
    import numpy as xp  # Fallback to NumPy
    xp_available = False
    print("CuPy is not available. Running on CPU.")

# Load the CSV files
csv1_path = 'dataset/filtered_images_with_classification.csv'
csv2_path = 'dataset/all_image_paths_tvt.csv'
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path, header=None, names=['check_string'])

input_column = 'Image Path'

# Compile the regex pattern once
pattern = re.compile(r".*/\d+_(\w+)_([\d]+_[\d]+)_(\d+)\.png$")

# Extract components from the input strings
def extract_components(image_path):
    match = pattern.match(image_path)
    if match:
        return tuple(map(str.lower, match.groups()))  # Convert to lowercase
    return None

# Move the check strings to the processing device (GPU or CPU)
check_strings = xp.array(df2['check_string'].str.lower().values)

# Check if all components exist in any string from the set
def find_match(input_string):
    components = extract_components(input_string)
    if not components:
        return None  # No match if components can't be extracted

    substrings = xp.array(components)
    for check_string in check_strings:
        # Substring existence check
        if xp.all(xp.array([substring in check_string for substring in substrings])):
            return check_string
    return None

# Process rows and add progress bar
def process_row(row):
    return find_match(row[input_column])

if __name__ == '__main__':
    results = []

    # Process each row with a progress bar
    for row in tqdm(df1[input_column], desc="Processing rows"):
        results.append(process_row({'Image Path': row}))

    # Convert results to CPU-compatible format if needed
    if xp_available:
        results = xp.asnumpy(xp.array(results))

    # Assign results back to the dataframe
    df1['FP'] = results

    # Save the updated CSV
    output_csv_path = 'updated_csv1.csv'
    df1.to_csv(output_csv_path, index=False)
    print(f"Updated CSV file saved to: {output_csv_path}")
