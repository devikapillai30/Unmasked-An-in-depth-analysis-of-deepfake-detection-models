import pandas as pd
import re

# Load the first CSV (containing input strings)
csv1_path = 'dataset/filtered_images_with_classification.csv'  # Replace with the path to your first CSV
df1 = pd.read_csv(csv1_path)

# Load the second CSV (containing strings to check against, no header)
csv2_path = 'dataset/all_image_paths_tvt.csv'  # Replace with the path to your second CSV
df2 = pd.read_csv(csv2_path, header=None, names=['check_string'])

# Assuming the relevant column in CSV1 is named 'input_string'
input_column = 'Image Path'  # Replace with the column name in CSV1

# Extract components from the input strings in CSV1
def extract_components(image_path):
    pattern = r".*/\d+_(\w+)_([\d_]+)_(\d+)\.png$"
    match = re.match(pattern, image_path)
    if match:
        print(match.groups())
        return match.groups()  # Returns a tuple: (NeuralTextures, 455_471, 151)
    return None

# Check if all components exist in any string from CSV2 (case-insensitive)
def find_match(input_string, check_strings):
    components = extract_components(input_string)
    if not components:
        return None  # No match if components can't be extracted
    
    # Convert components and check strings to lowercase for case-insensitive matching
    components_lower = [component.lower() for component in components]
    
    for check_string in check_strings:
        check_string_lower = check_string.lower()  # Convert to lowercase
        if all(component in check_string_lower for component in components_lower):
            print(check_string)
            return check_string  # Return the matching string
    return None

# Apply the matching logic
check_strings_list = df2['check_string'].tolist()
df1['FP'] = df1[input_column].apply(lambda x: find_match(x, check_strings_list))

# Save the updated first CSV
output_csv_path = 'updated_csv1.csv'
df1.to_csv(output_csv_path, index=False)

print(f"Updated CSV file saved to: {output_csv_path}")
