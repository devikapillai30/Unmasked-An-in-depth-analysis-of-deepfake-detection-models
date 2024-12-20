import pandas as pd
import re
from multiprocessing import Pool
from tqdm import tqdm

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

# Create a set of lowercase check strings
check_strings_set = set(df2['check_string'].str.lower())

# Check if all components exist in any string from the set
def find_match(input_string):
    components = extract_components(input_string)
    if not components:
        return None  # No match if components can't be extracted
    
    # Create substrings to match
    substrings = [component for component in components]
    for check_string in check_strings_set:
        if all(substring in check_string for substring in substrings):
            return check_string  # Return the matching string
    return None

# Use multiprocessing for faster execution with tqdm progress bar
def process_row(row):
    return find_match(row[input_column])

if __name__ == '__main__':
    # Use a pool of workers to parallelize and integrate tqdm
    with Pool() as pool:
        # Create a progress bar
        results = list(
            tqdm(
                pool.imap(process_row, [row for _, row in df1.iterrows()]),
                total=len(df1),
                desc="Processing rows"
            )
        )
        # Assign results to the dataframe
        df1['FP'] = results

    # Save the updated CSV
    output_csv_path = 'updated_csv1.csv'
    df1.to_csv(output_csv_path, index=False)
    print(f"Updated CSV file saved to: {output_csv_path}")
