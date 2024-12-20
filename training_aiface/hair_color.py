import pandas as pd
import re

# Load the CSV files
main_df = pd.read_csv('defake/AIFace_Unmasked/AI-Face-FairnessBench/dataset/fake_train_small.csv')
ffpp_df = pd.read_csv('ff++.csv')

# Function to extract parts from `main.csv` paths
def extract_main_parts(path):
    match = re.search(r'/([^/]+)/.*?/(\d+_\d+)\.mp4/0-(\d+)\.jpeg', path)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None

# Function to extract parts from `ff++.csv` paths
def extract_ffpp_parts(path):
    match = re.search(r'/([^/]+)/.*/(\d+_\d+)/frame(\d+)\.png', path)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None

# Add extracted parts as new columns in both DataFrames
main_df[['group', 'video_id', 'frame']] = main_df['image_path'].apply(
    lambda x: pd.Series(extract_main_parts(x))
)
ffpp_df[['group', 'video_id', 'frame']] = ffpp_df['image_path'].apply(
    lambda x: pd.Series(extract_ffpp_parts(x))
)
merged_df = pd.merge(
    main_df,
    ffpp_df[['group', 'video_id', 'frame', 'grey_hair', 'black_hair', 'blonde_hair']],
    on=['group', 'video_id', 'frame'],
    how='left'  # Keep all rows from main_df and add matching data from ffpp_df
)

# Save the updated main_df back to a new CSV
merged_df.to_csv('main_with_hair_color.csv', index=False)