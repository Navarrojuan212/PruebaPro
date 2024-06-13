import pandas as pd
import numpy as np
import os
from scripts.violinplot import create_violin_plot
from scripts.histogram import create_histogram
from scripts.scatterplot import create_scatterplot
from scripts.statistics import calculate_statistics
from scripts.data_analysis import analyze_data  # Import the new analysis function


# Define the base path for data files
base_path = os.path.join(os.getcwd(), 'Data')

# Define the path for the images folder
images_path = os.path.join(os.getcwd(), 'Images')
os.makedirs(images_path, exist_ok=True)

# Define the file names
file_names = [
    'Piso1_103_5MHz_SNR.xlsx',
    'Piso2_103_5MHz_SNR.xlsx',
    'Piso3_103_5MHz_SNR.xlsx',
    'Piso4_103_5MHz_SNR.xlsx',
    'Piso5_103_5MHz_SNR.xlsx',
    'Piso6_103_5MHz_SNR.xlsx',
    'Piso6_Outdoor_103_5MHz_SNR.xlsx',
    'Sotano1_103_5MHz_SNR_B.csv',
    'Sotano2_103_5MHz_SNR_E.csv'
]

# Function to translate the file name to a more user-friendly name
def translate_name(file_name):
    mapping = {
        'Piso1_103_5MHz_SNR.xlsx': 'Floor 1',
        'Piso2_103_5MHz_SNR.xlsx': 'Floor 2',
        'Piso3_103_5MHz_SNR.xlsx': 'Floor 3',
        'Piso4_103_5MHz_SNR.xlsx': 'Floor 4',
        'Piso5_103_5MHz_SNR.xlsx': 'Floor 5',
        'Piso6_103_5MHz_SNR.xlsx': 'Floor 6',
        'Piso6_Outdoor_103_5MHz_SNR.xlsx': 'Outdoor Floor 6',
        'Sotano1_103_5MHz_SNR_B.csv': 'Basement 1',
        'Sotano2_103_5MHz_SNR_E.csv': 'Basement 2'
    }
    return mapping.get(file_name, "Unknown Source")  # Returns 'Unknown Source' if the file is not found

# Function to read CSV or Excel files
def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path, engine='openpyxl')
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

# Load the data and adjust the source names
data_frames = {}
for file_name in file_names:
    file_path = os.path.join(base_path, file_name)
    try:
        df = load_data(file_path)
        df['Source'] = translate_name(file_name)  # Use the translation function
        df['Time_Index'] = np.arange(0, len(df) * 0.1, 0.1)
        data_frames[file_name] = df
        print(f"Loaded {file_name} with {len(df)} rows")
    except Exception as e:
        print(f"Error processing the file {file_name}: {e}")

# Verify the size of each DataFrame
for file_name, df in data_frames.items():
    print(f"{translate_name(file_name)}: {len(df)} rows")  # Use the translation here as well

# Combine all DataFrames into one
combined_df = pd.DataFrame()
for file_name, df in data_frames.items():
    combined_df = pd.concat([combined_df, df])

# Create the violin plot and save it in the Images folder
create_violin_plot(combined_df, images_path)

# Create the histogram and save it in the Images folder
create_histogram(data_frames, images_path, translate_name)

# Create the scatter plot and save it in the Images folder
create_scatterplot(data_frames, images_path, translate_name)

# Create a new DataFrame to store the statistics
statistics_df = pd.DataFrame()

# Calculate the statistics for each DataFrame and add them to the new DataFrame
for file_name, df in data_frames.items():
    df['Source'] = translate_name(file_name)  # Use the translated name here
    stats_df = calculate_statistics(df)
    statistics_df = pd.concat([statistics_df, stats_df], ignore_index=True)

# Display the new DataFrame with the statistics
print(statistics_df)

# Call the data analysis function and pass the combined dataframe and images path
analyze_data(combined_df, images_path)