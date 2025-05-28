# Just a helper code that randomly samples 1000 files from the EdNet-KT1 dataset 
# & merges them into a single DataFrame. The sampled dataset is then saved as sampled_kt1_logs.csv

# Import necessary libraries
import os # For interacting with the operating system (e.g., file paths)
import pandas as pd # For working with structured data (DataFrames)
import random # For selecting files randomly
from pathlib import Path # For handling file paths in a platform-independent way

def sample_kt1_logs(folder_path: str, sample_size: int = 1000) -> pd.DataFrame:
    """
    Randomly sample user logs from the EdNet-KT1 directory and combine them into one DataFrame.

    Args:
        folder_path (str): Path to the folder containing KT1 user .csv logs.
        sample_size (int): Number of user logs to sample.

    Returns:
        pd.DataFrame: Combined DataFrame of sampled logs.
    """
    
    # Convert the folder path to a Path object for easy file handling
    path = Path(folder_path)
    # Find all CSV files that match the pattern 'u*.csv' (e.g., user logs)
    all_files = list(path.glob('u*.csv'))  
    
    # Check if the requested sample size exceeds available files
    if sample_size > len(all_files):
        # Raise an error if sample size is too large
        raise ValueError("Sample size exceeds available files")

    # Randomly select 'sample_size' number of files from the available logs
    sampled_files = random.sample(all_files, sample_size)
    
    # Initialize an empty list to store individual DataFrames
    dataframes = []  
    
    for file in sampled_files:
        # Read each sampled CSV file into a DataFrame
        df = pd.read_csv(file)
        # Extract the user ID from the filename (e.g., u123456) and add as a new column  
        df['user_id'] = file.stem
        # Append the processed DataFrame to the list  
        dataframes.append(df)  

    # Merge all sampled DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df  # Return the final combined DataFrame


# Sample KT1 dataset & save as 'sampled_kt1_logs.csv'
# Call the function to sample logs
sampled_df = sample_kt1_logs("./KT1", sample_size=1000)
# Save the sampled data to a CSV file without index  
sampled_df.to_csv("sampled_kt1_logs.csv", index=False)  