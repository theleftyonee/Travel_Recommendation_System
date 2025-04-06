import pandas as pd

# Load the dataset
file_path = "C:/Users/KIIT/Desktop/mini project/Cleaned_Indian_Travel_Dataset.csv"
df = pd.read_csv(file_path)

# Get unique place types
unique_types = df["Type"].unique()

# Print all available types
print("Available Place Types:")
for place_type in unique_types:
    print("-", place_type)
