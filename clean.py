import pandas as pd

# 🔹 Corrected File Path
file_path = r"C:\Users\KIIT\Desktop\mini project\Top Indian Places to Visit.csv"

# 🔹 Load the Dataset
df = pd.read_csv(file_path)

# 🔹 Print column names to check correct names
print("Columns in dataset:", df.columns)

# 🔹 Fill Missing Values (Fixed 'Rating' issue)
df.fillna({
    "State": "Unknown", 
    "Category": "General", 
    "Google review rating": df["Google review rating"].mean()  # Corrected column name
}, inplace=True)

# 🔹 Save Cleaned Data
df.to_csv("Cleaned_Indian_Travel_Dataset.csv", index=False)

print("✅ Data Cleaning Completed!")
