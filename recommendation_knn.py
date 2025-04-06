import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

# ✅ Load and preprocess dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv(r"C:\Users\KIIT\Desktop\mini project\Cleaned_Indian_Travel_Dataset.csv")


    # ✅ Encoding categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_types = encoder.fit_transform(df[["Type"]])
    
    # ✅ Scale numerical features (excluding Google review rating)
    scaler = StandardScaler()
    numerical_features = df[["Entrance Fee in INR", "time needed to visit in hrs"]]
    scaled_numerical = scaler.fit_transform(numerical_features)

    # ✅ Combine encoded categorical & scaled numerical data
    df_filtered = np.hstack((encoded_types, scaled_numerical))

    return df, df_filtered, encoder, scaler

# ✅ KNN-based recommendation function
def recommend_places(df, df_filtered, encoder, scaler, user_preferences):
    # ✅ Encode the 'Type' input from user
    type_encoded = encoder.transform([[user_preferences["Type"]]])  # One-hot encode user input

    # ✅ Scale numerical user preferences
    numeric_values = np.array([[user_preferences["Entrance Fee in INR"],
                                user_preferences["time needed to visit in hrs"]]])
    scaled_numeric = scaler.transform(numeric_values)

    # ✅ Combine categorical and numerical data
    user_vector = np.hstack((type_encoded, scaled_numeric))

    # ✅ Apply KNN model to find similar places
    knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    knn.fit(df_filtered)

    # ✅ Find the 5 nearest neighbors
    distances, indices = knn.kneighbors(user_vector)

    # ✅ Get recommended places
    recommendations = df.iloc[indices[0]]

    return recommendations[["Name", "State", "City", "Type"]]

# ✅ Load data
df, df_filtered, encoder, scaler = load_and_preprocess_data("C:/Users/KIIT/Desktop/mini project/Cleaned_Indian_Travel_Dataset.csv")

# ✅ Get user input
type_input = input("Enter type of place (e.g., Beach, Adventure, Historical, etc.): ")
fee_input = int(input("Enter maximum entrance fee in INR (e.g., 500): "))
time_input = float(input("Enter max time you want to spend in hours (e.g., 2.5): "))

# ✅ Ensure user input matches dataset categories
if type_input not in encoder.categories_[0]:
    print("Warning: Type not found, defaulting to a general category.")
    type_input = encoder.categories_[0][0]  # Set to first category as fallback

# ✅ Create user preference dictionary
user_preferences = {
    "Type": type_input,
    "Entrance Fee in INR": fee_input,
    "time needed to visit in hrs": time_input
}

# ✅ Get recommendations
recommended_places = recommend_places(df, df_filtered, encoder, scaler, user_preferences)

# ✅ Print recommendations
print("\nTop Recommended Places:")
print(recommended_places)
