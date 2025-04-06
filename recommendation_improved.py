import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(r"C:\Users\KIIT\Desktop\mini project\Cleaned_Indian_Travel_Dataset.csv")


    # One-Hot Encode "Type" column
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    type_encoded = encoder.fit_transform(df[["Type"]])

    # Convert to DataFrame and merge with original data
    type_encoded_df = pd.DataFrame(type_encoded, columns=encoder.get_feature_names_out(["Type"]))
    df = pd.concat([df, type_encoded_df], axis=1)

    # Select relevant numerical features
    features = type_encoded_df.columns.tolist() + ["Google review rating", "Entrance Fee in INR", "time needed to visit in hrs"]
    df_filtered = df[features]

    return df, df_filtered, encoder

def recommend_places(df, df_filtered, user_preferences, encoder):
    # One-Hot Encode user input for "Type"
    type_encoded = encoder.transform([[user_preferences["Type"]]])
    type_encoded_df = pd.DataFrame(type_encoded, columns=encoder.get_feature_names_out(["Type"]))

    # Create user vector with correct format
    user_vector = np.hstack([
        type_encoded_df.values.flatten(),  # One-hot encoded type
        [user_preferences["Google review rating"]],
        [user_preferences["Entrance Fee in INR"]],
        [user_preferences["time needed to visit in hrs"]]
    ]).reshape(1, -1)

    # Compute similarity scores
    similarity_scores = cosine_similarity(user_vector, df_filtered)
    df["Similarity_Score"] = similarity_scores[0]

    # Get top 5 recommendations
    recommendations = df.sort_values(by="Similarity_Score", ascending=False).head(5)
    return recommendations[["Name", "State", "City", "Type", "Google review rating", "Similarity_Score"]]

# Load and preprocess data
df, df_filtered, encoder = load_and_preprocess_data("/mnt/data/Cleaned_Indian_Travel_Dataset.csv")

# Get user input
type_input = input("Enter type of place (e.g., Beach, Adventure, Historical, etc.): ").strip()
rating_input = float(input("Enter minimum Google review rating (e.g., 4.0): "))
fee_input = int(input("Enter maximum entrance fee in INR (e.g., 500): "))
time_input = float(input("Enter max time you want to spend in hours (e.g., 2.5): "))

# Store user preferences
user_preferences = {
    "Type": type_input,
    "Google review rating": rating_input,
    "Entrance Fee in INR": fee_input,
    "time needed to visit in hrs": time_input
}

# Get recommendations
recommended_places = recommend_places(df, df_filtered, user_preferences, encoder)
print("\nTop Recommended Places:")
print(recommended_places)
