import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Encode categorical features
    encoder = LabelEncoder()
    df["Type_Encoded"] = encoder.fit_transform(df["Type"])
    df["State_Encoded"] = encoder.fit_transform(df["State"])
    df["City_Encoded"] = encoder.fit_transform(df["City"])
    
    # Select relevant numerical features for similarity
    features = ["Type_Encoded", "Google review rating", "Entrance Fee in INR", "time needed to visit in hrs"]
    df_filtered = df[features]
    
    return df, df_filtered, encoder

def recommend_places(df, df_filtered, user_preferences):
    # Convert user input to match dataset features
    user_vector = np.array([user_preferences["Type_Encoded"],
                            user_preferences["Google review rating"],
                            user_preferences["Entrance Fee in INR"],
                            user_preferences["time needed to visit in hrs"]]).reshape(1, -1)
    
    # Compute similarity scores
    similarity_scores = cosine_similarity(user_vector, df_filtered)
    df["Similarity_Score"] = similarity_scores[0]
    
    # Get top 5 recommendations
    recommendations = df.sort_values(by="Similarity_Score", ascending=False).head(5)
    return recommendations[["Name", "State", "City", "Type", "Google review rating", "Similarity_Score"]]

# Load and preprocess data
df, df_filtered, encoder = load_and_preprocess_data(r"C:\Users\KIIT\Desktop\mini project\Cleaned_Indian_Travel_Dataset.csv")


# Get user input
type_input = input("Enter type of place (e.g., Beach, Adventure, Historical, etc.): ")
rating_input = float(input("Enter minimum Google review rating (e.g., 4.0): "))
fee_input = int(input("Enter maximum entrance fee in INR (e.g., 500): "))
time_input = float(input("Enter max time you want to spend in hours (e.g., 2.5): "))

# Convert user input
type_encoded = encoder.transform([type_input])[0] if type_input in encoder.classes_ else 0
user_preferences = {
    "Type_Encoded": type_encoded,
    "Google review rating": rating_input,
    "Entrance Fee in INR": fee_input,
    "time needed to visit in hrs": time_input
}

# Get recommendations
recommended_places = recommend_places(df, df_filtered, user_preferences)
print("\nTop Recommended Places:")
print(recommended_places)