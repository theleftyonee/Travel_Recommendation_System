from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai

app = Flask(__name__)

# Load and preprocess data
def load_and_preprocess_data():
    df = pd.read_csv('Cleaned_Indian_Travel_Dataset.csv')
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    type_encoded = encoder.fit_transform(df[["Type"]])
    type_encoded_df = pd.DataFrame(type_encoded, columns=encoder.get_feature_names_out(["Type"]))
    df = pd.concat([df, type_encoded_df], axis=1)
    features = type_encoded_df.columns.tolist() + ["Google review rating", "Entrance Fee in INR", "time needed to visit in hrs"]
    df_filtered = df[features]
    return df, df_filtered, encoder

# Dataset-based recommendation function
def get_dataset_recommendations(user_preferences):
    df, df_filtered, encoder = load_and_preprocess_data()
    type_encoded = encoder.transform([[user_preferences["Type"]]])
    type_encoded_df = pd.DataFrame(type_encoded, columns=encoder.get_feature_names_out(["Type"]))
    
    user_vector = np.hstack([
        type_encoded_df.values.flatten(),
        [user_preferences["Google review rating"]],
        [user_preferences["Entrance Fee in INR"]],
        [user_preferences["time needed to visit in hrs"]]
    ]).reshape(1, -1)
    
    similarity_scores = cosine_similarity(user_vector, df_filtered)
    df["Similarity_Score"] = similarity_scores[0]
    recommendations = df.sort_values(by="Similarity_Score", ascending=False).head(5)
    return recommendations[["Name", "State", "City", "Type", "Google review rating", "Similarity_Score"]].to_dict('records')

# Configure Gemini API
genai.configure(api_key='AIzaSyDiimaJ3ZEcYlNUjZv43gHiIZahvQSG8vM')
model = genai.GenerativeModel('gemini-2.0-flash')

# AI-based recommendation function
def get_ai_recommendations(user_query):
    prompt = f"""
Act as an expert Indian travel advisor. Based on these preferences: {user_query}
Suggest 5 travel destinations in India. For each destination, create a well-formatted response using the following structure:

<div class="destination">
<h3><b>[Destination Name]</b></h3>
<p><i>Location: [State], [City]</i></p>
<p><strong>Type of Attraction:</strong> [e.g., Historical, Beach, etc.]</p>

<div class="details">
<h4>About the Destination:</h4>
<p>[Provide a compelling description of the destination's historical significance or natural beauty]</p>

<h4>Best Time to Visit:</h4>
<p>[Specify the ideal months/season and explain why]</p>

<h4>Key Attractions:</h4>
<ul>
[List 3-4 must-visit spots or experiences]
</ul>

<h4>Cultural Significance:</h4>
<p>[Explain the cultural importance and unique traditions]</p>

<h4>Unique Experiences:</h4>
<ul>
[List 2-3 unique activities or experiences visitors shouldn't miss]
</ul>
</div>
</div>
<hr>

Ensure each section is properly formatted with HTML tags and provides detailed, engaging information. Focus on creating an immersive description that captures the essence of each destination.
"""
    response = model.generate_content(prompt)
    formatted_text = response.text.replace('\n', '')
    return formatted_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dataset_form')
def dataset_form():
    return render_template('dataset_form.html')

@app.route('/ai_form')
def ai_form():
    return render_template('ai_form.html')

@app.route('/get_dataset_recommendations', methods=['POST'])
def dataset_recommendations():
    user_preferences = {
        "Type": request.form['type'],
        "Google review rating": float(request.form['rating']),
        "Entrance Fee in INR": int(request.form['fee']),
        "time needed to visit in hrs": float(request.form['time'])
    }
    recommendations = get_dataset_recommendations(user_preferences)
    return jsonify(recommendations)

@app.route('/get_ai_recommendations', methods=['POST'])
def ai_recommendations():
    user_query = request.form['query']
    recommendations = get_ai_recommendations(user_query)
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)