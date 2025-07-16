# Import necessary libraries
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from flask_cors import CORS # NEW: Import CORS

# Initialize the Flask application
app = Flask(__name__)
CORS(app) # NEW: Enable CORS for your Flask app. This allows cross-origin requests.

# --- Global variables for the recommendation system ---
movies_df = None
tfidf_vectorizer = None
cosine_sim_matrix = None
movie_titles = []

def load_and_process_data():
    """
    Loads movie data from movies.csv, preprocesses it,
    initializes TF-IDF Vectorizer, and computes the cosine similarity matrix.
    This function is called once when the Flask app starts.
    """
    global movies_df, tfidf_vectorizer, cosine_sim_matrix, movie_titles

    movies_file_path = os.path.join(os.path.dirname(__file__), 'movies.csv')

    try:
        movies_df = pd.read_csv(movies_file_path, usecols=['movieId', 'title', 'genres'])
        print(f"Loaded {len(movies_df)} movies from movies.csv")

        movies_df['genres'] = movies_df['genres'].fillna('')
        movies_df['combined_features'] = movies_df['genres']

        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['combined_features'])

        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        movie_titles = sorted(movies_df['title'].unique().tolist())

        print("Data loaded and similarity matrix computed successfully.")

    except FileNotFoundError:
        print(f"Error: movies.csv not found at {movies_file_path}. Please ensure the file is in the same directory as app.py")
        exit()
    except Exception as e:
        print(f"An error occurred during data loading or processing: {e}")
        exit()


def get_recommendations(movie_title, cosine_sim=None, df=None, num_recommendations=10):
    """
    Generates movie recommendations based on the similarity of their combined features.
    """
    if cosine_sim is None:
        cosine_sim = cosine_sim_matrix
    if df is None:
        df = movies_df

    if df is None or cosine_sim is None:
        print("Error: Data or similarity matrix not loaded.")
        return []

    movie_indices = df[df['title'] == movie_title].index.tolist()

    if not movie_indices:
        print(f"Movie '{movie_title}' not found in the database.")
        return []

    idx = movie_indices[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]

    recommended_movie_indices = [i[0] for i in sim_scores]

    return df['title'].iloc[recommended_movie_indices].drop_duplicates().tolist()

# --- Flask Routes ---

@app.route('/')
def index():
    """
    Renders the main HTML page for the recommendation system.
    Passes the list of available movie titles to the template.
    """
    if movies_df is None:
        load_and_process_data()
    return render_template('index.html', movie_titles=movie_titles)

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    API endpoint to get movie recommendations.
    Expects a JSON payload with 'movie_title'.
    Returns a JSON response with 'recommendations'.
    """
    data = request.get_json()
    movie_title = data.get('movie_title')

    if not movie_title:
        return jsonify({'error': 'No movie title provided'}), 400

    recommendations = get_recommendations(movie_title, cosine_sim_matrix, movies_df)

    return jsonify({'recommendations': recommendations})

@app.route('/get_all_movies', methods=['GET'])
def get_all_movies():
    """
    API endpoint to get all movie titles and their genres.
    Used by the frontend for search suggestions.
    """
    if movies_df is None:
        load_and_process_data()

    return jsonify({'movies': movies_df[['title', 'genres']].to_dict(orient='records')})

# --- Main execution block ---
if __name__ == '__main__':
    load_and_process_data()
    app.run(debug=True)
