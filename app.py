# Import necessary libraries
import os # Added for environment variable access
from dotenv import load_dotenv # Added to load environment variables from .env

# Load environment variables from .env file at the very start of the script
load_dotenv()

from flask import Flask, render_template, request, jsonify, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import requests
from flask_cors import CORS

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# --- TMDB API Configuration ---
# IMPORTANT: The TMDB API Key is now loaded from your .env file
TMDB_API_KEY = os.getenv("TMDB_API_KEY") # Reads the key from the environment
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500" # w500 is a common size for posters

# --- Global variables for the recommendation system ---
movies_df = None
tfidf_vectorizer = None
cosine_sim_matrix = None
all_movies_data = [] # Stores movie data including TMDB image URLs and new details

def get_movie_details_from_tmdb(movie_title):
    """
    Fetches detailed information for a given movie title from TMDB,
    including poster, overview, release date, vote average, director, and cast.
    Returns a dictionary of movie details or None if not found/error.
    """
    # Check if the API key is set before making a request
    if not TMDB_API_KEY:
        print("TMDB API Key is not set in environment variables. Cannot fetch movie details from TMDB.")
        return None

    # Step 1: Search for the movie to get its ID
    search_url = f"{TMDB_BASE_URL}/search/movie"
    search_params = {
        "api_key": TMDB_API_KEY,
        "query": movie_title
    }
    try:
        search_response = requests.get(search_url, params=search_params)
        search_response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        search_data = search_response.json()

        if not search_data or not search_data['results']:
            print(f"No search results found for '{movie_title}'.")
            return None

        movie_id = search_data['results'][0]['id']
        
        # Step 2: Get detailed movie info using the ID
        details_url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        credits_url = f"{TMDB_BASE_URL}/movie/{movie_id}/credits" # To get cast and director
        
        details_params = {"api_key": TMDB_API_KEY}
        
        details_response = requests.get(details_url, params=details_params)
        details_response.raise_for_status()
        details = details_response.json()

        credits_response = requests.get(credits_url, params=details_params)
        credits_response.raise_for_status()
        credits = credits_response.json()

        # Extract director and main cast
        director = "N/A"
        cast = []
        for crew_member in credits.get('crew', []):
            if crew_member.get('job') == 'Director':
                director = crew_member.get('name')
                break
        
        for cast_member in credits.get('cast', []):
            cast.append(cast_member.get('name'))
            if len(cast) >= 5: # Limit to top 5 cast members
                break

        return {
            "title": details.get('title'),
            "overview": details.get('overview'),
            "release_date": details.get('release_date'),
            "vote_average": details.get('vote_average'),
            "poster_path": f"{TMDB_IMAGE_BASE_URL}{details.get('poster_path')}" if details.get('poster_path') else None,
            "director": director,
            "cast": ", ".join(cast) if cast else "N/A"
        }

    except requests.exceptions.RequestException as e:
        print(f"Error fetching details for '{movie_title}' from TMDB: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response for '{movie_title}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred for '{movie_title}': {e}")
    
    return None # Return None if any error occurs

def load_and_process_data():
    """
    Loads movie data, preprocesses it, initializes TF-IDF Vectorizer,
    computes the cosine similarity matrix, and fetches TMDB poster URLs and details.
    This function is called once when the Flask app starts.
    """
    global movies_df, tfidf_vectorizer, cosine_sim_matrix, all_movies_data

    # Expanded hardcoded DataFrame for demonstration of more movies.
    # In a real application, you'd load from a CSV, database, or API.
    data = {
        'title': [
            'The Dark Knight', 'Inception', 'Interstellar', 'Pulp Fiction',
            'Forrest Gump', 'The Matrix', 'Avatar', 'Titanic',
            'Spirited Away', 'Princess Mononoke', 'Eternal Sunshine of the Spotless Mind',
            'Arrival', 'Blade Runner 2049', 'Parasite', 'Whiplash',
            'Fight Club', 'Goodfellas', 'Seven', 'Django Unchained', 'Inglourious Basterds',
            'Home Alone', 'Home Alone 2: Lost in New York', 'Toy Story', 'The Lion King',
            'Finding Nemo', 'Shrek', 'Harry Potter and the Sorcerer\'s Stone',
            'The Lord of the Rings: The Fellowship of the Ring', 'Star Wars: A New Hope',
            'E.T. the Extra-Terrestrial', 'Jurassic Park', 'Back to the Future',
            'Ghostbusters', 'Indiana Jones and the Raiders of the Lost Ark',
            'The Shawshank Redemption', 'The Godfather', 'Schindler\'s List',
            'The Green Mile', 'Catch Me If You Can', 'Saving Private Ryan', 'Gladiator',
            'The Departed', 'Good Will Hunting', 'American History X', 'Leon: The Professional',
            'Terminator 2: Judgment Day', 'Alien', 'Aliens', 'Blade Runner',
            'Apocalypse Now', 'Full Metal Jacket', 'The Shining', '2001: A Space Odyssey',
            'A Clockwork Orange', 'Dr. Strangelove', 'Paths of Glory', 'Rear Window',
            'Psycho', 'Vertigo', 'North by Northwest', 'Sunset Boulevard',
            'Some Like It Hot', 'The Apartment', 'Amelie', 'City of God',
            'Oldboy', 'Pan\'s Labyrinth', 'The Secret in Their Eyes', 'Incendies',
            'Her', 'Ex Machina', 'Mad Max: Fury Road', 'Dunkirk',
            'Joker', '1917', 'Knives Out', 'Once Upon a Time in Hollywood',
            'Ford v Ferrari', 'Little Women', 'Marriage Story', 'The Irishman',
            'Jojo Rabbit', 'Parasite', 'Nomadland', 'Minari',
            'Sound of Metal', 'Promising Young Woman', 'Judas and the Black Messiah',
            'The Father', 'Mank', 'Trial of the Chicago 7', 'Soul',
            'Wolfwalkers', 'Over the Moon', 'Tenet', 'Wonder Woman 1984',
            'Mulan', 'Raya and the Last Dragon', 'Luca', 'Encanto',
            'Dune', 'No Time to Die', 'Spider-Man: No Way Home', 'The Power of the Dog',
            'West Side Story', 'Don\'t Look Up', 'Licorice Pizza', 'Belfast',
            'CODA', 'King Richard', 'Drive My Car', 'Flee',
            'The Worst Person in the World', 'Parallel Mothers', 'Titane', 'The Hand of God'
        ],
        'genres': [
            'Action, Crime, Drama', 'Action, Sci-Fi, Thriller', 'Adventure, Drama, Sci-Fi',
            'Crime, Drama', 'Drama, Romance', 'Action, Sci-Fi',
            'Action, Adventure, Fantasy', 'Drama, Romance',
            'Animation, Adventure, Family', 'Animation, Adventure, Fantasy',
            'Drama, Romance, Sci-Fi', 'Drama, Sci-Fi',
            'Sci-Fi, Thriller', 'Comedy, Drama, Thriller', 'Drama, Music',
            'Drama', 'Crime, Drama', 'Crime, Drama, Mystery', 'Drama, Western', 'Adventure, Drama, War',
            'Comedy, Family', 'Comedy, Family', 'Animation, Adventure, Comedy', 'Animation, Adventure, Drama',
            'Animation, Adventure, Family', 'Animation, Adventure, Comedy', 'Adventure, Family, Fantasy',
            'Adventure, Drama, Fantasy', 'Action, Adventure, Sci-Fi',
            'Family, Sci-Fi', 'Adventure, Sci-Fi, Thriller', 'Adventure, Comedy, Sci-Fi',
            'Comedy, Fantasy', 'Action, Adventure',
            'Drama', 'Crime, Drama', 'Biography, Drama, History',
            'Crime, Drama, Fantasy', 'Biography, Crime, Drama', 'Drama, War', 'Action, Adventure, Drama',
            'Crime, Drama, Thriller', 'Drama, Romance', 'Crime, Drama', 'Action, Crime, Drama',
            'Action, Sci-Fi', 'Horror, Sci-Fi', 'Action, Sci-Fi', 'Sci-Fi, Thriller',
            'Drama, War', 'Drama, War', 'Horror', 'Sci-Fi, Drama',
            'Sci-Fi, Drama', 'Comedy, War', 'Drama, War', 'Mystery, Thriller',
            'Horror, Mystery', 'Mystery, Romance', 'Action, Adventure, Thriller', 'Drama, Film-Noir',
            'Comedy, Romance', 'Comedy, Drama, Romance', 'Comedy, Drama, Romance', 'Crime, Drama',
            'Action, Crime, Drama', 'Drama, Fantasy', 'Drama, Mystery, Thriller', 'Drama, War',
            'Drama, Romance, Sci-Fi', 'Drama, Sci-Fi', 'Action, Adventure, Sci-Fi', 'Drama, War',
            'Crime, Drama, Thriller', 'Drama, War', 'Comedy, Crime, Drama', 'Comedy, Drama',
            'Biography, Drama, Sport', 'Drama, Romance', 'Comedy, Drama, Romance', 'Crime, Drama',
            'Comedy, Drama, War', 'Comedy, Drama, Thriller', 'Drama', 'Drama',
            'Drama, Music', 'Crime, Drama, Thriller', 'Biography, Drama, History',
            'Drama', 'Biography, Drama', 'Drama, History', 'Animation, Adventure, Comedy',
            'Animation, Adventure, Family', 'Action, Sci-Fi, Thriller', 'Action, Adventure, Fantasy',
            'Animation, Action, Adventure', 'Animation, Adventure, Family', 'Animation, Adventure, Family', 'Animation, Adventure, Family',
            'Action, Adventure, Sci-Fi', 'Action, Adventure, Thriller', 'Action, Adventure, Sci-Fi', 'Drama, Western',
            'Drama, Musical, Romance', 'Comedy, Drama, Sci-Fi', 'Comedy, Drama, Romance', 'Drama, History',
            'Drama, Music', 'Biography, Drama, Sport', 'Drama', 'Animation, Biography, Drama',
            'Comedy, Drama, Romance', 'Drama', 'Drama, Horror', 'Drama', 'Drama'
        ]
    }
    movies_df = pd.DataFrame(data)

    # Fetch TMDB poster URLs and other details for each movie
    temp_all_movies_data = []
    for index, row in movies_df.iterrows():
        movie_details = get_movie_details_from_tmdb(row['title'])
        if movie_details:
            # Merge original data with TMDB details
            movie_data = {
                "title": row['title'],
                "genres": row['genres'],
                "image_url": movie_details.get('poster_path', '/static/placeholder.png'),
                "overview": movie_details.get('overview', 'No overview available.'),
                "release_date": movie_details.get('release_date', 'N/A'),
                "vote_average": movie_details.get('vote_average', 'N/A'),
                "director": movie_details.get('director', 'N/A'),
                "cast": movie_details.get('cast', 'N/A')
            }
            temp_all_movies_data.append(movie_data)
        else:
            # Fallback to local placeholder and minimal data if TMDB fetch fails
            temp_all_movies_data.append({
                "title": row['title'],
                "genres": row['genres'],
                "image_url": '/static/placeholder.png',
                "overview": 'No overview available.',
                "release_date": 'N/A',
                "vote_average": 'N/A',
                "director": 'N/A',
                "cast": 'N/A'
            })
    
    all_movies_data = temp_all_movies_data
    movies_df = pd.DataFrame(all_movies_data) # Update movies_df with full data

    # Convert genres into a TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])

    # Compute the cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    print("Data loaded and similarity matrix computed successfully.")


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

    # Exclude the movie itself from recommendations
    sim_scores = [score for score in sim_scores if score[0] != idx]

    recommended_movie_indices = [i[0] for i in sim_scores[:num_recommendations]]

    # Return full movie data for recommendations, including image_url
    recommended_movies = []
    for rec_idx in recommended_movie_indices:
        recommended_movies.append({
            "title": df['title'].iloc[rec_idx],
            "image_url": df['image_url'].iloc[rec_idx]
        })
    return recommended_movies

# --- Flask Routes ---

@app.route('/')
def index():
    """
    Renders the main HTML page for the recommendation system.
    Passes the Firebase config to the template.
    """
    # These are placeholders for the backend. Frontend uses its own hardcoded config.
    firebase_config = {
        "apiKey": "",
        "authDomain": "",
        "projectId": "",
        "storageBucket": "",
        "messagingSenderId": "",
        "appId": "",
        "measurementId": ""
    }
    firebase_config_json = json.dumps(firebase_config)

    return render_template('index.html', firebase_config=firebase_config_json)

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    API endpoint to get movie recommendations.
    Expects a JSON payload with 'movie_title'.
    Returns a JSON response with 'recommendations' (including image URLs).
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
    API endpoint to get all movie titles and their genres, with pagination support.
    Now returns full movie details for the grid.
    """
    # Ensure data is loaded if not already
    if movies_df is None:
        with app.app_context():
            load_and_process_data()

    offset = request.args.get('offset', default=0, type=int)
    limit = request.args.get('limit', default=20, type=int)

    # Return full movie data for pagination
    paginated_movies = all_movies_data[offset:offset + limit]
    
    has_more = (offset + limit) < len(all_movies_data)

    return jsonify({'movies': paginated_movies, 'has_more': has_more})

@app.route('/movie_details/<path:movie_title>', methods=['GET'])
def movie_details(movie_title):
    """
    New API endpoint to get comprehensive details for a single movie.
    Used for the modal display.
    """
    # Ensure data is loaded
    if movies_df is None:
        with app.app_context():
            load_and_process_data()

    # Find the movie in our loaded data
    movie_data = next((movie for movie in all_movies_data if movie['title'] == movie_title), None)

    if movie_data:
        return jsonify(movie_data)
    else:
        return jsonify({'error': 'Movie not found'}), 404


@app.route('/personalized_recommendations', methods=['POST'])
def personalized_recommendations():
    """
    New API endpoint to get personalized recommendations based on user's rated movies.
    Expects a JSON payload with 'rated_movies': [{'title': 'Movie A', 'rating': 5}, ...].
    """
    data = request.get_json()
    rated_movies = data.get('rated_movies', [])

    if not rated_movies:
        return jsonify({'recommendations': []})

    # Simple logic: Take the top 3 highest-rated movies and get recommendations for each.
    # You can refine this by averaging scores, using a more complex model, etc.
    sorted_rated_movies = sorted(rated_movies, key=lambda x: x['rating'], reverse=True)
    
    all_personal_recs = []
    processed_rec_titles = set() # To avoid duplicate recommendations

    # Get recommendations from the top-rated movies
    for movie_info in sorted_rated_movies[:3]: # Consider top 3 rated movies
        recs_for_this_movie = get_recommendations(movie_info['title'], cosine_sim_matrix, movies_df, num_recommendations=5)
        for rec in recs_for_this_movie:
            if rec['title'] not in processed_rec_titles and rec['title'] != movie_info['title']:
                all_personal_recs.append(rec)
                processed_rec_titles.add(rec['title'])
    
    # Limit total personalized recommendations to a reasonable number, e.g., 10-15
    final_personal_recs = all_personal_recs[:15]

    return jsonify({'recommendations': final_personal_recs})


# --- Main execution block ---
if __name__ == '__main__':
    # Load and process data when the application starts
    with app.app_context():
        load_and_process_data()
    # Run the Flask application in debug mode
    app.run(debug=True)
