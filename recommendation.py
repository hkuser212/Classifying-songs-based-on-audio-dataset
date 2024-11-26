import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset and trained model
data = pd.read_csv('features_3_sec.csv', index_col='filename')
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('xgb.model')  # Load the trained XGBoost model

# Scaling the features
scaler = MinMaxScaler()
X = data.drop(columns=['label'])  # Features (remove label column)
X_scaled = scaler.fit_transform(X)  # Scale the features
# Function to recommend genre and similar songs
def recommend_genre_and_songs(song_filename):
    try:
        # Get the features of the song to be recommended
        song_features = data.loc[song_filename].drop('label')
        song_scaled = scaler.transform([song_features])  # Scale the song's features

        # Predict the genre
        predicted_genre_index = xgb_model.predict(song_scaled)[0]

        # Map predicted genre index to genre name
        genre_mapping = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
                         5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}
        predicted_genre = genre_mapping.get(predicted_genre_index, 'Unknown')

        print(f"The predicted genre for '{song_filename}' is: {predicted_genre}")

        # Find similar songs based on cosine similarity
        song_index = data.index.get_loc(song_filename)
        song_vector = X_scaled[song_index].reshape(1, -1)  # Get the feature vector of the song

        # Compute cosine similarity between the given song and all other songs
        cosine_similarities = cosine_similarity(song_vector, X_scaled)

        # Get top 5 most similar songs (excluding the song itself)
        similar_indices = np.argsort(cosine_similarities[0])[::-1][1:6]
        similar_songs = data.index[similar_indices]

        print("\nTop 5 similar songs:")
        for i, similar_song in enumerate(similar_songs):
            print(f"{i + 1}. {similar_song}")

    except KeyError:
        print(f"Error: The song '{song_filename}' is not found in the dataset.")

# Example usage
song_filename = 'pop.00003.5.wav'  # Replace this with the actual song filename you want to recommend
recommend_genre_and_songs(song_filename)

