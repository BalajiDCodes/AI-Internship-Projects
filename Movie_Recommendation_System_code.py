# Movie Recommendation System

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Movie dataset
movies = {
    'title': [
        'Avengers',
        'Iron Man',
        'Batman',
        'Superman',
        'Titanic',
        'The Notebook',
        'Doctor Strange',
        'Spider Man'
    ],
    'genre': [
        'Action Superhero',
        'Action Superhero',
        'Action Dark',
        'Action Hero',
        'Romance Drama',
        'Romance Love',
        'Action Magic',
        'Action Superhero'
    ]
}

# Create DataFrame
df = pd.DataFrame(movies)

# Convert genre text into numbers
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['genre'])

# Calculate similarity
similarity = cosine_similarity(count_matrix)

# Recommendation function
def recommend(movie_name):

    # Check if movie exists
    if movie_name not in df['title'].values:
        print("Movie not found!")
        return

    # Get movie index
    movie_index = df[df['title'] == movie_name].index[0]

    # Get similarity scores
    scores = list(enumerate(similarity[movie_index]))

    # Sort movies based on similarity
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nRecommended movies for", movie_name, ":\n")

    # Show top 3 recommendations
    count = 0

    for movie in sorted_scores[1:]:

        recommended_movie = df.iloc[movie[0]]['title']

        print(recommended_movie)

        count += 1

        if count == 3:
            break


# Ask user to enter movie name
movie_input = input("Enter movie name: ")

# Call function
recommend(movie_input)