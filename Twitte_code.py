import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Bigger dataset
data = {
    'tweet': [
        'I love this product',
        'This is amazing',
        'Very bad experience',
        'I hate this',
        'Awesome service',
        'Worst app ever',
        'Happy with the results',
        'Not good',
        'Excellent work',
        'Terrible support',
        'Fantastic experience',
        'I am disappointed',
        'Super happy',
        'Very poor quality',
        'Best purchase ever',
        'Waste of money',
        'Really satisfied',
        'I will not recommend this',
        'Absolutely wonderful',
        'Very frustrating',
        'I am very happy',
        'This service is excellent',
        'I enjoy using this app',
        'Bad customer support',
        'This is horrible',
        'Very nice experience'
    ],
    'sentiment': [
        'Positive',
        'Positive',
        'Negative',
        'Negative',
        'Positive',
        'Negative',
        'Positive',
        'Negative',
        'Positive',
        'Negative',
        'Positive',
        'Negative',
        'Positive',
        'Negative',
        'Positive',
        'Negative',
        'Positive',
        'Negative',
        'Positive',
        'Negative',
        'Positive',
        'Positive',
        'Positive',
        'Negative',
        'Negative',
        'Positive'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and labels
X = df['tweet']
y = df['sentiment']

# Convert words to numbers
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vectorized, y)

# Test sentence
custom_tweet = ["I am very happy with this service"]

# Transform input
custom_vector = vectorizer.transform(custom_tweet)

# Predict
prediction = model.predict(custom_vector)

# Print result
print("Tweet:", custom_tweet[0])
print("Predicted Sentiment:", prediction[0])