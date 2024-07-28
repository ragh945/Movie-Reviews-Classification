import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image

# Correct file paths for images
inno = Image.open("Inno.png")
st.image(inno, use_column_width=True)
st.title("South Indian Movie Reviews Classification")
Telugu_logo = Image.open("Telugu_movies.jpeg")
st.image(Telugu_logo, caption='Email', use_column_width=True)

# Load the pre-trained model and vectorizer
model_path = "South Telugu Movies Reviews.pkl"
vectorizer_path = "bow_vectorization.pkl"

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    bow = pickle.load(vectorizer_file)

# Input review from user
comment = st.text_input("Enter your review: ")

# Transform the input review using the loaded vectorizer
if st.button("Submit"):
    data = bow.transform([comment]).toarray()
    pred = model.predict(data)[0]
    
    if pred == 'negative':
        st.write("Boring")
        neg_r = Image.open("boring.jpg")
        st.image(neg_r, use_column_width=False)
        
        # List of top 5 boring movies and their image paths
        boring_movies = [
            ("Liger", "Liger.jpg"),
            ("Bramhostavam", "Bramhostavam.jpg"),
            ("VVR","VVR.jpg")
        ]
        
        st.write("Boring Movies")
        for movie_name, image_path in boring_movies:
            st.subheader(movie_name)
            movie_image = Image.open(image_path)
            st.image(movie_image, use_column_width=True)
    else:
        st.write("Blockbuster")
        pos_r = Image.open("Five_star.jpg")
        st.image(pos_r, use_column_width=False)

        # List of top 5 blockbuster movies and their image paths
        top_movies = [
            ("RRR", "RRR.jpeg"),
            ("Bahubali", "Bahubali.jpeg"),
            ("KGF", "KGF.jpeg"),
            ("Kalki", "Kalki.jpeg"),
            ("Pushpa", "Pushpa.jpeg")
        ]

        # Display the top 5 blockbuster movies with images
        st.write("Top 5 Blockbuster Movies")
        for movie_name, image_path in top_movies:
            st.subheader(movie_name)
            movie_image = Image.open(image_path)
            st.image(movie_image, use_column_width=True)
