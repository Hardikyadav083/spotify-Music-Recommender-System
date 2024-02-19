import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("D:/HARDIK YADAV/datasets/data.csv")

# Features for normalization
features = ['acousticness', 'danceability', 'duration_ms', 'energy',
            'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
            'speechiness', 'tempo', 'time_signature', 'valence']

# Normalize the features
scaler = MinMaxScaler()
normalize_df = scaler.fit_transform(df[features])

# Compute cosine similarity matrix
cosine = cosine_similarity(normalize_df)

# Create indices Series
indices = pd.Series(df.index, index=df["song_title"]).drop_duplicates()

# Streamlit app
st.title("Song Recommendation App")

# Input for user to select a song
selected_song = st.selectbox("Select a song:", df["song_title"].values)

# Function to generate recommendations
def generate_recommendation(song_title):
    index = indices[song_title]
    similar_songs = sorted(list(enumerate(cosine[index])), key=lambda x: x[1], reverse=True)[1:11]
    recommended_songs = [df['song_title'].iloc[i[0]] for i in similar_songs]
    return recommended_songs

# Display recommendations
if st.button("Generate Recommendations"):
    recommendations = generate_recommendation(selected_song)
    st.subheader("Top 10 Recommended Songs:")
    for i, song in enumerate(recommendations):
        st.write(f"{i + 1}. {song}")
