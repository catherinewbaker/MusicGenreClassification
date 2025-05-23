"""
Purpose: Analyzes the relationship between artists and their associated genres,
         specifically focusing on removing duplicate genre mappings per artist
         to reduce one-to-one mapping bias in the dataset.

Key Functions:
- main(): Creates a dictionary of artist-to-genre mappings and outputs results to
          'artist_genre_mappings.txt'
          
Notes:
- Outputs mappings to a text file for further analysis
- Helps reduce bias in cases where artists are strongly associated with single genres
"""

from initialPreprocessing import gen_Train_and_Test, top_tracks
import pandas as pd
from sklearn.model_selection import train_test_split
from svm import svm
from knn import knn
from nb import nb

if __name__ == "__main__":                                                       
    sample = top_tracks()
    mappings = {}

    # for each artist and track add the artist as a key to the dictionary and add the track title to a list as the value
    for index, row in sample.iterrows():
        artist = row['artist_name']
        genre = row['genre_label']
        
        # Check if the artist is not in the dictionary
        if artist not in mappings:
            # Add the artist as a key with an empty list as the value
            mappings[artist] = []
        mappings[artist].append(genre)

    with open('artist_genre_mappings.txt', 'w', encoding='utf-8') as file:
        for artist, genres in mappings.items():
            file.write(f'{artist}: {genres}\n')

    print(len(mappings))
    