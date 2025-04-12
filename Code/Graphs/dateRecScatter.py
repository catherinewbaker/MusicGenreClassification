"""
Purpose: Analyzes and visualizes the distribution of music tracks across recording dates, 
         creating stacked bar plots to show genre distribution over time (1970-2020).

Key Functions:
- top_tracks(daterecorded=False): 
    Loads and preprocesses track data with optional date handling.
    Returns cleaned DataFrame with genre labels.

- plot_date_recorded_distribution(data, feature='track_date_recorded', genre_column='genre_label'): 
    Creates stacked bar visualization showing genre distribution over time.
    Handles five main genres: Rock, Electronic, Hip-Hop, Folk, Pop.

Notes:
- Implements date normalization relative to earliest recording
- Handles missing dates and ensures consistent year range (1970-2020)
"""

import pandas as pd
import matplotlib.pyplot as plt
from initialPreprocessing import top_tracks
import os
import numpy as np
import matplotlib.pyplot as plt
import math

def top_tracks(daterecorded=False):
    track_headers = pd.read_csv('fma_metadata/tracks.csv',nrows=3, header=None)
    new_track_headers = []

    for col in track_headers:
        if not isinstance(track_headers[col].iloc[0],str) :
            new_track_headers.append(track_headers[col].iloc[2])
        else:
            new_track_headers.append(track_headers[col].iloc[0]+"_"+track_headers[col].iloc[1])

    tracks = pd.read_csv('fma_metadata/tracks.csv',skiprows=[0,1,2], header=None)
    tracks.columns = new_track_headers
    genre_info = pd.read_csv('fma_metadata/genres.csv')
    topg_tracks = tracks.dropna(subset=['track_genre_top']).copy()
    topg_tracks = topg_tracks.dropna(subset=['track_title']).copy()

    if daterecorded:
        topg_tracks['track_date_recorded'] = pd.to_datetime(topg_tracks['track_date_recorded'])

        # Calculate the number of days since the first date in the dataset
        min_date =  topg_tracks['track_date_recorded'].min()
        topg_tracks['days_since_first'] = (topg_tracks['track_date_recorded'] - min_date).dt.days
        topg_tracks = topg_tracks.dropna(subset=['track_date_recorded']).copy()
        print("with date"+str(len(topg_tracks)))

    topg_tracks['genre_label'] = topg_tracks['track_genre_top']

    return topg_tracks

def plot_date_recorded_distribution(data, feature='track_date_recorded', genre_column='genre_label'):
    data[feature] = pd.to_datetime(data[feature])

    # Filter data to include only specified genres
    selected_genres = ["Rock", "Electronic", "Hip-Hop", "Folk", "Pop"]
    filtered_data = data[data[genre_column].isin(selected_genres)]

    # Group filtered data by year and genre, and count occurrences
    grouped_data = filtered_data.groupby([filtered_data[feature].dt.year, genre_column])[genre_column].count().unstack(fill_value=0)

    # Reindex the grouped data with a full range of years
    min_year = 1970
    max_year = 2020
    full_year_range = range(min_year, max_year + 1)
    grouped_data = grouped_data.reindex(full_year_range, fill_value=0)

    # Create a color map for genres
    unique_genres = filtered_data[genre_column].unique()
    genre_colors = {genre: plt.cm.tab10(i) for i, genre in enumerate(unique_genres)}

    # Plot stacked bars for each year
    plt.figure(figsize=(14, 6))
    ax = plt.subplot()

    # Calculate cumulative sum of frequencies for each genre within each year
    stacked_data = grouped_data.cumsum(axis=1)

    for genre in grouped_data.columns:
        ax.bar(grouped_data.index, grouped_data[genre], bottom=stacked_data[genre] - grouped_data[genre],
               color=genre_colors[genre], label=genre, alpha=0.7)

    ax.set_title(f'Distribution of {feature} for selected genres (Stacked)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Frequency')
    ax.legend(title=genre_column)

    # Set x-axis ticks to label only the years
    year_starts = pd.date_range(start=f'{min_year}-01-01', end=f'{max_year + 1}-01-01', freq='YS')
    plt.xticks(year_starts, [str(year) for year in range(min_year, max_year + 2)], rotation=45)

    # Set the y-axis limit to accommodate the maximum frequency
    ax.set_ylim(0, 100)  # Adjust as needed

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sample = top_tracks(True)

    # Call the function to plot the distribution
    plot_date_recorded_distribution(sample)