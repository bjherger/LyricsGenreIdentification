#!/usr/bin/env python
"""
coding=utf-8
"""
# imports
# *********************************
import csv
import random
import pandas as pd
import sys

# import MusicXMatch
import LyricGrabber
# global variables
# *********************************

random.seed(0)

__author__ = 'bjherger'
__version__ = '1.0'
__email__ = 'b@revupsoftware.com'
__status__ = 'Development'
__maintainer__ = 'bjherger'


# functions
# *********************************

def scrape_lyrics():
    # raw_df = pd.read_csv('billboard_songs.csv', index_col=0)

    dict_reader = csv.DictReader(open('unique_songs.csv'))

    dict_reader = [entry for entry in dict_reader]

    output_list = list()
    for row in dict_reader:
        print row['']
        del row['']

        artist = row['artist'].split(" featuring", 1)[0]
        song = row['song']
        print song
        lyrics = LyricGrabber.get_lyrics(artist=artist, track=song)
        # lyrics_dict = MusicXMatch.get_lyrics(artist='lady gaga', track='just dance')
        row['lyrics'] = lyrics

        output_list.append(row)

    with_lyrics = pd.DataFrame(output_list)
    with_lyrics.to_pickle('with_lyrics.pkl')
    with_lyrics.to_csv('with_lyrics.csv', encoding='utf-8')



def subset_dataframe():
    raw_df = pd.read_csv('song_list.csv',)
    lyrics = pd.read_csv('with_lyrics.csv')



    with_lyrics_df =pd.merge(raw_df, lyrics, how = 'left',
                                    on=['artist', 'song'])

    with_lyrics_df.to_csv('output/songs_full_with_lyrics.csv')

    unique_df= raw_df[['artist', 'song', 'genre']]
    unique_df = unique_df.drop_duplicates()

    unique_df = pd.merge(unique_df, lyrics, how='left', on = ['artist',
                                                              'song'])

    unique_df= unique_df[['artist', 'song', 'genre', 'lyrics']]

    unique_df = unique_df[pd.notnull(unique_df['lyrics'])]

    rows = random.sample(unique_df.index, 10)

    holdout = unique_df.ix[rows]
    train = unique_df.drop(rows)

    holdout.to_csv('output/holdout_unique_tracks.csv')
    train.to_csv('output/unique_tracks.csv')

# main
# *********************************

if __name__ == '__main__':
    # scrape_lyrics()
    subset_dataframe()



