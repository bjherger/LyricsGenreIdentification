#!/usr/bin/env python
"""
coding=utf-8
"""
# imports
# *********************************

# global variables
# *********************************

import nltk
import collections
import numpy as np
import pandas as pd
import bhUtilities
import datetime


__author__ = 'bjherger'
__version__ = '1.0'
__email__ = 'b@revupsoftware.com'
__status__ = 'Development'
__maintainer__ = 'bjherger'


# functions
# *********************************

def clean_df(df):

    # remove rows
    df = df[pd.notnull(df['artist'])]
    df = df[pd.notnull(df['song'])]



    # normalize lyrics
    df["lyrics_body"] = df["lyrics_body"].apply(lambda x: x.lower())
    # has lyrics or not
    df["has_lyrics"] = df["lyrics_body"]
    df["has_lyrics"] = df["has_lyrics"].apply(
        lambda lyrics: 1 if type(lyrics) == str else 0)

    # number of characters
    df["lyrics_num_char"] = df["lyrics_body"]
    df["lyrics_num_char"] = df["lyrics_num_char"].apply(
        lambda lyrics: len(lyrics) if type(lyrics) == str else np.nan)

    # number of unique_words
    df["lyrics_num_words"] = df["lyrics_body"]
    df["lyrics_num_unique_words"] = df["lyrics_num_words"].apply(
        lambda lyrics: len(set(lyrics.split())) if type(
            lyrics) == str else np.nan)

    # number of words
    df["lyrics_num_words"] = df["lyrics_body"]
    df["lyrics_num_words"] = df["lyrics_num_words"].apply(
        lambda lyrics: len(lyrics.split()) if type(lyrics) == str else np.nan)


    # number of lines
    df["lyrics_num_lines"] = df["lyrics_body"]
    df["lyrics_num_lines"] = df["lyrics_num_lines"].apply(
        lambda lyrics: len(lyrics.split("\n")) if type(
            lyrics) == str else np.nan)

    # most common word
    df["lyrics_most_common_words"] = df["lyrics_body"]
    df["lyrics_most_common_words"] = df["lyrics_most_common_words"].apply(
        lambda lyrics: zip(*collections.Counter(lyrics.split()).most_common())[
            0] if type(lyrics) == str else np.nan)

    # average word length in chars
    df["lyrics_avg_word_len_in_chars"] = df["lyrics_num_char"] / df[
        "lyrics_num_words"]

    # average line length
    df["lyrics_avg_line_len_in_chars"] = df["lyrics_num_char"] / df[
        "lyrics_num_lines"]

    # average line length
    df["lyrics_avg_line_len_in_words"] = df["lyrics_num_words"] / df[
        "lyrics_num_lines"]

    # print df[["lyrics_num_char", "lyrics_num_words", "lyrics_num_lines", "lyrics_most_common_words", "lyrics_avg_word_len_in_chars", "lyrics_avg_line_len_in_chars", "lyrics_avg_line_len_in_words"]]
    # sys.exit()
    return df


def compute_stopword_count(input_string_list):
    stopword_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours',
                     'ourselves', 'you', 'your', 'yours',
                     'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                     'she', 'her', 'hers',
                     'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                     'theirs', 'themselves',
                     'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                     'those', 'am', 'is', 'are',
                     'was', 'were', 'be', 'been', 'being', 'have', 'has',
                     'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
                     'or', 'because', 'as', 'until',
                     'while', 'of', 'at', 'by', 'for', 'with', 'about',
                     'against', 'between', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below',
                     'to', 'from', 'up', 'down',
                     'in', 'out', 'on', 'off', 'over', 'under', 'again',
                     'further', 'then', 'once', 'here',
                     'there', 'when', 'where', 'why', 'how', 'all', 'any',
                     'both', 'each', 'few', 'more',
                     'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                     'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
                     'don', 'should', 'now']

    input_counter = collections.Counter(input_string_list)

    words_count = 0
    for word in stopword_list:
        words_count += input_counter.get(word, 0)

    return words_count


def compute_cuss_count(input_string_list):
    stopword_list = ['fuck', 'fucker', 'fucking', 'fucks', 'nigga', 'niggas',
                     'nigger', 'shit', 'bitch', 'bitches', 'cunt', 'ass',
                     'asses', 'piss', 'cocksucker', 'motherfucker', 'tits',
                     'dick', 'dicks', 'pussy', 'pussies']

    input_counter = collections.Counter(input_string_list)

    words_count = 0
    for word in stopword_list:
        words_count += input_counter.get(word, 0)

    return words_count


def clean_df2(df):

    df = df.drop_duplicates(subset=['artist', 'song'])

    df = df[pd.notnull(df['artist'])]
    df = df[pd.notnull(df['song'])]

    # Normalize text
    df['lyrics_body'] = df['lyrics_body'].apply(
        lambda x: x.decode('ascii', errors='ignore'))

    # Get words
    df['words_list'] = df['lyrics_body'].apply(bhUtilities.splitAndCleanString)
    df['word_set'] = df['words_list'].apply(set)
    df['nltk_tokenizer'] = df['lyrics_body'].apply(nltk.tokenize.word_tokenize)
    df['nltk_pos'] = df['nltk_tokenizer'].apply(nltk.pos_tag)

    # Raw counts
    df['num_words'] = df['words_list'].apply(len).apply(float)
    df['num_unique_words'] = df['word_set'].apply(len).apply(float)
    df['number_of_lines'] = df['lyrics_body'].apply(
        lambda x: len(x.split('\n'))).apply(float)
    df['num_stop_words'] = df['word_set'].apply(compute_stopword_count).apply(
        float)
    df['num_cuss_words'] = df['word_set'].apply(compute_cuss_count).apply(
        float)

    # Part of speech counts
    df['pos_list'] = df['nltk_pos'].apply(
        lambda x: [value for key, value in x])
    df['pos_counter'] = df['pos_list'].apply(collections.Counter)
    df['pos_num_nouns'] = df['pos_counter'].apply(
        lambda x: x.get('NN', 0) + x.get('NNS', 0) + x.get('NNP', 0)
                  + x.get('NNPS', 0)).apply(float)
    df['pos_num_verbs'] = df['pos_counter'].apply(
        lambda x: x.get('VB', 0) + x.get('VBD', 0) + x.get('VBG', 0) +
                  x.get('VBN', 0) + x.get('VBP', 0) +
                  x.get('VBZ', 0)).apply(float)

    # Densities
    df['density_unique_word'] = df['num_unique_words'] / df['num_words']
    df['density_noun'] = df['pos_num_nouns'] / df['num_words']
    df['density_verb'] = df['pos_num_verbs'] / df['num_words']

    df['density_stop_word'] = df['num_stop_words'] / df['num_words']
    df['density_cuss_words'] = df['num_cuss_words'] / df['num_words']

    df['avg_words_per_line'] = df['num_words'] / df['number_of_lines']
    # print df
    print df.columns

    return df


def main():
    raw_df = pd.read_csv('../../data/raw/rb_adultcontemp_train.csv')
    normalized = clean_df2(raw_df)
    normalized.to_csv('train_with_features.csv')
    print 'hello world'

# main
# *********************************

if __name__ == '__main__':
    print datetime.datetime.now()
    main()
    print datetime.datetime.now()


