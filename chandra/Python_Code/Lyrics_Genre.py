#__author__ ='Chandra'
from __future__ import division
from nltk.corpus import stopwords
from collections import Counter
from string import whitespace
import nltk, re, pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab as pl





import pandas as pd
dat = pd.read_csv("./rb_adultcontemp_train.csv")
newdat=dat.drop_duplicates(cols=['artist','lyrics_body','year'], take_last=True)
stop = stopwords.words('english')

# Function for splitting lyrics on newline
def newLineSplit(row):
    return str(row['lyrics_body']).split("\n")
#Function for splitting each word in lyrics
def lyricsWords(row):
    return (" ".join(row['newLineSplit']).split())
#Function for removing stopwords
def removeStops(row):
    from nltk.corpus import stopwords
    stopwords =stopwords.words('english')
    mynewtext = [w for w in row['lyricsWordswithStop'] if w not in stopwords]
    return mynewtext
#Function for counting number of lines in song
def numberOfLines(row):
    return len(row['newLineSplit'])
#Function for counting number of characters in lyrics
def numberOfChars(row):
    return  len(''.join(row['newLineSplit']))
#Function for counting number of words in lyrics
def numberOfWords(row):
     return len(row['lyricsWords'])
#Function for counting nubmber of unique words
def numberOfUniqueWords(row):
     return len(set(row['lyricsWords']))
#Function for cunting most common words in a lyric
def mostcommon(row):
    return [word for word, w in Counter(row['lyricsWords']).most_common(1)]

# Returns Part of Speech tag for words in lyrics
def postag(row):
    return nltk.pos_tag(row['lyricsWords'])

# Returns  NOUN count
def nouncount(row):
    counts =  Counter(tag for word,tag in row['posTag'])
    return counts['NN']

# Returns Verb Count
# VBD Verb, past tense  - VBG Verb, gerund or present # VBN Verb, past participle # VBP Verb, non-3rd person singular
# VBZ Verb, 3rd person singular 

def verbcount(row):
    counts =  Counter(tag for word,tag in row['posTag'])
    return (counts['VB']+counts['VBD']+counts['VBG']+counts['VBN']+counts['VBP']+counts['VBZ'])

# #Creating new columns in data frame
newdat.loc[:,'newLineSplit'] = newdat.apply(newLineSplit, axis=1)
newdat.loc[:,'lyricsWordswithStop'] = newdat.apply(lyricsWords,axis=1)
newdat.loc[:,'lyricsWords'] = newdat.apply(removeStops,axis=1)
newdat.loc[:,'numberOfWords']=newdat.apply(numberOfWords,axis=1)
newdat.loc[:,'numberOfLines'] = newdat.apply(numberOfLines,axis=1)
newdat.loc[:,'numberOfChars']= newdat.apply(numberOfChars,axis=1)
newdat.loc[:,'numberOfUniqueWords'] = newdat.apply(numberOfUniqueWords,axis=1)
newdat.loc[:,'mostCommon'] = newdat.apply(mostcommon,axis=1)
newdat.loc[:,'posTag'] = newdat.apply(postag,axis=1)
newdat.loc[:,'nounCount'] = newdat.apply(nouncount,axis=1)
newdat.loc[:,'verbCount'] = newdat.apply(verbcount,axis=1)

#Creating features for machine learning
newdat.loc[:,'nonUniqueWords'] = newdat['numberOfWords'] -newdat['numberOfUniqueWords']
newdat.loc[:,'avgWordLength'] = newdat['numberOfChars'] / newdat['numberOfWords']
newdat.loc[:,'avgLineLength'] = newdat['numberOfChars'] / newdat['numberOfLines']
newdat.loc[:,'avgLineLengthWords'] = newdat['numberOfWords']/newdat['numberOfLines']
newdat.loc[:,'nounDensity']= newdat['nounCount']/newdat['numberOfWords']
newdat.loc[:,'verbDensity'] = newdat['verbCount']/newdat['numberOfWords']
newdat.loc[:,'uniqueWordDensity'] = newdat['numberOfUniqueWords']/newdat['numberOfWords']
newdat.loc[:,'NonuniqueWordDensity'] = newdat['nonUniqueWords']/newdat['numberOfWords']

#Writing Data set with features to a file
newdat.loc[newdat.genre=='adult_contemp', 'Target'] =1
newdat.loc[newdat.genre=='R_and_B', 'Target'] =0

newdat.to_csv("DatasetwithFeatures.csv")


newdat = pd.read_csv("DatasetwithFeatures.csv")
# print type(newdat)
import Logistic
import RandomForest
import knn
import SVCModel
print newdat.shape
print newdat.head()
y = newdat.iloc[:,[25]]
# y = newdat.iloc[:,[26]]
print y.head()
X = newdat.iloc[:,[16,17,18,19,20,21,22,23,24]]
# X = newdat.iloc[:,[17,18,19,20,21,22,23,24,25]]
cols = X.columns
print " --- "
print X.head()
# Logistic.Logist(X,y)
# RandomForest.RandomF(X,y)
# knn.knn(X,y)
# SVCModel.SVCM(X,y)
