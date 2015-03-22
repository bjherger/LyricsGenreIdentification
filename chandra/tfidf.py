from __future__ import division
import nltk, re, pprint
import pandas as pd
from nltk.stem.porter import *
import collections
import math
import itertools
from collections import Counter
import random

def low(row):
    a = re.sub(r'[^a-zA-Z\'\t\n\s\r]','', row['lyrics_body'])
    return a.lower()

def newLineSplit(row):
    return str(row['lyrics_body']).split("\n")

def lyricsWords(row):
    return (" ".join(row['newLineSplit']).split())


def removeStops(row):
    from nltk.corpus import stopwords
    stopwords =stopwords.words('english')
    mynewtext = [w for w in row['lyricsWordswithStop'] if w not in stopwords]
    return mynewtext

def stem(row):

    stemmer = PorterStemmer()
    wor = []
    for word in row['lyricsWords']:
        wor.append(stemmer.stem(word))
    return wor


def prePare(df):
    newdat=df.drop_duplicates(cols=['artist','lyrics_body','year'], take_last=True)
    newdat['lyrics_body'] = newdat.apply(low,axis=1)
    newdat['newLineSplit'] = newdat.apply(newLineSplit, axis=1)
    newdat['lyricsWordswithStop'] = newdat.apply(lyricsWords,axis=1)
    newdat['lyricsWords'] = newdat.apply(removeStops,axis=1)
    newdat['stemWords'] = newdat.apply(stem,axis=1)
    return newdat

def TermFreq(Genre):
    Genre_len = collections.defaultdict(list)
    termFrequency = collections.defaultdict(list)
    for key,val in Genre.items():
        wor = list(itertools.chain.from_iterable(val))
        Genre_len[key] = len(list(itertools.chain.from_iterable(val)))
        termFrequency[key] = Counter(wor)
    return Genre_len, termFrequency

def DocFreq(Uniq,TermFreq,totalDocs):
    docFrequency = collections.defaultdict(list)
    for word in Uniq:
        count = 0
        for key in TermFreq:
            if(word in TermFreq[key].keys()):
                count +=1
        docFrequency[word] = math.log((totalDocs+1) / (count+1))
    return docFrequency

def TFIDF(termFrequency,docFrequency,uniq):
    tfidf = collections.defaultdict(list)
    for key in termFrequency:
        tfidf[key] = {}
        for word in uniq:
            tfidf[key] [word] = termFrequency[key][word]* docFrequency[word]
    return tfidf


def tfIDFScore(row,tfidf):
    scores=[]
    for key in tfidf.keys():
        sumsc = 0
        for word in row['stemWords']:
            try:
                # print word, " -- ", key, " --- " ,tfidf[key][word]
                sumsc += tfidf[key][word]
            except:
                pass
        scores.append(sumsc)
    return scores

def predict(row,tfidf):
    max_value = max(row['scores'])
    max_index = row['scores'].index(max_value)
    return  tfidf.keys()[max_index]

def accuracyReturn(row):
    if(row['genre']==row['Prediction']):
        return 1
    else:
        return 0


def TrainData(newdat):
    Genre = collections.defaultdict(list)
    termF = collections.defaultdict(list)
    allwords=[]

    for index, row in newdat.iterrows():
        Genre[newdat.ix[index]['genre']].append(newdat.ix[index]['stemWords'])
        allwords.append(newdat.ix[index]['stemWords'])

    uniq = set(list(itertools.chain(*allwords)))

    Genre_len, termFrequency = TermFreq(Genre)
    totalDocs = len(termFrequency.keys())

    for key, val  in termFrequency.items():
        for i in val:
            val[i] = (val[i]/Genre_len[key])

    docFrequency=DocFreq(uniq,termFrequency,totalDocs)
    tfIDF =  TFIDF(termFrequency,docFrequency,uniq)
    return tfIDF



def testTrainSplit(df):
    testloc = random.sample(range(df.shape[0]),  math.trunc((df.shape[0])*0.1))
    trainloc = list(set(range(df.shape[0])) -set(testloc))
    test = df.iloc[testloc,:]
    train = df.iloc[trainloc,:]
    return test,train

def accuracy(dataframe,tfidict):
    dataframe['scores']=dataframe.apply(tfIDFScore,args=(tfidict,),axis=1)
    dataframe['Prediction']=dataframe.apply(predict,args=(tfidict,),axis=1)
    dataframe['accuracy'] = dataframe.apply(accuracyReturn,axis=1)
    accura = sum(dataframe['accuracy'])/len(dataframe)
    return accura




def crossvalidation(dataframe,noFolds,tfIDF):
    for i in range(noFolds):
        test,train = testTrainSplit(dataframe)
        tfid = TrainData(train)
        print "Test Set accuracy", accuracy(test,tfid)
        # print "Training Set accuracy ",accuracy(train,tfid)


def main():
    dat = pd.read_csv("train.csv")
    newdat = prePare(dat)
    tfIDF= TrainData(newdat)
    print "Accuracy on Complete Training  Data Set", accuracy(newdat,tfIDF)
    print "Cross Validation "
    crossvalidation(newdat,5,tfIDF)

main()


