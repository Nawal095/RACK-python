# importing all necessary modules for "ConnectionManager" class
import sqlite3

# importing all necessary modules for "CosineSimilarityMeasure" class
import traceback
import logging
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action = 'ignore')
import gensim
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from collections import Counter
import math

# importing all necessary modules for "TokenStemmer" class
import nltk
import string
from nltk.stem import PorterStemmer

# importing all necessary modules and resources for "StopWordRemover" class
from nltk.corpus import stopwords
nltk.download('stopwords')

# importing all necessary modules for "LexicalSimilarityProvider" class
import re

# importing all necessary modules for Driver class
import sys

# Static Data
MAXAPI = 10
DELTA1 = 10
DELTA2 = 10
alpha = 0.325
beta = 0.575
psi = 0.10
gamma = 0
GOLDSET_SIZE = 10
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
stemmer = PorterStemmer()


## Stop Word Remover
class StopWordRemover:

    def getRefinedSentence(self, text):
        # punctuation removed
        new_string = text.translate(str.maketrans('', '', string.punctuation))
        # stop word removed
        text_tokens = word_tokenize(new_string)
        tokens_without_sw = [word.strip() for word in text_tokens if not word in stopwords.words()]
        return ' '.join(tokens_without_sw)
    
    def removeStopWords(self, text_tokens):
        tokens_without_sw = [word.strip() for word in text_tokens if not word in stopwords.words()]
        return tokens_without_sw

## Token Stemmer
class TokenStemmer:

    def performStemming(self, args):
        if type(args) == list:
            return [stemmer.stem(token) for token in args]
        elif type(args) == str:
            return stemmer.stem(args[0])


## Item Sorter
class ItemSorter:
    
    def sortHashMapInt(self, wordMap):
        return {k: v for k, v in sorted(wordMap.items(), key=lambda item: item[1])}
    
    def sortHashMapDouble(self, wordMap):
        return {k: v for k, v in sorted(wordMap.items(), key=lambda item: item[1])}


## API Token
class APIToken:

    def __init__(self):
        self.token = ''
        self.KACScore = 0
        self.KKCScore = 0
        self.KPACScore = 0
        self.totalScore = 0

## Cosine Similarity Measure
class CosineSimilarityMeasure:
    
    def __init__(self, first, second):
        self.sqliteConnection = None
        self.first = first
        self.second = second
    
    def getCosineSimilarityScore(self):
        if type(self.first) == str and type(self.second) == str:
            return model.similarity(self.first, self.second)
        
        elif type(self.first) == list and type(self.second) == list:
            c1 = Counter(self.first)
            c2 = Counter(self.second)

            terms = set(c1).union(c2)
            dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
            magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
            magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
            return dotprod / (magA * magB)
        
        else:
            return -1


## Connection Manager
class ConnectionManager:
    
    def __init__(self):
        self.sqliteConnection = None
    
    def getConnection(self):
        try:
            self.sqliteConnection = sqlite3.connect('RACK-EMSE.db')
            cursor = self.sqliteConnection.cursor()
            print("Database Connected to SQLite")

            sqlite_select_Query = "select sqlite_version();"
            cursor.execute(sqlite_select_Query)
            record = cursor.fetchall()
            # print("SQLite Database Version is: ", record)
            cursor.close()
            
            return self.sqliteConnection

        except sqlite3.Error as error:
            print("Error while connecting to sqlite", error)
            
            return None


## Adjacency Score Provider
class AdjacencyScoreProvider:
    
    adjacencymap = {}
    
    def __init__(self, queryTerms):
        self.queryTerms = queryTerms
        self.keys = []
        self.simscores = []
    
    def collectAdjacentTerms(self):
        try:
            conn = ConnectionManager()
            sqliteConnection = conn.getConnection()
            
            if sqliteConnection != None:
                for key in self.queryTerms:
                    cursor = sqliteConnection.cursor()

                    sqlite_select_Query = "select distinct Token from TextToken where EntryID in (select EntryID from TextToken where Token='" + key + "') and Token!='" + key + "'"
                    cursor.execute(sqlite_select_Query)
                    record = cursor.fetchall()                    
                    
                    adjacent = []
                    
                    for rec in record:
                        adjacent.append(rec[0])
                        
                    self.adjacencymap[key] = adjacent
                
                self.keys = list(self.adjacencymap.keys())
                sqliteConnection.close()
                
        except Exception as e:
            logging.error(traceback.format_exc())
            
    def collectAdjacencyScores(self):
        self.keys = list(self.adjacencymap.keys())
        dimension = len(self.keys)
        
        for i in range(dimension):
            self.simscores.append([-1]*dimension) 
        
        for i in range(len(self.keys)):
            first = self.keys[i]
            for j in range(len(self.keys)):
                if j > i:
                    second = self.keys[j]
                    cos = CosineSimilarityMeasure(first, second)
                    simscore = cos.getCosineSimilarityScore()
                    self.simscores[i][j] = simscore
                    self.simscores[j][i] = simscore
                                
        return self.simscores
    
    def getQueryTermAdjacencyScores(self):
        self.collectAdjacentTerms()
        print(self.collectAdjacencyScores())

## Lexical Similarity Provider
class LexicalSimilarityProvider:
    
    def __init__(self, queryTerms, candidates):
        self.queryTerms = queryTerms
        self.candidates = candidates
        self.simScoreMap = {}
        
    def decomposeCamelCase(self, token):
        return re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', token)).split()
    
    def clearTheTokens(self, tokenParts):
        refined = StopWordRemover.removeStopWords(tokenParts)
        stemmed = TokenStemmer.performStemming(refined)
        return stemmed
    
    def normalizeAPIToken(self, apiToken):
        # normalize the API token into granular tokens
        decomposed = self.decomposeCamelCase(apiToken)
        normalized = self.clearTheTokens(decomposed)
        return normalized
    
    def getLexicalSimilarityScores(self):
        for apiName in self.candidates:
            normalizedTokens = self.normalizeAPIToken(apiName)
            cosMeasure = CosineSimilarityMeasure(normalizedTokens, self.candidates)
            simScore = cosMeasure.getCosineSimilarityScore()
            if apiName not in self.simScoreMap.keys():
                self.simScoreMap[apiName] = simScore
            return self.simScoreMap


## Relevant API Collector
class RelevantAPICollector:
    
    def __init__(self, queryTerms):
        self.queryTerms = queryTerms
        
    def collectAPIsforQuery(self):
        tokenmap = {}
        try:
            conn = ConnectionManager()
            sqliteConnection = conn.getConnection()
            
            if sqliteConnection != None:
                for texttoken in self.queryTerms:
                    cursor = sqliteConnection.cursor()

                    sqlite_select_Query = "select ct.Token from CodeToken as ct, TextToken as tt where ct.EntryID=tt.EntryID and tt.Token='"+ texttoken + "' group by ct.Token order by count(*) desc limit "+ str(DELTA1);
                    cursor.execute(sqlite_select_Query)
                    results = cursor.fetchall()
                    
                    apis = []
                    
                    for res in results:
                        apis.append(res[0])
                        
                    tokenmap[texttoken] = apis
                
                sqliteConnection.close()
                return tokenmap
                    
        except Exception as e:
            logging.error(traceback.format_exc())
            return None


## Coocurrence Score Provider
class CoocurrenceScoreProvider:
    
    def __init__(self, queryTerms):
        self.queryTerms = queryTerms
        self.keys = list(set(queryTerms))
        self.coocAPIMap = {}
        self.coocScoreMap = {}
        
    def getKeyPairs(self):
        temp = []
        for i in range(len(self.keys)):
            first = self.keys[i]
            for j in range(i + 1,  len(self.keys)):
                second = self.keys[j]
                keypair = first + "-" + second
                temp.append(keypair)
        return temp
    
    def collectCoocAPIs(self, keypairs):
        try:
            conn = ConnectionManager()
            sqliteConnection = conn.getConnection()
            
            if sqliteConnection != None:
                for keypair in keypairs:
                    cursor = sqliteConnection.cursor()
                    
                    parts = keypair.split("-")
                    first = parts[0]
                    second = parts[1]

                    sqlite_select_Query = "select Token from CodeToken where EntryID in(select EntryID from TextToken where Token='"+ first + "' intersect select EntryID from TextToken where Token='"+ second + "') group by Token order by count(*) desc limit " + str(DELTA1);
                    
                    cursor.execute(sqlite_select_Query)
                    results = cursor.fetchall()
                    
                    temp = []
                    
                    for res in results:
                        temp.append(res[0])
                        
                    self.coocAPIMap[keypair] = temp
                
                sqliteConnection.close()
                return self.coocAPIMap
                    
        except Exception as e:
            logging.error(traceback.format_exc())
            return None
        
    def generateCoocScores(self):
        keySet = list(self.coocAPIMap.keys())
        for keypair in keySet:
            apis = self.coocAPIMap[keypair]
            length = len(apis)
            for i in range(length):
                score = 1 - i / length
                api = apis[i]
                if api in self.coocScoreMap.keys():
                    newScore = self.coocScoreMap[api] + score;
                    self.coocScoreMap[api] = newScore
                else:
                    self.coocScoreMap[api] = score
                    
    def normalizeScores(self):
        maxScore = 0
        for api in list(self.coocScoreMap.keys()):
            score = self.coocScoreMap[api]
            if score > maxScore:
                maxScore = score
        
        for api in list(self.coocScoreMap.keys()):
            nScore = self.coocScoreMap[api] / maxScore
            self.coocScoreMap[api] = nScore
            
    def getCoocScores(self):
        keypairs = self.getKeyPairs()
        self.collectCoocAPIs(keypairs)
        self.generateCoocScores()
        self.normalizeScores()
        return self.coocScoreMap

## Code Token Provider
class CodeTokenProvider:
    
    def __init__(self, query):
        self.query = query
        self.tokenScoreMap = {}
        self.stemmedQuery = []
        self.KACMap = {}
        self.KPACMap = {}
        self.KKCMap = {}
        
    def decomposeQueryTerms(self):
        tempQuery = self.query.lower()
        swr = StopWordRemover()
        tempQuery = swr.getRefinedSentence(tempQuery)
        tokens = word_tokenize(tempQuery)
        refined = swr.removeStopWords(tokens)
        
        ts = TokenStemmer()
        stemmed = ts.performStemming(refined)
        stemmedQuery = []
        for token in stemmed:
            if token.isnumeric() or len(token) <= 0:
                continue
            else:
                stemmedQuery.append(token)
                
        self.stemmedQuery = stemmedQuery
        return stemmedQuery

    def collectTokenScores(self, queryTerms):
        adjacent = AdjacencyScoreProvider(queryTerms)
        adjacent.collectAdjacentTerms() # self.adjacencymap was created
        simscores = adjacent.collectAdjacencyScores()
        keys = list(adjacent.keys)
        
        collector = RelevantAPICollector(queryTerms)
        tokenmap = collector.collectAPIsforQuery()
        
        self.tokenScoreMap = {}
        
        # KAC scores
        self.addAssociationFrequencyScores(tokenmap)
        # KKC scores
        self.addTokenSimilarityScores(keys, simscores, tokenmap)
        # KPAC scores
        self.addDirectCoocScores()
        # add the textual similarity scores
        self.addExtraLayerScoreComputation()
        
    def collectTokenScoresKAC(self, queryTerms):
        # collecting token scores based on KAC
        collector = RelevantAPICollector(queryTerms)
        tokenmap = collector.collectAPIsforQuery()
        
        self.tokenScoreMap = {}
        
        # now add the scores
        self.addAssociationFrequencyScores(tokenmap)
        
        
    def collectTokenScoresKKC(self, queryTerms):
        # collecting scores based on AAC
        adjacent = AdjacencyScoreProvider(queryTerms)
        # adjacency scores
        adjacent.collectAdjacentTerms()
        simscores = adjacent.collectAdjacencyScores()
        keys = list(adjacent.keys) # keys from queries
        
        collector = RelevantAPICollector(queryTerms)
        tokenmap = collector.collectAPIsforQuery()
        
        self.tokenScoreMap = {}
        
        # now add the scores
        self.addTokenSimilarityScores(keys, simscores, tokenmap)
        
    def collectTokenScoresKPAC(self, tokenmap):
        self.tokenScoreMap = {}
        self.addDirectCoocScores()
        
    def addTokenSimilarityScores(self, keys, simscores, tokenmap):
        for i in range(len(keys)):
            first = keys[i]
            firstapi = tokenmap[first]
            for j in range(i+1, len(keys)):
                second = keys[j]
                secondapi = tokenmap[second]
                common = self.intersect(firstapi, secondapi)
                simscore = simscores[i][j]
                
                if simscore > gamma:
                    for token in common:
                        if token in self.tokenScoreMap.keys():
                            newOldScore = self.tokenScoreMap[token] + simscore
                            self.tokenScoreMap[token] = newOldScore
                        else:
                            self.tokenScoreMap[token] = simscore

                        # adding to the extra map
                        if token in self.KKCMap.keys():
                            newOldScore = self.KKCMap[token] + simscore
                            self.KKCMap[token] = newOldScore
                        else:
                            self.KKCMap[token] = simscore
        
    def addAssociationFrequencyScores(self, tokenmap):
        # association frequency score between text token and code token
        for key in list(tokenmap.keys()):
            apis = tokenmap[key]
            
            length = len(apis)
            
            for i in range(len(apis)):
                
                # now determine the score
                score = 1 - i / length;

                # add the weight
                # score = score * StaticData.alpha
                
                api = apis[i]
                # now check the score for each API
                # add the score to the map
                if api in self.tokenScoreMap.keys():
                    newScore = self.tokenScoreMap[api] + score
                    self.tokenScoreMap[api] = newScore
                else:
                    self.tokenScoreMap[api] = score
                    
                # adding scores to the extra map
                if api in self.KACMap.keys():
                    newScore = self.KACMap[api] + score
                    self.KACMap[api] = newScore
                else:
                    self.KACMap[api] = score
        
    def addDirectCoocScores(self):
        # adding direct cooccurrence scores
        coocProvider = CoocurrenceScoreProvider(self.stemmedQuery)
        coocScoreMap = coocProvider.getCoocScores()
        for apiKey in list(coocScoreMap.keys()):
            
            coocScore = coocScoreMap[apiKey]
            
            # add the weight
            # coocScore = coocScore * StaticData.beta
            
            # adding to the token map
            # adding to the token score map
            if apiKey in list(self.tokenScoreMap.keys()):
                newScore = self.tokenScoreMap[apiKey] + coocScore
                self.tokenScoreMap[apiKey] = newScore
            else:
                self.tokenScoreMap[apiKey] = coocScore
                
            # adding to the extra map
            if apiKey in list(self.KPACMap.keys()):
                newScore = self.KPACMap[apiKey] + coocScore
                self.KPACMap[apiKey] = newScore
            else:
                self.KPACMap[apiKey] = coocScore
        
    def intersect(self, s1, s2):
        # intersecting the two sets / list of items
        common = [value for value in s1 if value in s2]
        return common
        
    def rankAPIElements(self, scoreMap = None): # overloaded method
        # rank the API names
        isort = ItemSorter()
        if scoreMap == None:
            sorted = isort.sortHashMapDouble(self.tokenScoreMap)
        else:
            sorted = isort.sortHashMapDouble(scoreMap)
            
        rankedAPIs = []
        for k, v in sorted.items():
            rankedAPIs.append(k)
        
        topRanked = [value.strip() for value in rankedAPIs if value.strip() != '']
        
        # returning the ranked APIs
        if scoreMap == None:
            return rankedAPIs
        else:
            return topRanked[:MAXAPI]
        
    def addExtraLayerScoreComputation(self):
        kacs = self.rankAPIElements(self.KACMap)
        kacScoreMap = self.getNormScore(kacs)
        
        kpacs = self.rankAPIElements(self.KPACMap)
        kpacScoreMap = self.getNormScore(kpacs)
        
        kkcs = self.rankAPIElements(self.KKCMap)
        kkcScoreMap = self.getNormScore(kkcs)
        
        self.addCombinedRankingsV2(kacScoreMap, kpacScoreMap, kkcScoreMap, alpha, beta, psi)
        
    def addCombinedRankings(self, kacMap, kpacMap, kkcMap, alpha1, beta1, psi1):
        # get the combined rankings
        self.tokenScoreMap = {}
        # HashMap<String, Double> tokenScoreMap = new HashMap<>()
        for key in kacMap.keys():
            score = kacMap[key]
            score = score * alpha1
            if key in list(self.tokenScoreMap.keys()):
                newScore = self.tokenScoreMap[key] + score
                self.tokenScoreMap[key] = newScore
            else:
                self.tokenScoreMap[key] = score
                
        for key in list(kpacMap.keys()):
            score = kpacMap[key]
            score = score * beta1
            if key in list(self.tokenScoreMap.keys()):
                newScore = self.tokenScoreMap[key] + score
                self.tokenScoreMap[key] =  newScore
            else:
                self.tokenScoreMap[key] = score                
                
        for key in list(kkcMap.keys()):
            score = kkcMap[key]
            score = score * psi1
            if key in list(self.tokenScoreMap.keys()):
                newScore = self.tokenScoreMap[key] + score
                self.tokenScoreMap[key] = newScore
            else:
                self.tokenScoreMap[key] = score
                
        # return rankAPIElements(tokenScoreMap);
    
    def addCombinedRankingsV2(self, kacMap, kpacMap, kkcMap, alpha1, beta1, psi1):
        # get the combined rankings
        self.tokenScoreMap = {}
        # HashMap<String, Double> tokenScoreMap = new HashMap<>()
        for key in kacMap.keys():
            score = kacMap[key]
            score = score * alpha1
            if key in list(self.tokenScoreMap.keys()):
                newScore = max(self.tokenScoreMap[key] , score)
                self.tokenScoreMap[key] = newScore
            else:
                self.tokenScoreMap[key] = score
                
        for key in list(kpacMap.keys()):
            score = kpacMap[key]
            score = score * beta1
            if key in list(self.tokenScoreMap.keys()):
                newScore = max(self.tokenScoreMap[key] , score)
                self.tokenScoreMap[key] =  newScore
            else:
                self.tokenScoreMap[key] = score                
                
        for key in list(kkcMap.keys()):
            score = kkcMap[key]
            score = score * psi1
            if key in list(self.tokenScoreMap.keys()):
                newScore = max(self.tokenScoreMap[key] , score)
                self.tokenScoreMap[key] = newScore
            else:
                self.tokenScoreMap[key] = score
                
        # return rankAPIElements(tokenScoreMap)
        
    def getNormScore(self, apis):
        tempMap = {}
        index = 0
        for api in apis:
            index = index + 1
            score = 1 - index / len(apis)
            tempMap[api] = score
            # index++
        
        return tempMap
    
    def recommendRelevantAPIs(self, key):
        # recommend API names for a query
        queryTerms = self.decomposeQueryTerms()
        # collecting scores
        if len(key) > 0:
            if key[0] == "KAC":
                self.collectTokenScoresKAC(queryTerms)
            elif key[0] == "KPAC":
                self.collectTokenScoresKPAC(queryTerms)
            elif key[0] == "KKC":
                self.collectTokenScoresKKC(queryTerms)
            elif key[0] == "all":
                self.collectTokenScores(queryTerms)
            else:
                self.collectTokenScores(queryTerms)
        else:
            self.collectTokenScores(queryTerms)
            
        apis = self.rankAPIElements()
        
        # now normalize the component scores
        self.KACMap = self.normalizeMapScores(self.KACMap)
        self.KPACMap = self.normalizeMapScores(self.KPACMap)
        self.KKCMap = self.normalizeMapScores(self.KKCMap);

        # now demonstrate the API
        resultAPIs = []
        suggestedResults = []

        for api in apis:
            if api.strip() == '':
                continue;
            
            # adding the results with scores
            atoken = APIToken()
            atoken.token = api
            if api in self.KACMap.keys():
                atoken.KACScore = self.KACMap[api]
            if api in self.KPACMap.keys():
                atoken.KPACScore = self.KPACMap[api]
            if api in self.KKCMap.keys():
                atoken.KKCScore = self.KKCMap[api]
            if api in self.tokenScoreMap.keys():
                atoken.totalScore = self.tokenScoreMap[api]
                
            suggestedResults.append(atoken)
            resultAPIs.append(api)
            
            if len(resultAPIs) == MAXAPI:
                break
            
        # showAPIs(apis)
        return suggestedResults
    
    def showAPIs(self, apis):
        print(apis)
        
    def normalizeMapScores(self, tempScoreMap):
        maxScore = 0
        for api in tempScoreMap.keys():
            score = tempScoreMap[api]
            if score > maxScore:
                maxScore = score
                
        for api in tempScoreMap.keys():
            myscore = tempScoreMap[api]
            normScore = myscore / maxScore
            tempScoreMap[api] = normScore

        return tempScoreMap
    
    def discardDuplicates(self, results):
        return list(set(results))

### driver for ConnectionManager
# conn = ConnectionManager()
# sqliteConnection = conn.getConnection()
# if sqliteConnection:
#     sqliteConnection.close()

### driver for AdjacencyScoreProvider
# provider = AdjacencyScoreProvider(['extract','method','class'])
# provider.getQueryTermAdjacencyScores()

### driver for TokenStemmer
# tokStem = TokenStemmer()
# print(tokStem.performStemming(['extracting', 'methodify', 'classify']))

### driver for StopWordRemover
# text = "Nick likes to play football, however he is not too fond of tennis."
# swr = StopWordRemover()
# print(swr.getRefinedSentence(text))

### driver for RelevantAPICollector
# apiCollector = RelevantAPICollector(['extract','method','class'])
# print(apiCollector.collectAPIsforQuery())

### driver for CoocurrenceScoreProvider
# queryTerms = []
# queryTerms.append("copi")
# queryTerms.append("file")
# queryTerms.append("jdk")
# print(CoocurrenceScoreProvider(queryTerms).getCoocScores())

### driver for CodeTokenProvider
# query = "How to parse HTML in Java?"
# provider = CodeTokenProvider(query)
# results = provider.recommendRelevantAPIs("all")
# for atoken in results:
#     print(atoken.token + " " + str(atoken.KACScore) + " " + str(atoken.KPACScore) + " " + str(atoken.KKCScore) + " " + str(atoken.totalScore))

if __name__ == "__main__":
    list_of_arguments = sys.argv
    query = list_of_arguments[1]
    provider = CodeTokenProvider(query)
    results = provider.recommendRelevantAPIs("all")
    for atoken in results:
        print(atoken.token + " " + str(atoken.KACScore) + " " + str(atoken.KPACScore) + " " + str(atoken.KKCScore) + " " + str(atoken.totalScore))