#################################
#Program to calculate perplexity#
#################################

import pandas as pd
import numpy as np
import nltk
import matplotlib
from matplotlib import pyplot as plt
import random
from nltk.corpus import brown,gutenberg
from nltk import bigrams
from nltk import trigrams
from nltk import ngrams
import copy

#################################################
# Generate Seed for uniform splitting and corpus#
#################################################

random.seed(22)

##################
#Gutenberg Corpus#
##################

#Generate Train set for Gutenberg corpus
guten_train = random.sample(list(gutenberg.fileids()), 12)

#Generate Dev set for Gutenberg corpus
guten_dev_1 = [i for i in list(gutenberg.fileids()) if i not in guten_train]
guten_dev = random.sample(guten_dev_1, 3)

#Generate Test set for Gutenberg corpus
guten_test = [j for j in list(guten_dev_1) if j not in guten_dev]
for i, j, k in zip(guten_train, guten_dev, guten_test):
    assert i not in guten_dev
    assert i not in guten_test
    assert j not in guten_train
    assert j not in guten_test
    assert k not in guten_dev
    assert k not in guten_train

#############################
#Generating for Brown Corpus#
#############################

random.seed(22)
brown_train = []
brown_dev = []
brown_test = []
for cat in brown.categories():
    brown_train.append(random.sample(list(brown.fileids(categories = cat)), int((0.6)* len(brown.fileids(categories = cat)))))
brown_train = [x for y in brown_train for x in y]

#Generating Dev set for Brown Corpus
for cat in brown.categories():
    brown_dev_1 = [i for i in brown.fileids(categories = cat) if i not in brown_train]
    brown_dev.append(random.sample(brown_dev_1, int(0.5* len(brown_dev_1))))
brown_dev = [x for y in brown_dev for x in y]

#Generating Test set for Brown Corpus
for cat in brown.categories():
    brown_test_1 = [i for i in brown.fileids(categories = cat) if i not in brown_train if i not in brown_dev]
    brown_test.append(brown_test_1)
brown_test = [x for y in brown_test for x in y]

for i,j,k in zip(brown_train, brown_dev, brown_test):
    assert i not in brown_dev
    assert i not in brown_test
    assert j not in brown_test
    assert j not in brown_train
    assert k not in brown_train
    assert k not in brown_dev

############################
#creating models for corpus#
############################

# Creating unigram models for Gutenberg corpus
unigram_guten_train = nltk.FreqDist([token.lower() for token in gutenberg.words(fileids = guten_train)])
unigram_guten_dev = nltk.FreqDist([token.lower() for token in gutenberg.words(fileids = guten_dev)])
unigram_guten_test = nltk.FreqDist([token.lower() for token in gutenberg.words(fileids = guten_test)])

# Creating unigram models for Brown Corpus
unigram_brown_train = nltk.FreqDist([token.lower() for token in brown.words(fileids = brown_train)])
unigram_brown_dev = nltk.FreqDist([token.lower() for token in brown.words(fileids = brown_dev)])
unigram_brown_test = nltk.FreqDist([token.lower() for token in brown.words(fileids = brown_test)])

# Creating bigram models for Gutenberg corpus
bigram_guten_train = nltk.FreqDist(bigrams([token.lower() for token in gutenberg.words(fileids = guten_train)]))
bigram_guten_dev = nltk.FreqDist(bigrams([token.lower() for token in gutenberg.words(fileids = guten_dev)]))
bigram_guten_test = nltk.FreqDist(bigrams([token.lower() for token in gutenberg.words(fileids = guten_test)]))

# Creating trigram models for Gutenberg corpus
trigram_guten_train = nltk.FreqDist(trigrams([token.lower() for token in gutenberg.words(fileids = guten_train)]))
trigram_guten_dev = nltk.FreqDist(trigrams([token.lower() for token in gutenberg.words(fileids = guten_dev)]))
trigram_guten_test = nltk.FreqDist(trigrams([token.lower() for token in gutenberg.words(fileids = guten_test)]))

# Creating bigram models for Gutenberg corpus
trigram_brown_train = nltk.FreqDist(trigrams([token.lower() for token in brown.words(fileids = brown_train)]))
trigram_brown_dev = nltk.FreqDist(trigrams([token.lower() for token in brown.words(fileids = brown_dev)]))
trigram_brown_test = nltk.FreqDist(trigrams([token.lower() for token in brown.words(fileids = brown_test)]))

#Create a combined Dictionary
dict_tri = trigram_brown_train.copy()
dict_tri.update(trigram_guten_train)
dict_bi = bigram_brown_train.copy()
dict_bi.update(bigram_guten_train)
dict_uni = unigram_brown_train.copy()
dict_uni.update(unigram_guten_train)

##############################
# Bigram Kneser-Ney Smoothing#
##############################

def biKNS(bigram, bigram_count, unigram_count):
    d = 0.75
    pkn = 0
    if bigram_count[bigram] > d and unigram_count[bigram[0]] !=0:
        pkn = (bigram_count[bigram] - d)/ (unigram_count[bigram[0]])
    
    
    value=[]
    key = []
    for keys, values in zip(bigram_count.keys(), bigram_count.values()):
        if keys[0] == bigram[0]:
            key.append(keys)
            value.append(values)
            
    if np.sum(value)!=0:
        lambda_1 = (d/np.sum(value))*(len(key)) 
    else:
        pkn = pkn + 0.75/len(unigram_count)
        return pkn
    
    
    key = []
    for keys, values in zip(bigram_count.keys(), bigram_count.values()):
        if keys[1] == bigram[1]:
            key.append(keys)
            
    pcont =  len(key)/len(bigram_count)
    
    pkn = pkn + lambda_1*pcont
    
    if pkn == 0:
        pkn = d/len(unigram_count)
    
    return pkn
#################################
#Generating Additional libraries#
#################################
trigram_guten_unique = list(dict_tri)
bigram_guten_unique = list(dict_bi)
trigram_unique_values = list(dict_tri.values())
bigram_unique_values = list(dict_bi.values())

bigram_1_2_count ={}
bigram_1_2_value ={}

bigram_2_3_count ={}
bigram_2_3_value ={}

unigram_1_count ={}
unigram_1_value ={}

unigram_2_count ={}
unigram_2_value ={}

for t,k in zip(trigram_guten_unique, trigram_unique_values):
    if t[0:2] in bigram_1_2_count:
        bigram_1_2_count[t[0:2]] += 1
        bigram_1_2_value[t[0:2]] += k
    else:
        bigram_1_2_count[t[0:2]] =1
        bigram_1_2_value[t[0:2]] = k

for t,k in zip(trigram_guten_unique, trigram_unique_values):
    if t[1:3] in bigram_2_3_count:
        bigram_2_3_count[t[1:3]] += 1
        bigram_2_3_value[t[1:3]] += k
    else:
        bigram_2_3_count[t[1:3]] =1
        bigram_2_3_value[t[1:3]] = k
        
for t,k in zip(bigram_guten_unique, bigram_unique_values):
    if t[0] in unigram_1_count:
        unigram_1_count[t[0]] += 1
        unigram_1_value[t[0]] += k
    else:
        unigram_1_count[t[0]] =1
        unigram_1_value[t[0]] = k

for t,k in zip(bigram_guten_unique, bigram_unique_values):
    if t[1] in unigram_2_count:
        unigram_2_count[t[1]] += 1
        unigram_2_value[t[1]] += k
    else:
        unigram_2_count[t[1]] =1
        unigram_2_value[t[1]] = k
        
        
#     bigram_1_2.append(t[0:2])
#     bigram_2_3.append(t[1:3])
# for t in bigram_guten_unique:
#     unigram_1.append(t[0])
#     unigram_2.append(t[1])

############################
#Implementing Trigram Model#
############################

def triKNS(trigram, trigram_count, bigram_count, unigram_count):
    pkn = 0
    d = 0.75
    residue_1, p_lower = bicont(trigram, trigram_count, bigram_count, unigram_count) 
    residue =  0.75/len(unigram_count) + residue_1
    #residue = 0.75/len(unigram_count)
    if trigram_count[trigram] > d :
        #p_lower = sum(value for keys, value in trigram_count.items() if keys[0:2] == trigram[0:2] )
        pkn = (trigram_count[trigram] - d)/ p_lower + residue
        #print(pkn)
    else:
        pkn = residue
    return pkn

##############################
# For bigram count generation#
##############################

def bicont(trigram, trigram_count, bigram_count, unigram_count):
    
    d = 0.75
    
    lower = 0
    upper = 0
    upper_1 = 0
    lower_1 = 0
    count = 0
    lambda_upper = 0
    lambda_lower = 0
    p_lower = 0
   
        
    if trigram[:2] in bigram_1_2_count:
        upper = bigram_1_2_count[trigram[:2]]
        lower = bigram_1_2_value[trigram[:2]]
        p_lower = bigram_1_2_value[trigram[:2]]  
        
    if trigram[1:3] in bigram_2_3_count:
        upper_1 = bigram_2_3_count[trigram[1:3]]
        #if keys[1] == trigram[1]:
            #lower_1 = lower_1 +1
        
   
    if lower !=0 and upper !=0:
        lambda_1 = (d/lower)*(upper)
    else:
        lambda_1 = 0
            
    uni_count = 0
    
        
    if trigram[1] in unigram_1_count:
        lambda_lower = unigram_1_value[trigram[1]]
        lambda_upper = unigram_1_count[trigram[1]]
        
    if trigram[1] in unigram_2_count:
        uni_count = unigram_2_count[trigram[1]]
        #addition
        lower_1 = unigram_2_count[trigram[1]]
            
    if upper_1 > d:
        pcont = (upper_1 - d)/lower_1
    else: 
        pcont = 0
    
    if lambda_lower != 0 and lambda_upper != 0 :
        lambda_2 = (d/lambda_lower)*(lambda_upper)
    else:
        lambda_2 = 0
        
    if uni_count >d :
        puni = (uni_count - d)/len(bigram_count)
    else:
        puni = 0
    
    if lambda_1 !=0:
        pcont = pcont*lambda_1
    
    if lambda_2 != 0 :
        
        pcont = pcont + puni*lambda_2
    
    return pcont, p_lower

########################
#Perplexity Calculation#
########################

x = int(input('Enter 1 for Bigram Model perplexity, or 2 for Trigram model perplexity: '))

if x ==1:
    corpus = int(input('Enter 1 for Brown corpus or 2 for guten corpus: '))
    
    if corpus == 1:
        ite = int(input('Enter the no. of values from test set to compute the perplexity: '))
        prob = [biKNS(i, bigram_brown_train, unigram_brown_train) for i in list(trigram_brown_test.keys())[:ite]]
        print('Perplexity = ',np.exp((1/ite)*(-(np.sum([np.log(prob) for prob in prob])))))


    if corpus == 2:
        ite = int(input('Enter the no. of values from test set to compute the perplexity: '))
        prob = [biKNS(i, bigram_guten_train, unigram_guten_train) for i in list(trigram_guten_test.keys())[:ite]]
        print('Perplexity = ',np.exp((1/ite)*(-(np.sum([np.log(prob) for prob in prob])))))



if x==2:
    corpus = int(input('Enter 1 for Brown corpus or 2 for guten corpus: '))

    if corpus == 1:
        ite = int(input('Enter the no. of values from test set to compute the perplexity: '))
        prob = [triKNS(i, trigram_brown_train, bigram_brown_train, unigram_brown_train) for i in list(trigram_brown_test.keys())[:ite]]
        print('Perplexity = ',np.exp((1/ite)*(-(np.sum([np.log(prob) for prob in prob])))))

    if corpus == 2:
        ite = int(input('Enter the no. of values from test set to compute the perplexity: '))
        prob = [triKNS(i, trigram_guten_train, bigram_guten_train, unigram_guten_train) for i in list(trigram_guten_test.keys())[:ite]]
        print('Perplexity = ',np.exp((1/ite)*(-(np.sum([np.log(prob) for prob in prob])))))
        
